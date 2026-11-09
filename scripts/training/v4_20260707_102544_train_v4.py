"""
v4 training driver — full GNS/MeshGraphNet-recipe upgrade of the Hybrid GNN.

Version ID : v4_20260707_102544
Model      : src/models/hybrid_gnn_v4.py  (HybridPhysicsGNNv4)
Data       : datasets/flip_simulations_v3p2_20260506_043045/  (reused, unchanged)

No existing code is modified. Prior drivers (train_improved.py,
v3p2_*_train_unified.py) remain runnable for baseline comparison.

Training-recipe upgrades implemented HERE (model-side upgrades U1-U6 are in
the model file):

  [T1] Training-noise injection (GNS random-walk).
       Gaussian random-walk noise is added to the velocity HISTORY of fluid
       particles (per-step std = noise_std / sqrt(history)), and the frame-t
       positions are drifted consistently (dt * accumulated noise). The
       target delta is computed against the NOISY v_t, so the model learns
       to correct perturbed states — the standard fix for autoregressive
       rollout divergence. Graph CONNECTIVITY is kept from clean positions
       (cached); only edge features are recomputed from noisy positions
       (noise << radius, connectivity change negligible).

  [T2] Dataset normalisation statistics.
       Node/edge/target mean+std computed over the TRAIN split only, then
       installed into the model's buffers (checkpoint-portable).

  [T3] Pushforward (2-step) loss  [Brandstetter et al., MP-PDE].
       With probability --pushforward-prob, the model first takes one
       no-grad step t -> t+1 (its own prediction perturbs the state), the
       graph is rebuilt at the predicted positions, and the supervised loss
       is applied on the second step t+1 -> t+2. Gradients flow only through
       step 2 (detached pushforward), directly training error-correction.

  [T4] Cosine LR schedule (lr -> lr_min over the full run) with optional
       plateau fallback (--sched plateau). Grad-clip 1.0, Adam, optional
       gradient accumulation (--accum-steps) as an effective-batch knob
       (graphs have variable size, so true batching is left out by design).

  [T5] v4 loss = normalised-delta MSE (primary)
              + pos-consistency + momentum + energy  (physical scale,
                same formulas/clamps as ImprovedPhysicsLoss).

Usage:
  python scripts/training/v4_20260707_102544_train_v4.py --tag full
  # ablations
  python scripts/training/v4_20260707_102544_train_v4.py --tag no_noise --noise-std 0
  python scripts/training/v4_20260707_102544_train_v4.py --tag no_push  --pushforward-prob 0
  python scripts/training/v4_20260707_102544_train_v4.py --tag l5      --num-layers 5
  # smoke test (CPU ok)
  python scripts/training/v4_20260707_102544_train_v4.py --tag smoke \
      --limit-sims 2 --limit-seq 6 --epochs 2 --num-layers 2
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4, V4_VERSION_ID

DATASET_ID = "v3p2_20260506_043045"
DATA_DIR = ROOT / "datasets" / f"flip_simulations_{DATASET_ID}"
DT = 0.01


# =============================================================================
# Dataset
# =============================================================================
class V4Dataset(Dataset):
    """
    One sim directory of frame_XXXX.npz files.

    Sample idx covers the transition t=idx -> t+1. Also carries t+2 (when it
    exists) for the pushforward loss, and a velocity history of `history`
    steps (clamped at sequence start).
    """

    def __init__(self, data_dir, radius=0.3, history=3, cache=True):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.history = history
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))
        if len(self.frame_files) < 2:
            raise ValueError(f"Need >=2 frames in {data_dir}")
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.frame_files) - 1

    def _load(self, i):
        d = np.load(self.frame_files[i])
        return (d["positions"].astype(np.float32),
                d["velocities"].astype(np.float32),
                d["particle_types"].astype(np.float32))

    def build_graph(self, positions):
        """Vectorised radius graph. Returns edge_index (2,E), edge_attr (E,4)."""
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if len(pairs) == 0:
            edges = np.array([[0], [0]], dtype=np.int64)
        else:
            edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
        rel = positions[edges[1]] - positions[edges[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        return edges, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        pos_t, vel_t, types = self._load(idx)
        pos_t1, vel_t1, _ = self._load(idx + 1)

        # Velocity history: oldest -> newest, clamped at sequence start.
        hist = [self._load(max(0, idx - k))[1] for k in range(self.history - 1, 0, -1)]
        vel_hist = np.stack(hist + [vel_t], axis=0)          # (C, N, 3)

        edge_index, edge_attr = self.build_graph(pos_t)

        has_t2 = idx + 2 < len(self.frame_files)
        if has_t2:
            pos_t2, vel_t2, _ = self._load(idx + 2)
        else:
            pos_t2 = np.zeros_like(pos_t1)
            vel_t2 = np.zeros_like(vel_t1)

        sample = {
            "vel_hist": torch.from_numpy(vel_hist),          # (C, N, 3)
            "pos_t": torch.from_numpy(pos_t),
            "pos_t1": torch.from_numpy(pos_t1),
            "vel_t1": torch.from_numpy(vel_t1),
            "pos_t2": torch.from_numpy(pos_t2),
            "vel_t2": torch.from_numpy(vel_t2),
            "has_t2": torch.tensor(has_t2),
            "particle_types": torch.from_numpy(types),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
        }
        if self._cache is not None:
            self._cache[idx] = sample
        return sample


def make_node_features(vel_hist: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """(C,N,3) + (N,) -> (N, 3C+1). History flattened oldest->newest [U3]."""
    C, N, _ = vel_hist.shape
    flat = vel_hist.permute(1, 0, 2).reshape(N, 3 * C)
    return torch.cat([flat, types.unsqueeze(-1)], dim=-1)


def edge_features_from(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Recompute [rel_pos, dist] for a fixed connectivity [T1/U4]."""
    rel = pos[edge_index[1]] - pos[edge_index[0]]
    dist = rel.norm(dim=1, keepdim=True)
    return torch.cat([rel, dist], dim=1)


# =============================================================================
# [T1] Noise injection
# =============================================================================
def apply_random_walk_noise(vel_hist, pos_t, types, noise_std, generator=None):
    """
    GNS-style random-walk noise on the velocity history of FLUID particles.
    Per-step std = noise_std / sqrt(C); the walk accumulates so the newest
    velocity carries ~noise_std total. Positions drift consistently by
    dt * cumulative noise. Boundary particles stay clean (GT boundary cond.).
    """
    if noise_std <= 0:
        return vel_hist, pos_t
    C = vel_hist.shape[0]
    step_std = noise_std / (C ** 0.5)
    steps = torch.randn(vel_hist.shape, generator=generator) * step_std
    walk = torch.cumsum(steps, dim=0)                        # (C, N, 3)
    fluid = (types < 0.5).unsqueeze(-1)                      # (N, 1)

    noisy_hist = vel_hist + walk * fluid.unsqueeze(0)
    pos_drift = walk.sum(dim=0) * DT                         # integrate the walk
    noisy_pos = pos_t + pos_drift * fluid
    return noisy_hist, noisy_pos


# =============================================================================
# [T5] Loss
# =============================================================================
class V4Loss(nn.Module):
    """
    Normalised-delta MSE (primary) + physical-scale physics terms.
    Physics formulas identical to scripts/training/improved_physics_loss.py
    (velocity-scale pos-consistency, clamped momentum & energy).
    """

    def __init__(self, model, delta_weight=1.0, pos_weight=0.1,
                 momentum_weight=0.05, energy_weight=0.05, dt=DT, eps=1e-8):
        super().__init__()
        self.model = model
        self.w = (delta_weight, pos_weight, momentum_weight, energy_weight)
        self.dt = dt
        self.eps = eps

    def forward(self, pred_norm, v_t, v_true_next, x_t, x_true_next, types):
        is_fluid = types < 0.5
        dw, pw, mw, ew = self.w

        # 1. Primary: MSE in normalised delta space [U1/U2]
        target_norm = self.model.normalize_target(v_true_next - v_t)
        delta_loss = torch.mean((pred_norm[is_fluid] - target_norm[is_fluid]) ** 2)

        # Physical reconstruction for the physics terms
        v_hat = (v_t + self.model.denormalize_output(pred_norm))[is_fluid]
        v_true = v_true_next[is_fluid]

        # 2. Position-consistency at velocity scale
        implied_vel = (x_true_next[is_fluid] - x_t[is_fluid]) / self.dt
        pos_loss = torch.mean((v_hat - implied_vel) ** 2)

        # 3. Momentum (normalised by velocity scale, clamped)
        n_f = v_hat.shape[0]
        if n_f > 0:
            p_hat = v_hat.sum(dim=0) / n_f
            p_true = v_true.sum(dim=0) / n_f
            v_scale = torch.mean(v_true ** 2) + self.eps
            momentum_loss = torch.clamp(
                torch.sum((p_hat - p_true) ** 2) / v_scale, max=10.0)
        else:
            momentum_loss = torch.tensor(0.0, device=pred_norm.device)

        # 4. Kinetic-energy conservation (relative, clamped)
        ke_hat = 0.5 * torch.sum(v_hat ** 2)
        ke_true = 0.5 * torch.sum(v_true ** 2)
        energy_loss = torch.clamp(
            torch.abs(ke_hat - ke_true) / (ke_true + self.eps), max=10.0)

        total = dw * delta_loss + pw * pos_loss + mw * momentum_loss + ew * energy_loss
        return total, {
            "delta_loss": delta_loss.item(),
            "pos_loss": pos_loss.item(),
            "momentum_loss": momentum_loss.item(),
            "energy_loss": energy_loss.item(),
        }


COMPONENT_KEYS = ("delta_loss", "pos_loss", "momentum_loss", "energy_loss")


# =============================================================================
# [T2] Normalisation statistics (train split only)
# =============================================================================
def compute_normalizer_stats(train_subsets, history):
    """Accumulate mean/std for node features, edge features, fluid delta targets."""
    node_dim, edge_dim = 3 * history + 1, 4
    acc = {
        "node": [np.zeros(node_dim), np.zeros(node_dim), 0],
        "edge": [np.zeros(edge_dim), np.zeros(edge_dim), 0],
        "target": [np.zeros(3), np.zeros(3), 0],
    }

    def add(key, arr):
        s, ss, n = acc[key]
        a = arr.astype(np.float64)
        acc[key] = [s + a.sum(0), ss + (a ** 2).sum(0), n + a.shape[0]]

    t0 = time.time()
    for subset in train_subsets:
        for i in range(len(subset)):
            s = subset[i]
            nf = make_node_features(s["vel_hist"], s["particle_types"]).numpy()
            add("node", nf)
            add("edge", s["edge_attr"].numpy())
            fluid = s["particle_types"].numpy() < 0.5
            v_t = s["vel_hist"][-1].numpy()
            add("target", (s["vel_t1"].numpy() - v_t)[fluid])

    stats = {}
    for key, (s, ss, n) in acc.items():
        mean = s / n
        var = np.maximum(ss / n - mean ** 2, 1e-12)
        stats[f"{key}_mean"], stats[f"{key}_std"] = mean, np.sqrt(var)
    print(f"[stats] computed over train split in {time.time() - t0:.1f}s "
          f"(nodes={acc['node'][2]:,}, edges={acc['edge'][2]:,}, "
          f"targets={acc['target'][2]:,})")
    print(f"[stats] target_std = {stats['target_std']}")
    return stats


# =============================================================================
# Train / validate
# =============================================================================
def one_step_loss(model, criterion, vel_hist, pos_t, types, edge_index,
                  vel_next, pos_next):
    """Forward + loss for a single transition with the given (possibly noisy) state."""
    nf = make_node_features(vel_hist, types)
    ea = edge_features_from(pos_t, edge_index)
    pred_norm = model(nf, edge_index, ea)
    return criterion(pred_norm, vel_hist[-1], vel_next, pos_t, pos_next, types), pred_norm


def pushforward_second_step(model, criterion, batch, vel_hist, pos_t, types,
                            edge_index, radius, device):
    """
    [T3] no-grad step t->t+1 with the model's own prediction, rebuild graph,
    then return the supervised loss on t+1 -> t+2.
    """
    fluid = (types < 0.5).unsqueeze(-1)
    with torch.no_grad():
        nf = make_node_features(vel_hist, types)
        ea = edge_features_from(pos_t, edge_index)
        v_hat1 = model.predict_next_velocity(nf, edge_index, ea)

    # Advance: fluid integrates prediction, rigid snaps to GT (boundary cond.)
    v1 = torch.where(fluid, v_hat1, batch["vel_t1"])
    x1 = torch.where(fluid, pos_t + v_hat1 * DT, batch["pos_t1"])
    hist1 = torch.cat([vel_hist[1:], v1.unsqueeze(0)], dim=0).detach()
    x1 = x1.detach()

    # Rebuild connectivity at the pushed-forward positions
    tree = cKDTree(x1.cpu().numpy())
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    ei1 = torch.from_numpy(edges).to(device)

    (loss, comps), _ = one_step_loss(
        model, criterion, hist1, x1, types, ei1,
        batch["vel_t2"], batch["pos_t2"])
    return loss, comps


def run_epoch(model, loader, criterion, optimizer, device, train,
              noise_std=0.0, pushforward_prob=0.0, radius=0.3,
              accum_steps=1, rng=None):
    model.train() if train else model.eval()
    total, comp, n_batches = 0.0, {k: 0.0 for k in COMPONENT_KEYS}, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    if train:
        optimizer.zero_grad()

    with ctx:
        for bi, batch in enumerate(loader):
            batch = {k: v.squeeze(0).to(device) for k, v in batch.items()}
            types = batch["particle_types"]
            edge_index = batch["edge_index"]
            vel_hist, pos_t = batch["vel_hist"], batch["pos_t"]

            use_push = (train and pushforward_prob > 0
                        and bool(batch["has_t2"])
                        and float(torch.rand((), generator=rng)) < pushforward_prob)

            if train and noise_std > 0:
                vel_hist, pos_t = apply_random_walk_noise(
                    vel_hist.cpu(), pos_t.cpu(), types.cpu(), noise_std, rng)
                vel_hist, pos_t = vel_hist.to(device), pos_t.to(device)

            if use_push:
                loss, c = pushforward_second_step(
                    model, criterion, batch, vel_hist, pos_t, types,
                    edge_index, radius, device)
            else:
                (loss, c), _ = one_step_loss(
                    model, criterion, vel_hist, pos_t, types, edge_index,
                    batch["vel_t1"], batch["pos_t1"])

            if train:
                (loss / accum_steps).backward()
                if (bi + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total += loss.item()
            for k in COMPONENT_KEYS:
                comp[k] += c[k]
            n_batches += 1

        if train and n_batches % accum_steps != 0:   # flush the tail
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    n = max(1, n_batches)
    return total / n, {k: v / n for k, v in comp.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="v4 Hybrid GNN training")
    p.add_argument("--tag", type=str, required=True,
                   help="run tag for checkpoint/log names (e.g. full, no_noise)")
    p.add_argument("--resume-from", type=str, default="",
                   help="checkpoint to resume from; --epochs then means ADDITIONAL epochs, "
                        "best ckpt keeps overwriting the same file (only on improvement)")
    p.add_argument("--resume-log", type=str, default="",
                   help="existing training log json to APPEND to (epoch numbering continues)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--sched", choices=["cosine", "plateau"], default="cosine")
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="[T1] total random-walk noise std on fluid velocities (0=off)")
    p.add_argument("--pushforward-prob", type=float, default=0.3,
                   help="[T3] probability of a 2-step pushforward sample (0=off)")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--delta-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=0.1)
    p.add_argument("--momentum-weight", type=float, default=0.05)
    p.add_argument("--energy-weight", type=float, default=0.05)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-type-weighting", action="store_true")
    p.add_argument("--boundary-importance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--out-dir", type=str, default=str(ROOT / "models"))
    p.add_argument("--limit-sims", type=int, default=0, help="debug: use only N sims")
    p.add_argument("--limit-seq", type=int, default=0, help="debug: max seq per sim")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    split_gen = torch.Generator().manual_seed(args.seed)

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / V4_VERSION_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        ckpt_path = Path(args.resume_from)          # keep overwriting the same artifact
    else:
        ckpt_path = Path(args.out_dir) / f"hybrid_v4_{args.tag}_{train_ts}_{V4_VERSION_ID}.pt"
    if args.resume_log:
        log_path = Path(args.resume_log)            # append to the existing history
    else:
        log_path = log_dir / f"training_{args.tag}_{train_ts}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\nV4 TRAINING  tag={args.tag}  version={V4_VERSION_ID}")
    print(f"  device={device}  epochs={args.epochs}  lr={args.lr} ({args.sched})")
    print(f"  noise_std={args.noise_std}  pushforward_prob={args.pushforward_prob}")
    print(f"  history={args.history}  layers={args.num_layers}  hidden={args.hidden_dim}")
    print(f"  ckpt -> {ckpt_path}\n  log  -> {log_path}\n{'=' * 70}")

    # --- data ---
    data_root = Path(args.data_dir)
    sim_dirs = sorted(d for d in data_root.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    if args.limit_sims:
        sim_dirs = sim_dirs[: args.limit_sims]
    print(f"Found {len(sim_dirs)} sims in {data_root}")

    train_subs, val_subs = [], []
    for sd in sim_dirs:
        ds = V4Dataset(sd, radius=args.radius, history=args.history, cache=True)
        n = len(ds) if not args.limit_seq else min(len(ds), args.limit_seq)
        if n < 2:
            continue
        idx = list(range(n))
        split = int(0.8 * n)
        perm = torch.randperm(n, generator=split_gen).tolist()
        tr_idx = [idx[i] for i in perm[:split]]
        va_idx = [idx[i] for i in perm[split:]]
        train_subs.append(torch.utils.data.Subset(ds, tr_idx))
        val_subs.append(torch.utils.data.Subset(ds, va_idx))

    train_loader = DataLoader(ConcatDataset(train_subs), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_subs), batch_size=1, shuffle=False)
    print(f"Train sequences: {len(train_loader.dataset)}  "
          f"Val sequences: {len(val_loader.dataset)}")

    # --- model (fresh, or rebuilt from the resume checkpoint's config) ---
    resume_ckpt = None
    start_epoch = 0
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        mc = {k: v for k, v in resume_ckpt["model_config"].items() if k != "version"}
        model = HybridPhysicsGNNv4(**mc).to(device)
        model.load_state_dict(resume_ckpt["state_dict"])   # weights + normaliser buffers
        start_epoch = resume_ckpt.get("epoch", 0)
        print(f"[resume] {args.resume_from}")
        print(f"[resume] from epoch {start_epoch} (val {resume_ckpt.get('val_loss', float('nan')):.5f}); "
              f"training {args.epochs} additional epochs")
    else:
        model = HybridPhysicsGNNv4(
            history=args.history,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_learned_attention=not args.no_attention,
            use_type_weighting=not args.no_type_weighting,
            boundary_importance=args.boundary_importance,
        ).to(device)
    print(f"Model params: {model.count_parameters():,}")

    # --- [T2] normaliser stats (train split only, also warms the cache) ---
    if resume_ckpt is None:
        stats = compute_normalizer_stats(train_subs, args.history)
        model.set_normalizer_stats(
            stats["node_mean"], stats["node_std"],
            stats["edge_mean"], stats["edge_std"],
            stats["target_mean"], stats["target_std"],
        )
        stats_path = log_dir / f"normalizer_{args.tag}_{train_ts}.json"
        stats_path.write_text(json.dumps({k: v.tolist() for k, v in stats.items()}, indent=2))
    else:
        print("[resume] normaliser stats restored from checkpoint buffers (not recomputed)")

    # --- loss / optim / sched ---
    criterion = V4Loss(model, args.delta_weight, args.pos_weight,
                       args.momentum_weight, args.energy_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)

    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("[resume] optimizer state restored")

    # --- loop (fresh history, or append to the existing log) ---
    if args.resume_log and Path(args.resume_log).exists():
        history = json.loads(Path(args.resume_log).read_text())
        history.setdefault("resumes", []).append(
            {"at_epoch": start_epoch, "train_ts": train_ts, "config": vars(args)})
        print(f"[resume] appending to {args.resume_log} "
              f"({len(history['val'])} epochs already logged)")
    else:
        history = {"train": [], "val": [], "config": vars(args),
                   "version": V4_VERSION_ID, "n_params": model.count_parameters()}
    best_val = resume_ckpt["val_loss"] if resume_ckpt is not None else float("inf")
    end_epoch = start_epoch + args.epochs
    t0 = time.time()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        ep_t = time.time()
        tr_loss, tr_c = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            noise_std=args.noise_std, pushforward_prob=args.pushforward_prob,
            radius=args.radius, accum_steps=args.accum_steps, rng=rng)
        va_loss, va_c = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        if args.sched == "cosine":
            scheduler.step()
        else:
            scheduler.step(va_loss)

        history["train"].append({"epoch": epoch, "loss": tr_loss, **tr_c})
        history["val"].append({"epoch": epoch, "loss": va_loss, **va_c})

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            torch.save({
                "state_dict": model.state_dict(),      # includes normaliser buffers
                "model_config": model.get_config(),
                "train_config": vars(args),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": va_loss,
                "version": V4_VERSION_ID,
            }, ckpt_path)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[v4/{args.tag}] ep {epoch:>3}/{end_epoch} "
              f"train={tr_loss:.5f} val={va_loss:.5f}  "
              f"d={va_c['delta_loss']:.4f} p={va_c['pos_loss']:.4f} "
              f"m={va_c['momentum_loss']:.4f} e={va_c['energy_loss']:.4f}  "
              f"lr={lr_now:.1e} ({time.time() - ep_t:.1f}s) "
              f"{'(best)' if improved else ''}")
        log_path.write_text(json.dumps(history, indent=2))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min   best val = {best_val:.5f}")
    print(f"Best ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
