"""
v4.1 training — tank walls as boundary particles under the SAME type-weighting
mechanism as the rigid bodies ("강체와 같이 weight 방식").

Version ID : v4p1_20260712_001555
Base       : v4 driver (scripts/training/v4_20260707_102544_train_v4.py) — all
             v4 recipe components (delta output, normalisation, history, noise,
             pushforward, cosine LR) are reused via import; nothing is modified.

Motivation (BC-compliance analysis, logs/v4_20260707_102544/bc_compliance.json):
  v4 respects the rigid-body boundary (penetration ~0.13-0.16%, GT-level) but
  leaks 8-12% of fluid through the TANK WALLS, because walls are not particles
  in the dataset and v4's translation-invariant inputs (no absolute positions)
  make walls literally invisible.

Fix (this file): sample static boundary particles on the tank faces and append
them to every frame with type c=1 and zero velocity. The existing Hybrid
machinery then treats walls exactly like rigid bodies:
  - graph edges wall<->fluid form within the connectivity radius,
  - type weighting w_ij = beta (2.0) up-weights wall interactions,
  - the loss ignores them (fluid-only), rollout snaps them (static).
Translation invariance is preserved — wall geometry enters only through
relative edge features, no absolute coordinates.

Efficiency: wall-wall edges are dropped at graph build (wall latents still
receive fluid messages; boundary-boundary message passing adds cost, no
gradient value — standard practice).

Usage:
  python scripts/training/v4p1_20260712_001555_train_wall.py --tag wall_hybrid
  python scripts/training/v4p1_20260712_001555_train_wall.py --tag wall_backbone \
      --no-attention --no-type-weighting
"""
import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))

v4 = importlib.import_module("v4_20260707_102544_train_v4")
from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

V4P1_VERSION_ID = "v4p1_20260712_001555"
DATA_DIR = v4.DATA_DIR
DT = v4.DT


# =============================================================================
# Wall particles
# =============================================================================
def make_wall_particles(sim_dir, spacing=0.1, approach=0.1, grid=0.05):
    """
    Static wall particles on the tank faces of one scenario.

    Faces are the fluid GT bounding box (over all frames) snapped OUTWARD to
    a `grid` lattice; a face is included only if the fluid ever approaches it
    within `approach` (open faces get no particles).
    """
    files = sorted(Path(sim_dir).glob("frame_*.npz"))
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for f in files[::5]:
        d = np.load(f)
        fl = d["positions"][d["particle_types"] < 0.5]
        lo = np.minimum(lo, fl.min(0))
        hi = np.maximum(hi, fl.max(0))
    face_lo = np.floor(lo / grid) * grid
    face_hi = np.ceil(hi / grid) * grid

    pts = []
    for ax in range(3):
        o1, o2 = (ax + 1) % 3, (ax + 2) % 3
        u = np.arange(face_lo[o1], face_hi[o1] + 1e-9, spacing)
        v = np.arange(face_lo[o2], face_hi[o2] + 1e-9, spacing)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        for side, bound in ((face_lo[ax], lo[ax]), (face_hi[ax], hi[ax])):
            if abs(bound - side) > approach:      # fluid never gets close
                continue
            p = np.zeros((uu.size, 3), dtype=np.float32)
            p[:, ax] = side
            p[:, o1] = uu.ravel()
            p[:, o2] = vv.ravel()
            pts.append(p)
    walls = np.concatenate(pts, axis=0)
    walls = np.unique(np.round(walls / (spacing * 0.5)).astype(np.int64),
                      axis=0) * (spacing * 0.5)   # dedupe face-edge overlaps
    return walls.astype(np.float32)


class WallDataset(v4.V4Dataset):
    """V4Dataset + static wall boundary particles appended to every frame."""

    def __init__(self, data_dir, walls, radius=0.3, history=3, cache=True):
        self.walls = walls
        self.n_walls = len(walls)
        d0 = np.load(sorted(Path(data_dir).glob("frame_*.npz"))[0])
        self.n_orig = len(d0["positions"])
        super().__init__(data_dir, radius=radius, history=history, cache=cache)

    def _load(self, i):
        pos, vel, types = super()._load(i)
        return (np.concatenate([pos, self.walls]),
                np.concatenate([vel, np.zeros((self.n_walls, 3), np.float32)]),
                np.concatenate([types, np.ones(self.n_walls, np.float32)]))

    def build_graph(self, positions):
        """Radius graph WITHOUT wall-wall edges."""
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if len(pairs) > 0:
            keep = ~((pairs[:, 0] >= self.n_orig) & (pairs[:, 1] >= self.n_orig))
            pairs = pairs[keep]
        if len(pairs) == 0:
            edges = np.array([[0], [0]], dtype=np.int64)
        else:
            edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
        rel = positions[edges[1]] - positions[edges[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        return edges, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if "n_orig" not in sample:
            sample["n_orig"] = torch.tensor(self.n_orig)
        return sample


def wall_pushforward(model, criterion, batch, vel_hist, pos_t, types,
                     edge_index, radius, device):
    """v4 pushforward with the wall-wall edge filter on the rebuilt graph."""
    n_orig = int(batch["n_orig"])
    fluid = (types < 0.5).unsqueeze(-1)
    with torch.no_grad():
        nf = v4.make_node_features(vel_hist, types)
        ea = v4.edge_features_from(pos_t, edge_index)
        v_hat1 = model.predict_next_velocity(nf, edge_index, ea)

    v1 = torch.where(fluid, v_hat1, batch["vel_t1"])
    x1 = torch.where(fluid, pos_t + v_hat1 * DT, batch["pos_t1"])
    hist1 = torch.cat([vel_hist[1:], v1.unsqueeze(0)], dim=0).detach()
    x1 = x1.detach()

    tree = cKDTree(x1.cpu().numpy())
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) > 0:
        keep = ~((pairs[:, 0] >= n_orig) & (pairs[:, 1] >= n_orig))
        pairs = pairs[keep]
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    ei1 = torch.from_numpy(edges).to(device)

    (loss, comps), _ = v4.one_step_loss(
        model, criterion, hist1, x1, types, ei1,
        batch["vel_t2"], batch["pos_t2"])
    return loss, comps


# run_epoch resolves pushforward through the v4 module globals — patch it so
# the rebuilt pushforward graph also excludes wall-wall edges.
v4.pushforward_second_step = wall_pushforward


# =============================================================================
# Main (v4 main with the wall dataset; resume/limit flags omitted for brevity)
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="v4.1 wall-boundary training")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--pushforward-prob", type=float, default=0.3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--wall-spacing", type=float, default=0.1)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-type-weighting", action="store_true")
    p.add_argument("--boundary-importance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-sims", type=int, default=0)
    p.add_argument("--limit-seq", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    split_gen = torch.Generator().manual_seed(args.seed)

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = ROOT / "models" / f"hybrid_{args.tag}_{train_ts}_{V4P1_VERSION_ID}.pt"
    log_dir = ROOT / "logs" / V4P1_VERSION_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{args.tag}_{train_ts}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\nV4.1 WALL TRAINING  tag={args.tag}  version={V4P1_VERSION_ID}")
    print(f"  wall_spacing={args.wall_spacing}  (walls as type-1 boundary particles)")
    print(f"  device={device}  epochs={args.epochs}  ckpt -> {ckpt_path}\n{'=' * 70}")

    sim_dirs = sorted(d for d in Path(DATA_DIR).iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    if args.limit_sims:
        sim_dirs = sim_dirs[: args.limit_sims]

    train_subs, val_subs = [], []
    n_wall_total = 0
    for sd in sim_dirs:
        walls = make_wall_particles(sd, spacing=args.wall_spacing)
        n_wall_total += len(walls)
        ds = WallDataset(sd, walls, radius=args.radius, history=args.history)
        n = len(ds) if not args.limit_seq else min(len(ds), args.limit_seq)
        if n < 2:
            continue
        split = int(0.8 * n)
        perm = torch.randperm(n, generator=split_gen).tolist()
        train_subs.append(torch.utils.data.Subset(ds, perm[:split]))
        val_subs.append(torch.utils.data.Subset(ds, perm[split:]))
    print(f"{len(sim_dirs)} sims, avg walls/sim = {n_wall_total // max(1, len(sim_dirs))}")

    train_loader = DataLoader(ConcatDataset(train_subs), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_subs), batch_size=1, shuffle=False)
    print(f"Train sequences: {len(train_loader.dataset)}  "
          f"Val sequences: {len(val_loader.dataset)}")

    model = HybridPhysicsGNNv4(
        history=args.history,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_learned_attention=not args.no_attention,
        use_type_weighting=not args.no_type_weighting,
        boundary_importance=args.boundary_importance,
    ).to(device)
    print(f"Model params: {model.count_parameters():,}")

    stats = v4.compute_normalizer_stats(train_subs, args.history)
    model.set_normalizer_stats(
        stats["node_mean"], stats["node_std"],
        stats["edge_mean"], stats["edge_std"],
        stats["target_mean"], stats["target_std"])
    (log_dir / f"normalizer_{args.tag}_{train_ts}.json").write_text(
        json.dumps({k: v_.tolist() for k, v_ in stats.items()}, indent=2))

    criterion = v4.V4Loss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    history = {"train": [], "val": [], "config": vars(args),
               "version": V4P1_VERSION_ID, "n_params": model.count_parameters()}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr, trc = v4.run_epoch(model, train_loader, criterion, optimizer, device,
                               train=True, noise_std=args.noise_std,
                               pushforward_prob=args.pushforward_prob,
                               radius=args.radius, accum_steps=args.accum_steps,
                               rng=rng)
        va, vac = v4.run_epoch(model, val_loader, criterion, optimizer, device,
                               train=False)
        scheduler.step()
        history["train"].append({"epoch": epoch, "loss": tr, **trc})
        history["val"].append({"epoch": epoch, "loss": va, **vac})

        improved = va < best_val
        if improved:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "model_config": model.get_config(),
                        "train_config": vars(args),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "val_loss": va,
                        "version": V4P1_VERSION_ID}, ckpt_path)
        print(f"[v4p1/{args.tag}] ep {epoch:>3}/{args.epochs} "
              f"train={tr:.5f} val={va:.5f}  "
              f"d={vac['delta_loss']:.4f} p={vac['pos_loss']:.4f} "
              f"m={vac['momentum_loss']:.4f} e={vac['energy_loss']:.4f}  "
              f"({time.time() - ep_t:.1f}s) {'(best)' if improved else ''}")
        log_path.write_text(json.dumps(history, indent=2))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min   best val = {best_val:.5f}")
    print(f"Best ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
