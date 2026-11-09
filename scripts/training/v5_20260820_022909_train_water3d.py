"""
v5 training - external validation on the public DeepMind GNS **Water-3D**
benchmark (84 GB TFRecord, 3D, 800-step trajectories).

Version ID : v5_20260820_022909
Base       : v4 driver (scripts/training/v4_20260707_102544_train_v4.py). The
             whole v4 recipe (normalised delta output, velocity history, noise
             injection, pushforward, cosine LR, 4-term physics loss) is reused
             by import - nothing about the method changes. Only the data
             adapter differs.

Why this run exists
-------------------
Every previous version (v2..v4.1) trained on self-generated SPH data
(datasets/flip_simulations_v3p2_*, ~900 particles, 30 scenarios). A journal
submission needs a public, validated benchmark. Water-3D is the standard one
for 3D particle fluids and is 5-12x larger per scene (4.6k-11k particles).

Two things differ from our own dataset and are handled here:

1. Velocity convention. Water-3D stores positions only; GNS defines velocity
   as the per-step position delta, i.e. dt = 1 in model units. DT = 1.0 here
   (v4 used 0.01 with physical velocities), so integration and the
   position-consistency loss term stay dimensionally correct.

2. No boundary particles. Every Water-3D particle is water (GNS type 5); the
   container exists only as metadata bounds. That is exactly the situation the
   v4.1 contribution addresses, so walls are sampled onto the six faces of the
   metadata box and appended as static type-1 particles, with wall-wall edges
   dropped. Use --boundary none for the no-wall ablation.

Usage:
  python scripts/training/v5_20260820_022909_train_water3d.py --tag w3d_wall
  python scripts/training/v5_20260820_022909_train_water3d.py --tag w3d_nowall \
      --boundary none
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))

v4 = importlib.import_module("v4_20260707_102544_train_v4")
from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

V5_VERSION_ID = "v5_20260820_022909"
DATA_DIR = ROOT / "datasets" / "gns_water3d_v5"

# GNS convention: velocity is the per-step position delta, so dt = 1.
DT = 1.0
v4.DT = DT          # v4.run_epoch / pushforward read this at call time


# =============================================================================
# Wall particles on the metadata box
# =============================================================================
def make_box_walls(bounds, spacing, margin=0.0, layers=1, layer_gap=None):
    """Static wall particles on the six faces of the [lo, hi] box.

    bounds : (2, 3) array - row 0 = lo, row 1 = hi.
    margin : push each face OUTWARD by this much. The Water-3D metadata bounds
             underestimate the true container - GT particles compress up to
             ~0.009 past them - so walls placed at the raw bounds sit inside
             the fluid region.
    layers : number of concentric wall shells (outer shells catch fast
             particles that tunnel through the first within one step).
    layer_gap : outward distance between shells (default: spacing).
    Returns float32 (M, 3), de-duplicated along shared face edges.
    """
    lo0 = np.asarray(bounds[0], np.float64) - margin
    hi0 = np.asarray(bounds[1], np.float64) + margin
    gap = spacing if layer_gap is None else layer_gap
    pts = []
    for layer in range(layers):
        lo, hi = lo0 - layer * gap, hi0 + layer * gap
        for ax in range(3):
            o1, o2 = (ax + 1) % 3, (ax + 2) % 3
            u = np.arange(lo[o1], hi[o1] + 1e-9, spacing)
            v = np.arange(lo[o2], hi[o2] + 1e-9, spacing)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            for side in (lo[ax], hi[ax]):
                p = np.zeros((uu.size, 3), dtype=np.float64)
                p[:, ax] = side
                p[:, o1] = uu.ravel()
                p[:, o2] = vv.ravel()
                pts.append(p)
    walls = np.concatenate(pts, axis=0)
    q = spacing * 0.5
    walls = np.unique(np.round(walls / q).astype(np.int64), axis=0) * q
    return walls.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================
class Water3DDataset(v4.V4Dataset):
    """One Water-3D trajectory (.npz) exposed with the v4 sample contract.

    Overrides the frame-file I/O of V4Dataset: a whole trajectory lives in one
    array, and velocities are derived as position deltas.
    """

    def __init__(self, traj_path, radius, history=3, walls=None, stride=1):
        d = np.load(traj_path)
        pos = d["positions"].astype(np.float32)           # (T, N, 3)
        self.n_orig = pos.shape[1]
        self.walls = (np.zeros((0, 3), np.float32) if walls is None
                      else walls.astype(np.float32))
        self.n_walls = len(self.walls)

        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]                                   # no delta at t=0
        self.pos, self.vel = pos, vel
        if self.n_walls:
            self.types = np.concatenate([np.zeros(self.n_orig, np.float32),
                                         np.ones(self.n_walls, np.float32)])
        else:
            self.types = np.zeros(self.n_orig, np.float32)

        self.radius = radius
        self.history = history
        self.stride = max(1, stride)
        self.n_frames = pos.shape[0]
        # v4.V4Dataset.__getitem__ only needs len(self.frame_files).
        self.frame_files = [None] * self.n_frames
        self._cache = None                                # graphs differ per frame
        self._idx = list(range(0, self.n_frames - 2, self.stride))

    def __len__(self):
        return len(self._idx)

    def _load(self, i):
        p, v = self.pos[i], self.vel[i]
        if self.n_walls:
            p = np.concatenate([p, self.walls])
            v = np.concatenate([v, np.zeros((self.n_walls, 3), np.float32)])
        return p, v, self.types

    def build_graph(self, positions):
        """Radius graph without wall-wall edges."""
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if self.n_walls and len(pairs) > 0:
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
        sample = super().__getitem__(self._idx[idx])
        sample["n_orig"] = torch.tensor(self.n_orig)
        return sample


def v5_pushforward(model, criterion, batch, vel_hist, pos_t, types,
                   edge_index, radius, device):
    """v4 pushforward with DT = 1 and the wall-wall edge filter."""
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


# run_epoch resolves the pushforward through the v4 module globals - patch it
# so the rebuilt graph also drops wall-wall edges and uses DT = 1.
v4.pushforward_second_step = v5_pushforward


# =============================================================================
def main():
    p = argparse.ArgumentParser(description="v5 Water-3D external validation")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=10)
    # GNS uses 6.7e-4 random-walk noise on Water-3D (vel_std ~2.4e-3).
    p.add_argument("--noise-std", type=float, default=6.7e-4)
    p.add_argument("--pushforward-prob", type=float, default=0.3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--boundary", choices=["walls", "none"], default="walls")
    p.add_argument("--wall-spacing", type=float, default=0.0,
                   help="0 = radius/2 from the dataset metadata")
    p.add_argument("--wall-margin", type=float, default=0.0,
                   help="push wall faces outward (true Water-3D container is "
                        "~0.007-0.009 beyond the metadata bounds)")
    p.add_argument("--wall-layers", type=int, default=1,
                   help="number of concentric wall shells")
    p.add_argument("--stride", type=int, default=10,
                   help="use every k-th transition (800 frames/traj is a lot)")
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-type-weighting", action="store_true")
    p.add_argument("--boundary-importance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-trajs", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    radius = meta["default_connectivity_radius"]
    bounds = np.asarray(meta["bounds"], dtype=np.float32).T       # (2, 3)
    spacing = args.wall_spacing or radius / 2

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = ROOT / "models" / ("hybrid_" + args.tag + "_" + train_ts
                                   + "_" + V5_VERSION_ID + ".pt")
    log_dir = ROOT / "logs" / V5_VERSION_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("training_" + args.tag + "_" + train_ts + ".json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bar = "=" * 70
    print("\n" + bar)
    print("V5 WATER-3D TRAINING  tag=%s  version=%s" % (args.tag, V5_VERSION_ID))
    print("  dataset=%s  radius=%g  DT=%g (GNS delta convention)"
          % (DATA_DIR.name, radius, DT))
    print("  boundary=%s%s" % (args.boundary,
          ("  wall_spacing=%.4f" % spacing) if args.boundary == "walls" else ""))
    print("  stride=%d  noise_std=%g  pushforward=%g"
          % (args.stride, args.noise_std, args.pushforward_prob))
    print("  device=%s  epochs=%d  ckpt -> %s" % (device, args.epochs, ckpt_path.name))
    print(bar)

    walls = make_box_walls(bounds, spacing, margin=args.wall_margin,
                           layers=args.wall_layers) \
        if args.boundary == "walls" else None
    if walls is not None:
        print("wall particles: %d  (margin=%g, layers=%d, box %s - %s)"
              % (len(walls), args.wall_margin, args.wall_layers,
                 bounds[0].tolist(), bounds[1].tolist()))

    def load_split(split):
        files = sorted((DATA_DIR / split).glob("traj_*.npz"))
        if args.limit_trajs:
            files = files[: args.limit_trajs]
        return [Water3DDataset(f, radius, args.history, walls, args.stride)
                for f in files]

    train_sets, val_sets = load_split("train"), load_split("valid")
    if not train_sets or not val_sets:
        raise SystemExit("no trajectories under %s - run "
                         "scripts/data_generation/v5_convert_water3d.py first"
                         % DATA_DIR)
    print("train trajs=%d  valid trajs=%d  fluid particles/traj~%d%s"
          % (len(train_sets), len(val_sets), train_sets[0].n_orig,
             "" if walls is None else " (+%d walls)" % len(walls)))

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=1, shuffle=False)
    print("Train samples: %d  Val samples: %d"
          % (len(train_loader.dataset), len(val_loader.dataset)))

    model = HybridPhysicsGNNv4(
        history=args.history,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_learned_attention=not args.no_attention,
        use_type_weighting=not args.no_type_weighting,
        boundary_importance=args.boundary_importance,
    ).to(device)
    print("Model params: %d" % model.count_parameters())

    # Normaliser stats over a bounded slice of the train split (a full pass is
    # unnecessary and slow at this scale).
    stat_sets = [Subset(ds, list(range(0, len(ds), max(1, len(ds) // 20))))
                 for ds in train_sets]
    stats = v4.compute_normalizer_stats(stat_sets, args.history)
    model.set_normalizer_stats(
        stats["node_mean"], stats["node_std"],
        stats["edge_mean"], stats["edge_std"],
        stats["target_mean"], stats["target_std"])
    (log_dir / ("normalizer_" + args.tag + "_" + train_ts + ".json")).write_text(
        json.dumps({k: v_.tolist() for k, v_ in stats.items()}, indent=2))

    criterion = v4.V4Loss(model, dt=DT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    history = {"train": [], "val": [], "config": vars(args),
               "version": V5_VERSION_ID, "dataset": "gns_water3d",
               "radius": radius, "dt": DT,
               "n_walls": 0 if walls is None else len(walls),
               "n_params": model.count_parameters()}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr, trc = v4.run_epoch(model, train_loader, criterion, optimizer, device,
                               train=True, noise_std=args.noise_std,
                               pushforward_prob=args.pushforward_prob,
                               radius=radius, accum_steps=args.accum_steps,
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
                        "radius": radius, "dt": DT,
                        "walls": None if walls is None else walls,
                        "version": V5_VERSION_ID}, ckpt_path)
        print("[v5/%s] ep %3d/%d train=%.5f val=%.5f  "
              "d=%.4f p=%.4f m=%.4f e=%.4f  (%.1fs) %s"
              % (args.tag, epoch, args.epochs, tr, va,
                 vac["delta_loss"], vac["pos_loss"],
                 vac["momentum_loss"], vac["energy_loss"],
                 time.time() - ep_t, "(best)" if improved else ""), flush=True)
        log_path.write_text(json.dumps(history, indent=2))

    print("\nDone in %.1f min   best val = %.5f" % ((time.time() - t0) / 60, best_val))
    print("Best ckpt: %s" % ckpt_path)


if __name__ == "__main__":
    main()
