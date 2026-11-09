# -*- coding: utf-8 -*-
"""v6 training - second public benchmark: DeepMind GNS WaterRamps (2D).

Version ID : v6_20261021_145750
Base       : the v4 driver (scripts/training/v4_20260707_102544_train_v4.py),
             imported and reused. The recipe does not change: normalised delta
             output, velocity history, random-walk noise, pushforward, cosine
             LR, the four-term physics loss. Only the data adapter and the
             spatial dimension differ.

Why this run exists
-------------------
Every result so far is on one public dataset, Water-3D. A single dataset is
the weakest point of the claim that boundary *placement* decides rollout
compliance, because a dataset-specific accident cannot be told apart from a
general effect. WaterRamps is the standard 2D GNS benchmark and differs from
Water-3D in three ways that matter here:

1. Dimension. 2D, so the model is instantiated through HybridPhysicsGNNv6,
   which carries the dimension as an argument. At dim=3 that class is
   bit-identical to HybridPhysicsGNNv4.

2. The obstacles are in the data. WaterRamps ships static ramps as GNS
   kinematic particles (type 3), 9 to 140 per trajectory, present in most but
   not all trajectories, and with exactly zero drift across all 601 frames.
   They arrive already in the representation this method argues for - static
   boundary particles - and they are not axis-aligned boxes. So this dataset
   tests the mechanism on geometry nobody chose for it.

3. The container overshoot is larger relative to the radius. Ground-truth
   water leaves the metadata box by up to 0.0153 on the y-lo face at a
   connectivity radius of 0.015 - about one full radius, against 0.26 radius
   on Water-3D. Walls laid on the raw metadata bounds therefore sit deeper
   inside the true fluid region here than they did there, so the placement
   effect should be larger, not smaller. That is a falsifiable prediction and
   the reason --wall-margin-mode measured exists.

The margin is measured from the training split rather than guessed: each face
is pushed out past the furthest ground-truth fluid position on that face, plus
a safety fraction of the particle spacing. The measurement is printed and
stored in the run log, because on Water-3D getting this wrong was the
difference between 1.29 percent and 74.04 percent escape.

Ablation arms (fixed model, fixed budget):
  --boundary walls --wall-margin-mode measured         proposed
  --boundary walls --wall-margin-mode measured --no-type-weighting
  --boundary walls --wall-margin 0                     walls on the metadata box
  --boundary none                                      ramps only, no container

Usage:
  python scripts/training/v6_20261021_145750_train_waterramps.py \
      --tag wr_wall2 --wall-layers 2 --epochs 100
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
from src.models.hybrid_gnn_v6 import (HybridPhysicsGNNv6, V6_VERSION_ID,
                                      make_node_features as mnf_anydim)

V6_ID = V6_VERSION_ID
DATA_DIR = ROOT / "datasets" / "gns_waterramps_v6"

# GNS convention: velocity is the per-step position delta, so dt = 1.
DT = 1.0
v4.DT = DT

# ---------------------------------------------------------------------------
# Dimension-generic replacements for the two places where the v4 driver
# assumes three spatial components. run_epoch / one_step_loss resolve these
# through the v4 module globals at call time, so patching them is enough.
# ---------------------------------------------------------------------------
v4.make_node_features = mnf_anydim


def compute_normalizer_stats_anydim(train_subsets, history):
    """v4.compute_normalizer_stats with the dimension read from the data."""
    probe = train_subsets[0][0]
    dim = probe["vel_hist"].shape[-1]
    node_dim, edge_dim = dim * history + 1, dim + 1
    acc = {
        "node": [np.zeros(node_dim), np.zeros(node_dim), 0],
        "edge": [np.zeros(edge_dim), np.zeros(edge_dim), 0],
        "target": [np.zeros(dim), np.zeros(dim), 0],
    }

    def add(key, arr):
        s, ss, n = acc[key]
        a = arr.astype(np.float64)
        acc[key] = [s + a.sum(0), ss + (a ** 2).sum(0), n + a.shape[0]]

    t0 = time.time()
    for subset in train_subsets:
        for i in range(len(subset)):
            s = subset[i]
            nf = mnf_anydim(s["vel_hist"], s["particle_types"]).numpy()
            add("node", nf)
            add("edge", s["edge_attr"].numpy())
            fluid = s["particle_types"].numpy() < 0.5
            v_t = s["vel_hist"][-1].numpy()
            add("target", (s["vel_t1"].numpy() - v_t)[fluid])

    stats = {}
    for key, (s, ss, n) in acc.items():
        mean = s / n
        var = np.maximum(ss / n - mean ** 2, 1e-12)
        stats[key + "_mean"], stats[key + "_std"] = mean, np.sqrt(var)
    print("[stats] computed over train split in %.1fs (nodes=%d, edges=%d, "
          "targets=%d)" % (time.time() - t0, acc["node"][2], acc["edge"][2],
                           acc["target"][2]))
    print("[stats] target_std = %s" % stats["target_std"])
    return stats


# =============================================================================
# Container walls, 2D
# =============================================================================
def make_box_walls_2d(bounds, spacing, margin_lo, margin_hi, layers=1,
                      layer_gap=None):
    """Static wall particles on the four sides of a 2D [lo, hi] box.

    bounds    : (2, 2) array - row 0 = lo, row 1 = hi.
    margin_lo : (2,) outward push of the lo face of each axis.
    margin_hi : (2,) outward push of the hi face of each axis.
    layers    : concentric shells; each further shell is pushed out by
                layer_gap (default: spacing) so a particle that tunnels
                through the first within one step still meets the second.

    De-duplication is exact (rounding only at 1e-9 to make float equality well
    defined), not snapped to a spacing/2 lattice as the published Water-3D
    walls were. That lattice moved face positions away from the requested
    margin and let two different margins produce an identical wall, which
    reads as a plateau in a margin sweep. Here the margin is the quantity
    under study, so the faces must land where they are asked to.
    """
    lo0 = np.asarray(bounds[0], np.float64) - np.asarray(margin_lo, np.float64)
    hi0 = np.asarray(bounds[1], np.float64) + np.asarray(margin_hi, np.float64)
    gap = spacing if layer_gap is None else layer_gap
    pts = []
    for layer in range(layers):
        lo, hi = lo0 - layer * gap, hi0 + layer * gap
        for ax in range(2):
            o = 1 - ax
            u = np.arange(lo[o], hi[o] + 1e-9, spacing)
            for side in (lo[ax], hi[ax]):
                p = np.zeros((u.size, 2), dtype=np.float64)
                p[:, ax] = side
                p[:, o] = u
                pts.append(p)
    walls = np.concatenate(pts, axis=0)
    walls = np.unique(np.round(walls, 9), axis=0)
    return walls.astype(np.float32)


def measure_fluid_overshoot(traj_paths, bounds):
    """How far ground-truth fluid goes past each face of the metadata box.

    Returns (over_lo, over_hi, fluid_lo, fluid_hi). The overshoots are
    non-negative. Measured over every frame of every trajectory given, with
    boundary particles excluded, since the question is where the water
    actually reaches.
    """
    dim = bounds.shape[1]
    f_lo = np.full(dim, np.inf)
    f_hi = np.full(dim, -np.inf)
    for path in traj_paths:
        d = np.load(path)
        n_fluid = int(d["n_fluid"])
        water = d["positions"][:, :n_fluid].reshape(-1, dim)
        f_lo = np.minimum(f_lo, water.min(0))
        f_hi = np.maximum(f_hi, water.max(0))
    over_lo = np.maximum(bounds[0] - f_lo, 0.0)
    over_hi = np.maximum(f_hi - bounds[1], 0.0)
    return over_lo.astype(np.float64), over_hi.astype(np.float64), f_lo, f_hi


# =============================================================================
# Dataset
# =============================================================================
class WaterRampsDataset(v4.V4Dataset):
    """One WaterRamps trajectory (.npz) exposed with the v4 sample contract.

    The file already holds fluid rows first and the ramp obstacles after them
    as boundary type 1. Synthesised container walls are appended after those,
    so every boundary particle - ramp or wall - sits at an index >= n_fluid
    and the boundary-boundary edge filter is a single index comparison.
    """

    def __init__(self, traj_path, radius, history=3, walls=None, stride=1):
        d = np.load(traj_path)
        pos = d["positions"].astype(np.float32)            # (T, N, 2)
        self.dim = pos.shape[-1]
        self.n_fluid = int(d["n_fluid"])
        self.n_ramp = pos.shape[1] - self.n_fluid
        self.walls = (np.zeros((0, self.dim), np.float32) if walls is None
                      else walls.astype(np.float32))
        self.n_walls = len(self.walls)
        self.n_boundary = self.n_ramp + self.n_walls

        # GNS velocity = per-step position delta. Ramp rows are constant, so
        # their delta is exactly zero without special-casing.
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]                                    # no delta at t=0
        self.pos, self.vel = pos, vel

        self.types = np.zeros(pos.shape[1] + self.n_walls, np.float32)
        self.types[self.n_fluid:] = 1.0

        self.radius = radius
        self.history = history
        self.stride = max(1, stride)
        self.n_frames = pos.shape[0]
        self.frame_files = [None] * self.n_frames   # V4Dataset reads len() only
        self._cache = None                          # connectivity differs per frame
        self._idx = list(range(0, self.n_frames - 2, self.stride))

    def __len__(self):
        return len(self._idx)

    def _load(self, i):
        p, v = self.pos[i], self.vel[i]
        if self.n_walls:
            p = np.concatenate([p, self.walls])
            v = np.concatenate([v, np.zeros((self.n_walls, self.dim), np.float32)])
        return p, v, self.types

    def build_graph(self, positions):
        """Radius graph with every boundary-boundary edge dropped.

        A boundary particle's state is an input, identical at every step, so a
        message from one to another carries no information and only costs
        memory. Ramp-ramp, ramp-wall and wall-wall edges all fall under this.
        """
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if self.n_boundary and len(pairs) > 0:
            nf = self.n_fluid
            keep = ~((pairs[:, 0] >= nf) & (pairs[:, 1] >= nf))
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
        sample["n_fluid"] = torch.tensor(self.n_fluid)
        return sample


def v6_pushforward(model, criterion, batch, vel_hist, pos_t, types,
                   edge_index, radius, device):
    """v4 pushforward with DT = 1 and the boundary-boundary edge filter."""
    n_fluid = int(batch["n_fluid"])
    fluid = (types < 0.5).unsqueeze(-1)
    with torch.no_grad():
        nf = mnf_anydim(vel_hist, types)
        ea = v4.edge_features_from(pos_t, edge_index)
        v_hat1 = model.predict_next_velocity(nf, edge_index, ea)

    v1 = torch.where(fluid, v_hat1, batch["vel_t1"])
    x1 = torch.where(fluid, pos_t + v_hat1 * DT, batch["pos_t1"])
    hist1 = torch.cat([vel_hist[1:], v1.unsqueeze(0)], dim=0).detach()
    x1 = x1.detach()

    tree = cKDTree(x1.cpu().numpy())
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) > 0:
        keep = ~((pairs[:, 0] >= n_fluid) & (pairs[:, 1] >= n_fluid))
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


v4.pushforward_second_step = v6_pushforward


# =============================================================================
def main():
    p = argparse.ArgumentParser(description="v6 WaterRamps (2D) training")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="0 = scale from the dataset vel_std as GNS does "
                        "(Water-3D used 6.7e-4 at vel_std 2.4e-3)")
    p.add_argument("--pushforward-prob", type=float, default=0.3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--boundary", choices=["walls", "none"], default="walls")
    p.add_argument("--wall-spacing", type=float, default=0.0,
                   help="0 = radius/2 from the dataset metadata")
    p.add_argument("--wall-margin", type=float, default=-1.0,
                   help="if >= 0, push every face out by this much; "
                        "-1 defers to --wall-margin-mode")
    p.add_argument("--wall-margin-mode", choices=["measured", "scalar"],
                   default="measured",
                   help="measured = per face, past the furthest ground-truth "
                        "fluid position on that face")
    p.add_argument("--wall-margin-safety", type=float, default=0.5,
                   help="extra clearance in units of wall spacing, added to "
                        "the measured overshoot")
    p.add_argument("--wall-layers", type=int, default=2)
    p.add_argument("--stride", type=int, default=5,
                   help="use every k-th transition (601 frames/traj)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-type-weighting", action="store_true")
    p.add_argument("--boundary-importance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-trajs", type=int, default=0)
    p.add_argument("--limit-val-trajs", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--out-root", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_root = Path(args.out_root) if args.out_root else ROOT
    meta = json.loads((data_dir / "metadata.json").read_text())
    radius = meta["default_connectivity_radius"]
    bounds = np.asarray(meta["bounds"], dtype=np.float64).T          # (2, dim)
    dim = int(bounds.shape[1])
    spacing = args.wall_spacing or radius / 2
    noise_std = args.noise_std if args.noise_std > 0 else \
        0.28 * float(np.mean(meta["vel_std"]))   # 6.7e-4 / 2.4e-3 on Water-3D

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "models" / ("hybrid_" + args.tag + "_" + train_ts
                                       + "_" + V6_ID + ".pt")
    log_dir = out_root / "logs" / V6_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("training_" + args.tag + "_" + train_ts + ".json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bar = "=" * 74
    print("\n" + bar)
    print("V6 WATERRAMPS TRAINING  tag=%s  version=%s" % (args.tag, V6_ID))
    print("  dataset=%s  dim=%d  radius=%g  DT=%g (GNS delta convention)"
          % (data_dir.name, dim, radius, DT))
    print("  stride=%d  noise_std=%.3e  pushforward=%g  workers=%d"
          % (args.stride, noise_std, args.pushforward_prob, args.num_workers))
    print("  device=%s  epochs=%d" % (device, args.epochs))
    print(bar, flush=True)

    train_files = sorted((data_dir / "train").glob("traj_*.npz"))
    val_files = sorted((data_dir / "valid").glob("traj_*.npz"))
    if args.limit_trajs:
        train_files = train_files[: args.limit_trajs]
    if args.limit_val_trajs:
        val_files = val_files[: args.limit_val_trajs]
    if not train_files or not val_files:
        raise SystemExit(
            "no trajectories under %s - run "
            "scripts/data_generation/v6_20261021_145750_convert_waterramps.py "
            "first" % data_dir)

    # --- where the walls go -------------------------------------------------
    walls = None
    wall_info = {"mode": args.boundary}
    if args.boundary == "walls":
        over_lo, over_hi, f_lo, f_hi = measure_fluid_overshoot(train_files,
                                                               bounds)
        print("ground-truth fluid extent over the training split:")
        for a in range(dim):
            print("  axis %d: [%.5f, %.5f]  metadata [%.3f, %.3f]  "
                  "overshoot lo %+.5f hi %+.5f  (= %.2f / %.2f radius)"
                  % (a, f_lo[a], f_hi[a], bounds[0][a], bounds[1][a],
                     over_lo[a], over_hi[a], over_lo[a] / radius,
                     over_hi[a] / radius))
        if args.wall_margin >= 0.0 or args.wall_margin_mode == "scalar":
            m = max(args.wall_margin, 0.0)
            m_lo = np.full(dim, m)
            m_hi = np.full(dim, m)
            wall_info["margin_source"] = "scalar %.5f" % m
        else:
            safety = args.wall_margin_safety * spacing
            m_lo = over_lo + safety
            m_hi = over_hi + safety
            wall_info["margin_source"] = ("measured + %.3f*spacing (%.5f)"
                                          % (args.wall_margin_safety, safety))
        print("wall margins: lo %s  hi %s   (%s)"
              % (np.round(m_lo, 5).tolist(), np.round(m_hi, 5).tolist(),
                 wall_info["margin_source"]))
        walls = make_box_walls_2d(bounds, spacing, m_lo, m_hi,
                                  layers=args.wall_layers)
        inner_lo = bounds[0] - m_lo
        inner_hi = bounds[1] + m_hi
        print("wall particles: %d  (spacing=%.5f, layers=%d, innermost shell "
              "%s - %s)" % (len(walls), spacing, args.wall_layers,
                            np.round(inner_lo, 5).tolist(),
                            np.round(inner_hi, 5).tolist()))
        # The thing that went wrong on Water-3D, checked instead of assumed.
        encroach = []
        for a in range(dim):
            if inner_lo[a] > f_lo[a]:
                encroach.append([a, "lo"])
            if inner_hi[a] < f_hi[a]:
                encroach.append([a, "hi"])
        if encroach:
            print("WARNING: the innermost shell sits INSIDE the ground-truth "
                  "fluid region on faces %s -- this is the Water-3D 74 percent "
                  "failure mode" % encroach)
        wall_info.update({"n_walls": int(len(walls)), "spacing": spacing,
                          "layers": args.wall_layers,
                          "margin_lo": m_lo.tolist(),
                          "margin_hi": m_hi.tolist(),
                          "fluid_lo": f_lo.tolist(), "fluid_hi": f_hi.tolist(),
                          "encroaches": encroach})

    def build(files):
        return [WaterRampsDataset(f, radius, args.history, walls, args.stride)
                for f in files]

    train_sets, val_sets = build(train_files), build(val_files)
    print("train trajs=%d  valid trajs=%d" % (len(train_sets), len(val_sets)))
    print("  fluid/traj  %d..%d" % (min(ds.n_fluid for ds in train_sets),
                                    max(ds.n_fluid for ds in train_sets)))
    print("  ramp/traj   %d..%d  (%d of %d trajectories carry ramps)"
          % (min(ds.n_ramp for ds in train_sets),
             max(ds.n_ramp for ds in train_sets),
             sum(1 for ds in train_sets if ds.n_ramp > 0), len(train_sets)))
    print("  container walls %d  -> nodes/traj %d..%d"
          % (0 if walls is None else len(walls),
             min(ds.n_fluid + ds.n_boundary for ds in train_sets),
             max(ds.n_fluid + ds.n_boundary for ds in train_sets)))

    loader_kw = dict(batch_size=1, num_workers=args.num_workers,
                     persistent_workers=args.num_workers > 0,
                     pin_memory=True)
    train_loader = DataLoader(ConcatDataset(train_sets), shuffle=True,
                              **loader_kw)
    val_loader = DataLoader(ConcatDataset(val_sets), shuffle=False, **loader_kw)
    print("train samples=%d  val samples=%d"
          % (len(train_loader.dataset), len(val_loader.dataset)), flush=True)

    model = HybridPhysicsGNNv6(
        dim=dim,
        history=args.history,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_learned_attention=not args.no_attention,
        use_type_weighting=not args.no_type_weighting,
        boundary_importance=args.boundary_importance,
    ).to(device)
    print("model params: %d" % model.count_parameters())

    stat_sets = [Subset(ds, list(range(0, len(ds), max(1, len(ds) // 20))))
                 for ds in train_sets]
    stats = compute_normalizer_stats_anydim(stat_sets, args.history)
    model.set_normalizer_stats(
        stats["node_mean"], stats["node_std"],
        stats["edge_mean"], stats["edge_std"],
        stats["target_mean"], stats["target_std"])
    (log_dir / ("normalizer_" + args.tag + "_" + train_ts + ".json")).write_text(
        json.dumps({k: v_.tolist() for k, v_ in stats.items()}, indent=2))

    criterion = v4.V4Loss(model, dt=DT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    history = {"train": [], "val": [], "config": vars(args),
               "version": V6_ID, "dataset": "gns_waterramps",
               "dim": dim, "radius": radius, "dt": DT,
               "noise_std": noise_std, "walls": wall_info,
               "n_params": model.count_parameters(),
               "train_trajs": [f.name for f in train_files],
               "valid_trajs": [f.name for f in val_files]}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr, trc = v4.run_epoch(model, train_loader, criterion, optimizer,
                               device, train=True, noise_std=noise_std,
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
                        "radius": radius, "dt": DT, "dim": dim,
                        "walls": None if walls is None else walls,
                        "wall_info": wall_info,
                        "version": V6_ID}, ckpt_path)
        print("[v6/%s] ep %3d/%d train=%.5f val=%.5f  "
              "d=%.4f p=%.4f m=%.4f e=%.4f  (%.1fs) %s"
              % (args.tag, epoch, args.epochs, tr, va,
                 vac["delta_loss"], vac["pos_loss"], vac["momentum_loss"],
                 vac["energy_loss"], time.time() - ep_t,
                 "(best)" if improved else ""), flush=True)
        log_path.write_text(json.dumps(history, indent=2))

    print("\nDone in %.1f min   best val = %.5f"
          % ((time.time() - t0) / 60, best_val))
    print("Best ckpt: %s" % ckpt_path)


if __name__ == "__main__":
    main()
