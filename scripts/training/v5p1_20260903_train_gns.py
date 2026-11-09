"""
v5p1 training - DeepMind GNS baseline (ICML 2020) on Water-3D, trained under
the SAME budget/data/seed as our v5 runs for a like-for-like comparison.

Version ID : v5p1_20260903
Purpose    : reviewer-grade comparison "proposed wall particles vs original
             GNS boundary handling". GNS represents the container ONLY through
             clipped boundary-distance node features (no wall particles); our
             proposed method uses type-weighted wall particles. Same 10 train /
             3 valid trajectories, stride 10, 100 epochs, seed 42.

Faithful GNS settings (paper defaults):
  hidden 128, 10 GN blocks with node+edge residuals, C = 5 velocity history,
  boundary-distance features clipped at the connectivity radius, acceleration
  target normalised by the dataset-provided stats (metadata.json), random-walk
  velocity noise with last-step std 6.7e-4, Adam with exponential lr decay
  1e-4 -> 1e-6 over 5M steps (effectively ~1e-4 at our scale), plain MSE loss.

Known deviations from the paper (unavoidable at this scale, reported openly):
  batch size 1 (paper 2) and 8k gradient steps instead of 20M.

Usage:
  python scripts/training/v5p1_20260903_train_gns.py --tag gns_e100
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.gns_baseline import GNSBaseline

V5P1_VERSION_ID = "v5p1_20260903"
DATA_DIR = ROOT / "datasets" / "gns_water3d_v5"
DT = 1.0            # GNS convention: velocity = per-step position delta


class GNSTrajectory(Dataset):
    """One Water-3D trajectory with the GNS sample contract (C=5 history)."""

    def __init__(self, traj_path, radius, history=5, stride=1, self_edges=False):
        pos = np.load(traj_path)["positions"].astype(np.float32)   # (T,N,3)
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]
        self.pos, self.vel = pos, vel
        self.radius = radius
        self.history = history
        # official connectivity_utils defaults to add_self_edges=True
        self.self_edges = self_edges
        # need t-(C-1) >= 0 and t+1 <= T-1
        self._idx = list(range(history - 1, pos.shape[0] - 1, max(1, stride)))

    def __len__(self):
        return len(self._idx)

    def build_graph(self, positions):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if len(pairs) == 0:
            edges = np.array([[0], [0]], dtype=np.int64)
        else:
            edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
        if self.self_edges:
            ids = np.arange(len(positions), dtype=np.int64)
            edges = np.concatenate([edges, np.stack([ids, ids])], axis=1)
        rel = (positions[edges[1]] - positions[edges[0]]) / self.radius
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        return edges, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def __getitem__(self, i):
        t = self._idx[i]
        C = self.history
        hist = self.vel[t - C + 1: t + 1]                     # (C,N,3)
        edge_index, edge_attr = self.build_graph(self.pos[t])
        return {
            "pos_t": torch.from_numpy(self.pos[t].copy()),
            "vel_hist": torch.from_numpy(hist.copy()),
            "vel_t1": torch.from_numpy(self.vel[t + 1].copy()),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
        }


def collate_one(batch):
    return batch[0]


def apply_gns_noise(vel_hist, pos_t, noise_std, rng):
    """GNS random-walk noise: accumulated last-step std = noise_std.

    Velocity history gets a random walk; the current position drifts by the
    accumulated velocity noise (dt = 1), like reading_utils.py.
    """
    C = vel_hist.shape[0]
    step_std = noise_std / (C - 1) ** 0.5 if C > 1 else noise_std
    steps = torch.randn(vel_hist.shape, generator=rng) * step_std
    steps[0].zero_()
    walk = torch.cumsum(steps, dim=0)
    pos_shift = walk.sum(dim=0)          # integrated velocity noise, dt = 1
    return vel_hist + walk, pos_t + pos_shift


def run_epoch(model, loader, optimizer, device, bounds_lo, bounds_hi, radius,
              train, noise_std=0.0, rng=None):
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            pos = batch["pos_t"]
            hist = batch["vel_hist"]
            if train and noise_std > 0:
                hist, pos = apply_gns_noise(hist, pos, noise_std, rng)
            pos = pos.to(device)
            hist = hist.to(device)
            ei = batch["edge_index"].to(device)
            ea = batch["edge_attr"].to(device)
            v_next = batch["vel_t1"].to(device)

            nf = model.build_features(pos, hist, bounds_lo, bounds_hi, radius)
            acc_pred_n = model(nf, ei, ea)
            acc_true = v_next - hist[-1]
            acc_true_n = (acc_true - model.acc_mean) / model.acc_std
            loss = torch.mean((acc_pred_n - acc_true_n) ** 2)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser(description="GNS baseline on Water-3D")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-decay-steps", type=float, default=5e6,
                   help="paper: exponential decay over 5M steps")
    p.add_argument("--history", type=int, default=5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=6.7e-4)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--self-edges", action="store_true",
                   help="include (i,i) edges as in the official connectivity_utils default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-trajs", type=int, default=0)
    p.add_argument("--bounds-margin", type=float, default=0.0,
                   help="push the wall-distance reference box OUTWARD by this "
                        "much. The Water-3D metadata bounds sit inside the real "
                        "container, so 0.010 gives the distance features the "
                        "same corrected geometry the wall particles get.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    radius = meta["default_connectivity_radius"]
    lo = torch.tensor([b[0] for b in meta["bounds"]], dtype=torch.float32) \
        - args.bounds_margin
    hi = torch.tensor([b[1] for b in meta["bounds"]], dtype=torch.float32) \
        + args.bounds_margin

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = ROOT / "models" / ("gns_" + args.tag + "_" + train_ts
                                   + "_" + V5P1_VERSION_ID + ".pt")
    log_dir = ROOT / "logs" / V5P1_VERSION_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("training_" + args.tag + "_" + train_ts + ".json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bar = "=" * 70
    print("\n" + bar)
    print("V5P1 GNS BASELINE  tag=%s  version=%s" % (args.tag, V5P1_VERSION_ID))
    print("  faithful ICML2020 config: hidden=%d blocks=%d C=%d noise=%g"
          % (args.hidden, args.num_blocks, args.history, args.noise_std))
    print("  boundary = clipped distance features (NO wall particles)")
    print("  self_edges=%s" % args.self_edges)
    print("  device=%s epochs=%d stride=%d  ckpt -> %s"
          % (device, args.epochs, args.stride, ckpt_path.name))
    print(bar)

    def load_split(split):
        files = sorted((DATA_DIR / split).glob("traj_*.npz"))
        if args.limit_trajs:
            files = files[: args.limit_trajs]
        return [GNSTrajectory(f, radius, args.history, args.stride,
                              self_edges=args.self_edges)
                for f in files]

    train_sets, val_sets = load_split("train"), load_split("valid")
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=1,
                              shuffle=True, collate_fn=collate_one)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=1,
                            shuffle=False, collate_fn=collate_one)
    print("Train samples: %d  Val samples: %d"
          % (len(train_loader.dataset), len(val_loader.dataset)))

    model = GNSBaseline(history=args.history, hidden=args.hidden,
                        num_blocks=args.num_blocks).to(device)
    model.set_stats(meta["vel_mean"], meta["vel_std"],
                    meta["acc_mean"], meta["acc_std"])
    print("Model params: %d" % model.count_parameters())

    lo_d, hi_d = lo.to(device), hi.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader.dataset)
    gamma = (args.lr_min / args.lr) ** (steps_per_epoch / args.lr_decay_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {"train": [], "val": [], "config": vars(args),
               "version": V5P1_VERSION_ID, "dataset": "gns_water3d",
               "radius": radius, "dt": DT,
               "n_params": model.count_parameters()}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, lo_d, hi_d,
                       radius, train=True, noise_std=args.noise_std, rng=rng)
        va = run_epoch(model, val_loader, optimizer, device, lo_d, hi_d,
                       radius, train=False)
        scheduler.step()
        history["train"].append({"epoch": epoch, "loss": tr})
        history["val"].append({"epoch": epoch, "loss": va})
        improved = va < best_val
        if improved:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "model_config": {"history": args.history,
                                         "hidden": args.hidden,
                                         "num_blocks": args.num_blocks},
                        "train_config": vars(args),
                        "epoch": epoch, "val_loss": va,
                        "radius": radius, "dt": DT,
                        "self_edges": args.self_edges,
                        "bounds_lo": lo.tolist(), "bounds_hi": hi.tolist(),
                        "version": V5P1_VERSION_ID}, ckpt_path)
        print("[v5p1/%s] ep %3d/%d train=%.5f val=%.5f  (%.1fs) %s"
              % (args.tag, epoch, args.epochs, tr, va,
                 time.time() - ep_t, "(best)" if improved else ""), flush=True)
        log_path.write_text(json.dumps(history, indent=2))

    print("\nDone in %.1f min   best val = %.5f" % ((time.time() - t0) / 60, best_val))
    print("Best ckpt: %s" % ckpt_path)


if __name__ == "__main__":
    main()
