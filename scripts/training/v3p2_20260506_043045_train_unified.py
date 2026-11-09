"""
Unified training driver for v2 dataset (3 architectures × 3 losses = 9 configs).

Reads `datasets/flip_simulations_v3p2_20260506_043045/` (no existing code modified).

Saves checkpoints to:
  models/{arch}_{loss}_{train_timestamp}_v3p2_20260506_043045.pt

Logs to:
  logs/v3p2_20260506_043045/training_{arch}_{loss}.log

Usage:
  python v3p2_20260506_043045_train_unified.py --arch hybrid --loss improved --epochs 100
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

from src.models.gnn import MeshGraphNet
from src.models.gat import GraphAttentionNet
from src.models.hybrid_gnn import HybridPhysicsGNN
from src.models.hybrid_gnn_tuned import HybridPhysicsGNNTuned
from scripts.training.improved_physics_loss import ImprovedPhysicsLoss


DATASET_ID = "v3p2_20260506_043045"
DATA_DIR = ROOT / "datasets" / f"flip_simulations_{DATASET_ID}"


# =============================================================================
# Dataset (functionally identical to v1's RigidBodySPHDataset; format unchanged)
# =============================================================================
class V2Dataset(Dataset):
    def __init__(self, data_dir, radius=0.3, cache=True):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))
        self._cache = {} if cache else None
        if len(self.frame_files) < 2:
            raise ValueError(f"Need >=2 frames in {data_dir}")

    def __len__(self):
        return len(self.frame_files) - 1

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        d_t = np.load(self.frame_files[idx])
        d_t1 = np.load(self.frame_files[idx + 1])
        pos_t = d_t["positions"]; vel_t = d_t["velocities"]
        ptypes = d_t["particle_types"]
        pos_t1 = d_t1["positions"]; vel_t1 = d_t1["velocities"]
        edge_index, edge_attr = self._build_graph(pos_t)
        node_feats = np.concatenate([pos_t, vel_t, ptypes[:, None]], axis=1)
        sample = {
            "node_features": torch.FloatTensor(node_feats),
            "edge_index": torch.LongTensor(edge_index),
            "edge_attr": torch.FloatTensor(edge_attr),
            "pos_t": torch.FloatTensor(pos_t),
            "pos_t1": torch.FloatTensor(pos_t1),
            "vel_t1": torch.FloatTensor(vel_t1),
            "particle_types": torch.FloatTensor(ptypes),
        }
        if self._cache is not None:
            self._cache[idx] = sample
        return sample

    def _build_graph(self, positions):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type="ndarray")
        if len(pairs) == 0:
            return np.array([[0], [0]]), np.array([[0.0, 0.0, 0.0]])
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
        feats = positions[edges[1]] - positions[edges[0]]
        return edges, feats


# =============================================================================
# Loss variants
# =============================================================================
class VanillaLoss(nn.Module):
    """Velocity MSE only (fluid particles only)."""
    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types):
        is_fluid = particle_types < 0.5
        vel_loss = torch.mean((pred_vel[is_fluid] - true_vel[is_fluid]) ** 2)
        return vel_loss, {
            "vel_loss": vel_loss.item(),
            "pos_loss": 0.0, "momentum_loss": 0.0, "energy_loss": 0.0,
        }


class PhysicsAwareLoss(nn.Module):
    """v1-style RigidBodyPhysicsLoss (vel + pos[buggy dt²] + momentum)."""
    def __init__(self, vel_w=1.0, pos_w=0.1, mom_w=0.05, dt=0.01):
        super().__init__()
        self.vel_w, self.pos_w, self.mom_w, self.dt = vel_w, pos_w, mom_w, dt

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types):
        is_fluid = particle_types < 0.5
        v_hat = pred_vel[is_fluid]; v_true = true_vel[is_fluid]
        x_t = pos_t[is_fluid]; x_t1 = pos_t1[is_fluid]
        vel = torch.mean((v_hat - v_true) ** 2)
        pos_pred = x_t + v_hat * self.dt
        pos = torch.mean((pos_pred - x_t1) ** 2)
        n = v_hat.shape[0]
        if n > 0:
            p_hat = v_hat.sum(dim=0) / n
            p_true = v_true.sum(dim=0) / n
            mom = torch.mean((p_hat - p_true) ** 2)
        else:
            mom = torch.tensor(0.0, device=pred_vel.device)
        total = self.vel_w * vel + self.pos_w * pos + self.mom_w * mom
        return total, {"vel_loss": vel.item(), "pos_loss": pos.item(),
                       "momentum_loss": mom.item(), "energy_loss": 0.0}


def make_loss(loss_name):
    if loss_name == "vanilla":   return VanillaLoss()
    if loss_name == "physics":   return PhysicsAwareLoss()
    if loss_name == "improved":  return ImprovedPhysicsLoss(
        vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, energy_weight=0.05, dt=0.01)
    raise ValueError(loss_name)


# =============================================================================
# Model factory
# =============================================================================
def make_model(arch_name, hidden_dim=64, num_layers=5):
    if arch_name == "gnn":
        return MeshGraphNet(input_node_dim=7, input_edge_dim=3,
                            hidden_dim=hidden_dim, output_dim=3, num_layers=num_layers)
    if arch_name == "gat":
        return GraphAttentionNet(input_node_dim=7, input_edge_dim=3,
                                 hidden_dim=hidden_dim, output_dim=3, num_layers=num_layers)
    if arch_name == "hybrid":
        return HybridPhysicsGNN(input_node_dim=7, input_edge_dim=3,
                                hidden_dim=hidden_dim, output_dim=3, num_layers=num_layers,
                                use_learned_attention=True, use_type_weighting=True,
                                boundary_importance=2.0)
    if arch_name == "hybrid_tuned":
        return HybridPhysicsGNNTuned(input_node_dim=7, input_edge_dim=3,
                                     hidden_dim=hidden_dim, output_dim=3, num_layers=num_layers,
                                     use_learned_attention=True, use_type_weighting=True,
                                     boundary_importance=2.0, dropout=0.1)
    raise ValueError(arch_name)


# =============================================================================
# Train / validate epoch
# =============================================================================
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total = 0.0
    comp = {"vel_loss": 0.0, "pos_loss": 0.0, "momentum_loss": 0.0, "energy_loss": 0.0}
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            nf = batch["node_features"].squeeze(0).to(device)
            ei = batch["edge_index"].squeeze(0).to(device)
            ea = batch["edge_attr"].squeeze(0).to(device)
            pt = batch["pos_t"].squeeze(0).to(device)
            pt1 = batch["pos_t1"].squeeze(0).to(device)
            vt1 = batch["vel_t1"].squeeze(0).to(device)
            types = batch["particle_types"].squeeze(0).to(device)
            pred = model(nf, ei, ea)
            loss, c = criterion(pred, vt1, pt, pt1, types)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            for k in comp: comp[k] += c.get(k, 0.0)
            n_batches += 1
    n = max(1, n_batches)
    return total / n, {k: v / n for k, v in comp.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["gnn", "gat", "hybrid", "hybrid_tuned"], required=True)
    p.add_argument("--loss", choices=["vanilla", "physics", "improved"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--out-dir", type=str, default=str(ROOT / "models"))
    args = p.parse_args()

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{args.arch}_{args.loss}_{train_ts}_{DATASET_ID}.pt"
    ckpt_path = Path(args.out_dir) / ckpt_name
    log_dir = ROOT / "logs" / DATASET_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{args.arch}_{args.loss}_{train_ts}.json"

    print(f"\n{'='*70}\nTRAIN  arch={args.arch}  loss={args.loss}  "
          f"epochs={args.epochs}  data={args.data_dir}\n"
          f"  ckpt -> {ckpt_path}\n  log  -> {log_path}\n{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all 30 sims, 80/20 train-val split per sim
    data_root = Path(args.data_dir)
    sim_dirs = sorted(d for d in data_root.iterdir()
                       if d.is_dir() and d.name.startswith("sph_"))
    print(f"Found {len(sim_dirs)} sims in {data_root}")
    train_subs, val_subs = [], []
    for sd in sim_dirs:
        ds = V2Dataset(sd, radius=args.radius, cache=True)
        if len(ds) > 0:
            split = int(0.8 * len(ds))
            tr, va = torch.utils.data.random_split(ds, [split, len(ds) - split])
            train_subs.append(tr); val_subs.append(va)

    train_loader = DataLoader(ConcatDataset(train_subs), batch_size=1, shuffle=True)
    val_loader   = DataLoader(ConcatDataset(val_subs),   batch_size=1, shuffle=False)
    print(f"Train sequences: {len(train_loader.dataset)}  "
          f"Val sequences: {len(val_loader.dataset)}")

    model = make_model(args.arch, args.hidden_dim, args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    criterion = make_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    history = {"train": [], "val": [], "config": vars(args), "n_params": n_params}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr_loss, tr_comp = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_comp = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        history["train"].append({"epoch": epoch, "loss": tr_loss, **tr_comp})
        history["val"].append({"epoch": epoch, "loss": va_loss, **va_comp})

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            torch.save({"state_dict": model.state_dict(),
                        "config": vars(args), "epoch": epoch,
                        "val_loss": va_loss, "n_params": n_params},
                       ckpt_path)

        print(f"[{args.arch}/{args.loss}] ep {epoch:>3}/{args.epochs} "
              f"train={tr_loss:.5f} val={va_loss:.5f}  "
              f"v={va_comp['vel_loss']:.4f} p={va_comp['pos_loss']:.4f} "
              f"m={va_comp['momentum_loss']:.4f} e={va_comp['energy_loss']:.4f}  "
              f"({time.time() - ep_t:.1f}s)  {'(best)' if improved else ''}")

        # Persist log every epoch
        log_path.write_text(json.dumps(history, indent=2))

    print(f"\nDone in {(time.time() - t0)/60:.1f} min   best val = {best_val:.5f}")
    print(f"Best ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
