"""
Train GNN / GAT / Hybrid for 100 epochs with the Improved Physics Loss
(scripts/training/improved_physics_loss.py).

Tracks and saves per-epoch vel / pos / momentum / energy components as
diagnostic history. Dataset and dataset-level optimisations (vectorised
graph build + cache) are reused from train_hybrid.py so results are
directly comparable to prior runs.

Usage:
    python scripts/training/train_improved.py --model gnn    \
        --save-name gnn_improved_100ep_best.pt
    python scripts/training/train_improved.py --model gat    \
        --save-name gat_improved_100ep_best.pt
    python scripts/training/train_improved.py --model hybrid \
        --save-name hybrid_improved_100ep_best.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from train_hybrid import RigidBodySPHDataset
from improved_physics_loss import ImprovedPhysicsLoss


COMPONENT_KEYS = ('vel_loss', 'pos_loss', 'momentum_loss', 'energy_loss')


def build_model(kind: str, device):
    if kind == "gnn":
        from src.models.gnn import MeshGraphNet
        model = MeshGraphNet(
            input_node_dim=7, input_edge_dim=3,
            hidden_dim=64, output_dim=3, num_layers=5,
        )
    elif kind == "gat":
        from src.models.gat import GraphAttentionNet
        model = GraphAttentionNet(
            input_node_dim=7, input_edge_dim=3,
            hidden_dim=64, output_dim=3, num_layers=5,
            num_heads=4, dropout=0.1,
        )
    elif kind == "hybrid":
        from src.models.hybrid_gnn import HybridPhysicsGNN
        model = HybridPhysicsGNN(
            input_node_dim=7, input_edge_dim=3,
            hidden_dim=64, output_dim=3, num_layers=5,
            use_learned_attention=True, use_type_weighting=True,
            boundary_importance=2.0,
        )
    else:
        raise ValueError(f"Unknown model: {kind}")
    return model.to(device)


def _run(model, loader, optimizer, criterion, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    comps = {k: 0.0 for k in COMPONENT_KEYS}
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            nf = batch['node_features'].squeeze(0).to(device)
            ei = batch['edge_index'].squeeze(0).to(device)
            ea = batch['edge_attr'].squeeze(0).to(device)
            pt = batch['pos_t'].squeeze(0).to(device)
            pt1 = batch['pos_t1'].squeeze(0).to(device)
            vt1 = batch['vel_t1'].squeeze(0).to(device)
            types = batch['particle_types'].squeeze(0).to(device)

            pred_vel = model(nf, ei, ea)
            loss, c = criterion(pred_vel, vt1, pt, pt1, types)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item()
            for k in COMPONENT_KEYS:
                comps[k] += c[k]

    n = len(loader)
    return total / n, {k: v / n for k, v in comps.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gnn", "gat", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-name", type=str, required=True)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vel-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=0.1)
    parser.add_argument("--momentum-weight", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.01
    radius = args.radius

    print("=" * 70)
    print(f"IMPROVED PHYSICS LOSS TRAINING  -  {args.model.upper()}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Loss weights: vel={args.vel_weight}, pos={args.pos_weight}, "
          f"mom={args.momentum_weight}, energy={args.energy_weight}")
    print(f"Save name   : {args.save_name}")

    # --- data ---
    data_root = ROOT / "datasets" / "flip_simulations"
    rigid_types = ["falling_ball", "paddle_wave", "rotating_stirrer",
                   "moving_obstacle", "double_paddle", "sphere_bounce"]
    scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("sph_")
        and any(rt in d.name for rt in rigid_types)
    ])
    print(f"Found {len(scenario_dirs)} sph_ scenarios")

    train_subsets, val_subsets = [], []
    for i, sdir in enumerate(scenario_dirs):
        ds = RigidBodySPHDataset(sdir, radius=radius)
        if len(ds) == 0:
            continue
        split = int(0.8 * len(ds))
        tr, va = torch.utils.data.random_split(ds, [split, len(ds) - split])
        train_subsets.append(tr)
        val_subsets.append(va)
        print(f"  [{i+1}/{len(scenario_dirs)}] {sdir.name}: "
              f"{len(ds)} seq (train={split}, val={len(ds)-split})")

    train_ds = ConcatDataset(train_subsets)
    val_ds = ConcatDataset(val_subsets)
    print(f"\nTotal train: {len(train_ds)}   val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- model / loss / optim ---
    model = build_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{args.model.upper()} parameters: {n_params:,}")

    criterion = ImprovedPhysicsLoss(
        vel_weight=args.vel_weight,
        pos_weight=args.pos_weight,
        momentum_weight=args.momentum_weight,
        energy_weight=args.energy_weight,
        dt=dt,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- training loop ---
    best_val = float("inf")
    train_losses, val_losses = [], []
    train_hist = {k: [] for k in COMPONENT_KEYS}
    val_hist = {k: [] for k in COMPONENT_KEYS}

    save_path = ROOT / "models" / args.save_name
    save_path.parent.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    t_total = time.time()

    for ep in range(args.epochs):
        t_ep = time.time()
        tl, tc = _run(model, train_loader, optimizer, criterion, device, train=True)
        vl, vc = _run(model, val_loader, optimizer, criterion, device, train=False)
        dt_ep = time.time() - t_ep

        train_losses.append(tl); val_losses.append(vl)
        for k in COMPONENT_KEYS:
            train_hist[k].append(tc[k])
            val_hist[k].append(vc[k])
        scheduler.step(vl)

        print(f"\nEpoch {ep+1}/{args.epochs}  ({dt_ep:.1f}s)")
        print(f"  Train Loss: {tl:.6f}  "
              f"(vel:{tc['vel_loss']:.6f}, pos:{tc['pos_loss']:.6f}, "
              f"mom:{tc['momentum_loss']:.6f}, en:{tc['energy_loss']:.6f})")
        print(f"  Val   Loss: {vl:.6f}  "
              f"(vel:{vc['vel_loss']:.6f}, pos:{vc['pos_loss']:.6f}, "
              f"mom:{vc['momentum_loss']:.6f}, en:{vc['energy_loss']:.6f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": tl, "val_loss": vl,
                "train_losses": train_losses, "val_losses": val_losses,
                "train_components_hist": train_hist,
                "val_components_hist": val_hist,
                "config": {
                    "model_kind": args.model,
                    "loss_kind": "improved_physics",
                    "loss_weights": {
                        "vel": args.vel_weight,
                        "pos": args.pos_weight,
                        "momentum": args.momentum_weight,
                        "energy": args.energy_weight,
                    },
                    "training": {"radius": radius, "dt": dt, "lr": args.lr},
                },
            }, save_path)
            print(f"  [SAVED] best to {save_path}")

    # Final checkpoint with full 100-epoch history
    final_path = save_path.with_name(save_path.stem + "_final" + save_path.suffix)
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_components_hist": train_hist,
        "val_components_hist": val_hist,
        "config": {
            "model_kind": args.model,
            "loss_kind": "improved_physics",
            "loss_weights": {
                "vel": args.vel_weight,
                "pos": args.pos_weight,
                "momentum": args.momentum_weight,
                "energy": args.energy_weight,
            },
            "training": {"radius": radius, "dt": dt, "lr": args.lr},
        },
    }, final_path)

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss     : {best_val:.6f}")
    print(f"Training time     : {total:.1f}s ({total/60:.1f} min)")
    print(f"Best model saved  : {save_path}")
    print(f"Final model saved : {final_path} (full {args.epochs}-epoch history)")
    print("=" * 70)


if __name__ == "__main__":
    main()
