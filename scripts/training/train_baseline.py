"""
Train baseline GNN or GAT with the same 100-epoch protocol used for the
hybrid model. Reuses the vectorised+cached dataset, physics loss, and
train/validate loops from train_hybrid.py so results are directly
comparable.

Usage:
    python scripts/training/train_baseline.py --model gnn --epochs 100 \
        --save-name gnn_rigid_body_sph_100ep_best.pt
    python scripts/training/train_baseline.py --model gat --epochs 100 \
        --save-name gat_rigid_body_sph_100ep_best.pt
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

from train_hybrid import (
    RigidBodySPHDataset, RigidBodyPhysicsLoss,
    train_epoch, validate,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gnn", "gat", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-name", type=str, required=True)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vel-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=0.1,
                       help="Physics position-consistency weight. "
                            "Set to 0 to train on velocity MSE only "
                            "(still logged as diagnostic).")
    parser.add_argument("--momentum-weight", type=float, default=0.05,
                       help="Physics momentum-conservation weight. "
                            "Set to 0 to disable in training.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radius = args.radius
    dt = 0.01

    print("=" * 70)
    print(f"TRAINING BASELINE {args.model.upper()}  ({args.epochs} epochs)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Save name: {args.save_name}")

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
              f"{len(ds)} sequences (train={split}, val={len(ds)-split})")

    train_ds = ConcatDataset(train_subsets)
    val_ds = ConcatDataset(val_subsets)
    print(f"\nTotal train: {len(train_ds)}   Total val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- model / loss / optimizer ---
    model = build_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{args.model.upper()} parameters: {n_params:,}")

    criterion = RigidBodyPhysicsLoss(
        vel_weight=args.vel_weight,
        pos_weight=args.pos_weight,
        momentum_weight=args.momentum_weight,
        dt=dt,
    )
    print(f"\nLoss weights: vel={args.vel_weight}, "
          f"pos={args.pos_weight}, momentum={args.momentum_weight}")
    if args.pos_weight == 0.0 and args.momentum_weight == 0.0:
        print("  [MODE] Pure velocity MSE training "
              "(physics components still logged as diagnostics)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- training loop ---
    best_val = float("inf")
    train_losses, val_losses = [], []
    train_hist = {"vel_loss": [], "pos_loss": [], "momentum_loss": []}
    val_hist = {"vel_loss": [], "pos_loss": [], "momentum_loss": []}

    save_path = ROOT / "models" / args.save_name
    save_path.parent.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    t_total = time.time()

    for ep in range(args.epochs):
        t_ep = time.time()
        tl, tc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, vc = validate(model, val_loader, criterion, device)
        dt_ep = time.time() - t_ep

        train_losses.append(tl); val_losses.append(vl)
        for k in train_hist:
            train_hist[k].append(tc[k])
            val_hist[k].append(vc[k])
        scheduler.step(vl)

        print(f"\nEpoch {ep+1}/{args.epochs}  ({dt_ep:.1f}s)")
        print(f"  Train Loss: {tl:.6f} "
              f"(vel: {tc['vel_loss']:.6f}, pos: {tc['pos_loss']:.9f}, "
              f"mom: {tc['momentum_loss']:.6f})")
        print(f"  Val Loss:   {vl:.6f} "
              f"(vel: {vc['vel_loss']:.6f}, pos: {vc['pos_loss']:.9f}, "
              f"mom: {vc['momentum_loss']:.6f})")
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
                    "training": {"radius": radius, "dt": dt, "lr": args.lr},
                },
            }, save_path)
            print(f"  [SAVED] Best model to {save_path}")

    # Save final checkpoint with FULL history (not truncated at best epoch)
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
            "loss_weights": {
                "vel": args.vel_weight,
                "pos": args.pos_weight,
                "momentum": args.momentum_weight,
            },
            "training": {"radius": radius, "dt": dt, "lr": args.lr},
        },
    }, final_path)

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Training time       : {total:.1f}s ({total/60:.1f} min)")
    print(f"Best model saved to : {save_path}")
    print(f"Final model saved to: {final_path}  (full {args.epochs}-epoch history)")
    print("=" * 70)


if __name__ == "__main__":
    main()
