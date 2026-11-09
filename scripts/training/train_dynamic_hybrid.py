"""
Train Hybrid Model on Dynamic Rigid Body Data (Two-way Coupling)
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_gnn import HybridPhysicsGNN
from train_hybrid import RigidBodySPHDataset, RigidBodyPhysicsLoss, train_epoch, validate
from torch.utils.data import ConcatDataset, DataLoader
import time


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4
    radius = 0.3

    print("="*70)
    print("TRAINING HYBRID ON DYNAMIC TWO-WAY COUPLING DATA")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*70)

    # Load ONLY dynamic data
    data_root = Path("datasets/flip_simulations")
    dynamic_types = ['dynamic_falling_ball', 'dynamic_floating_ball',
                     'dynamic_heavy_box', 'dynamic_multi_ball',
                     'dynamic_bouncing_sphere']

    dynamic_scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and any(dyn_type in d.name for dyn_type in dynamic_types)
    ])

    print(f"\nFound {len(dynamic_scenario_dirs)} DYNAMIC scenarios")

    train_datasets = []
    val_datasets = []

    for scenario_dir in dynamic_scenario_dirs:
        try:
            dataset = RigidBodySPHDataset(scenario_dir, radius=radius)
            if len(dataset) > 0:
                split_idx = int(0.8 * len(dataset))
                train_size = split_idx
                val_size = len(dataset) - split_idx

                train_subset, val_subset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

                train_datasets.append(train_subset)
                val_datasets.append(val_subset)
        except Exception as e:
            print(f"Error loading {scenario_dir.name}: {e}")

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Hybrid Model
    model = HybridPhysicsGNN(
        node_in_dim=7,
        edge_in_dim=3,
        node_out_dim=3,
        hidden_dim=64,
        num_layers=5,
        use_learned_attention=True,
        use_type_weighting=True,
        boundary_importance=2.5  # From GAT analysis
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = RigidBodyPhysicsLoss(vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    model_path = Path("models/hybrid_dynamic_rigid_body_best.pt")
    model_path.parent.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_comp = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"  *** Best model saved! (val_loss: {val_loss:.6f})")

    print("\n" + "="*70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
