"""
Train GNN on Dynamic Rigid Body Data (Two-way Coupling)

This script trains the GNN model specifically on the new dynamic rigid body
data generated with two-way coupling physics.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time

# Import model
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gnn import MeshGraphNet

# Import dataset and loss from train_gnn.py
exec(open(Path(__file__).parent / 'train_gnn.py').read(), globals())


def main():
    """Main training function for dynamic rigid body data"""

    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1  # Graph size varies, use batch_size=1
    num_epochs = 10
    learning_rate = 1e-4
    radius = 0.3

    print("="*70)
    print("TRAINING GNN ON DYNAMIC TWO-WAY COUPLING DATA")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Graph radius: {radius}")
    print("="*70)

    # Load datasets - ONLY dynamic scenarios
    data_root = Path("datasets/flip_simulations")

    # Dynamic scenario types (with two-way coupling physics)
    dynamic_types = ['dynamic_falling_ball', 'dynamic_floating_ball',
                     'dynamic_heavy_box', 'dynamic_multi_ball',
                     'dynamic_bouncing_sphere']

    dynamic_scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and any(dyn_type in d.name for dyn_type in dynamic_types)
    ])

    print(f"\nFound {len(dynamic_scenario_dirs)} DYNAMIC scenario directories")
    for d in dynamic_scenario_dirs[:5]:
        print(f"  - {d.name}")
    if len(dynamic_scenario_dirs) > 5:
        print(f"  ... and {len(dynamic_scenario_dirs) - 5} more")

    # Create datasets
    train_datasets = []
    val_datasets = []

    for i, scenario_dir in enumerate(dynamic_scenario_dirs):
        try:
            dataset = RigidBodySPHDataset(scenario_dir, radius=radius)
            if len(dataset) > 0:
                # 80/20 split
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
            continue

    if len(train_datasets) == 0:
        print("ERROR: No training data loaded!")
        return

    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = MeshGraphNet(
        node_in_dim=7,  # pos(3) + vel(3) + type(1)
        edge_in_dim=3,  # relative position
        node_out_dim=3,  # predicted velocity
        hidden_dim=128,
        num_layers=10
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = RigidBodyPhysicsLoss(
        vel_weight=1.0,
        pos_weight=0.1,
        momentum_weight=0.05
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    model_path = Path("models/gnn_dynamic_rigid_body_best.pt")
    model_path.parent.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_components = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"    - Velocity: {train_components['vel_loss']:.6f}")
        print(f"    - Position: {train_components['pos_loss']:.6f}")
        print(f"    - Momentum: {train_components['momentum_loss']:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"    - Velocity: {val_components['vel_loss']:.6f}")
        print(f"    - Position: {val_components['pos_loss']:.6f}")
        print(f"    - Momentum: {val_components['momentum_loss']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, model_path)
            print(f"  *** New best model saved! (val_loss: {val_loss:.6f})")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
