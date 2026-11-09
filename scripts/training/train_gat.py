"""
Train Graph Attention Network (GAT) with rigid body interactions

Uses the same dataset as GNN but with attention mechanism for better
feature aggregation from neighboring particles.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time

# Import GAT model
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gat import GraphAttentionNet


class RigidBodySPHDataset(Dataset):
    """Dataset for SPH simulation with rigid bodies"""

    def __init__(self, data_dir, radius=0.3):
        self.data_dir = Path(data_dir)
        self.radius = radius

        # Load all frame files
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))

        if len(self.frame_files) < 2:
            raise ValueError(f"Need at least 2 frames in {data_dir}, found {len(self.frame_files)}")

        print(f"  Loaded {len(self.frame_files)} frames from {data_dir}")

    def __len__(self):
        return len(self.frame_files) - 1

    def __getitem__(self, idx):
        # Load consecutive frames
        data_t = np.load(self.frame_files[idx])
        data_t1 = np.load(self.frame_files[idx + 1])

        # Extract data
        pos_t = data_t['positions']
        vel_t = data_t['velocities']
        particle_types = data_t['particle_types']  # 0=fluid, 1=boundary
        n_fluid = data_t['n_fluid']

        pos_t1 = data_t1['positions']
        vel_t1 = data_t1['velocities']

        # Build graph (all particles participate)
        edge_index, edge_attr = self.build_graph(pos_t)

        # Node features: [pos, vel, particle_type]
        node_features = np.concatenate([
            pos_t,
            vel_t,
            particle_types[:, np.newaxis]  # Add particle type as feature
        ], axis=1)

        return {
            'node_features': torch.FloatTensor(node_features),
            'edge_index': torch.LongTensor(edge_index),
            'edge_attr': torch.FloatTensor(edge_attr),
            'pos_t': torch.FloatTensor(pos_t),
            'vel_t': torch.FloatTensor(vel_t),
            'pos_t1': torch.FloatTensor(pos_t1),
            'vel_t1': torch.FloatTensor(vel_t1),
            'particle_types': torch.FloatTensor(particle_types),
            'n_fluid': n_fluid
        }

    def build_graph(self, positions):
        """Build k-NN graph"""
        tree = cKDTree(positions)
        edges = []
        edge_features = []

        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, self.radius)
            for j in neighbors:
                if i != j:
                    edges.append([i, j])
                    edge_features.append(positions[j] - positions[i])

        if len(edges) == 0:
            return np.array([[0], [0]]), np.array([[0, 0, 0]])

        return np.array(edges).T, np.array(edge_features)


class RigidBodyPhysicsLoss(nn.Module):
    """Physics-aware loss function for rigid body interactions"""

    def __init__(self, vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=0.01):
        super().__init__()
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.momentum_weight = momentum_weight
        self.dt = dt

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types, particle_mass=1.0):
        """Compute physics-aware loss (only for fluid particles)"""
        # Mask for fluid particles
        is_fluid = particle_types < 0.5  # 0 = fluid

        # Velocity loss (only for fluid particles)
        pred_vel_fluid = pred_vel[is_fluid]
        true_vel_fluid = true_vel[is_fluid]
        vel_loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)

        # Position consistency loss (only for fluid particles)
        pos_t_fluid = pos_t[is_fluid]
        pos_t1_fluid = pos_t1[is_fluid]
        predicted_pos_fluid = pos_t_fluid + pred_vel_fluid * self.dt
        pos_loss = torch.mean((predicted_pos_fluid - pos_t1_fluid) ** 2)

        # Momentum conservation (only for fluid particles)
        num_fluid = torch.sum(is_fluid).item()
        if num_fluid > 0:
            pred_momentum = torch.sum(pred_vel_fluid, dim=0) / num_fluid
            true_momentum = torch.sum(true_vel_fluid, dim=0) / num_fluid
            momentum_loss = torch.mean((pred_momentum - true_momentum) ** 2)
        else:
            momentum_loss = torch.tensor(0.0, device=pred_vel.device)

        # Total loss
        total_loss = (self.vel_weight * vel_loss +
                     self.pos_weight * pos_loss +
                     self.momentum_weight * momentum_loss)

        return total_loss, {
            'vel_loss': vel_loss.item(),
            'pos_loss': pos_loss.item(),
            'momentum_loss': momentum_loss.item()
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'vel_loss': 0, 'pos_loss': 0, 'momentum_loss': 0}

    for batch in dataloader:
        # Move to device and squeeze batch dimension (batch_size=1 for graphs)
        node_features = batch['node_features'].squeeze(0).to(device)
        edge_index = batch['edge_index'].squeeze(0).to(device)
        edge_attr = batch['edge_attr'].squeeze(0).to(device)
        pos_t = batch['pos_t'].squeeze(0).to(device)
        vel_t = batch['vel_t'].squeeze(0).to(device)
        pos_t1 = batch['pos_t1'].squeeze(0).to(device)
        vel_t1 = batch['vel_t1'].squeeze(0).to(device)
        particle_types = batch['particle_types'].squeeze(0).to(device)

        # Forward pass
        pred_vel = model(node_features, edge_index, edge_attr)

        # Compute loss
        loss, components = criterion(pred_vel, vel_t1, pos_t, pos_t1, particle_types)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for key in components:
            loss_components[key] += components[key]

    # Average
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    loss_components = {'vel_loss': 0, 'pos_loss': 0, 'momentum_loss': 0}

    with torch.no_grad():
        for batch in dataloader:
            node_features = batch['node_features'].squeeze(0).to(device)
            edge_index = batch['edge_index'].squeeze(0).to(device)
            edge_attr = batch['edge_attr'].squeeze(0).to(device)
            pos_t = batch['pos_t'].squeeze(0).to(device)
            vel_t = batch['vel_t'].squeeze(0).to(device)
            pos_t1 = batch['pos_t1'].squeeze(0).to(device)
            vel_t1 = batch['vel_t1'].squeeze(0).to(device)
            particle_types = batch['particle_types'].squeeze(0).to(device)

            pred_vel = model(node_features, edge_index, edge_attr)
            loss, components = criterion(pred_vel, vel_t1, pos_t, pos_t1, particle_types)

            total_loss += loss.item()
            for key in components:
                loss_components[key] += components[key]

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1  # Graph size varies, use batch_size=1
    num_epochs = 10
    learning_rate = 1e-4
    radius = 0.3

    print("="*70)
    print("TRAINING GAT WITH RIGID BODY INTERACTIONS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Graph radius: {radius}")
    print("="*70)

    # Load datasets
    data_root = Path("datasets/flip_simulations")

    # Only load rigid body scenarios
    rigid_body_types = ['falling_ball', 'paddle_wave', 'rotating_stirrer',
                        'moving_obstacle', 'double_paddle', 'sphere_bounce']

    rigid_scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and any(rb_type in d.name for rb_type in rigid_body_types)
    ])

    print(f"\nFound {len(rigid_scenario_dirs)} rigid body scenario directories")

    # Create datasets
    train_datasets = []
    val_datasets = []

    for i, scenario_dir in enumerate(rigid_scenario_dirs):
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

                print(f"  [{i+1}/{len(rigid_scenario_dirs)}] {scenario_dir.name}: "
                      f"{len(dataset)} sequences (train: {train_size}, val: {val_size})")
        except Exception as e:
            print(f"  [SKIP] {scenario_dir}: {e}")

    if len(train_datasets) == 0:
        print("\nNo datasets loaded! Make sure rigid body data generation completed.")
        return

    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"\nTotal training sequences: {len(train_dataset)}")
    print(f"Total validation sequences: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create GAT model
    model = GraphAttentionNet(
        input_node_dim=7,       # pos(3) + vel(3) + particle_type(1)
        input_edge_dim=3,       # relative position
        hidden_dim=64,
        output_dim=3,           # velocity prediction
        num_layers=5,
        num_heads=4,            # 4 attention heads
        dropout=0.1
    ).to(device)

    print(f"\nGAT Model:")
    print(f"  Hidden dim: 64")
    print(f"  Attention heads: 4")
    print(f"  Layers: 5")
    print(f"  Total parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = RigidBodyPhysicsLoss(
        vel_weight=1.0,
        pos_weight=0.1,
        momentum_weight=0.05,
        dt=0.01
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    save_path = Path("models/gat_rigid_body_sph_best.pt")
    save_path.parent.mkdir(exist_ok=True)

    # Loss history tracking
    train_losses = []
    val_losses = []

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_loss, train_components = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)

        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step(val_loss)

        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f} "
              f"(vel: {train_components['vel_loss']:.6f}, "
              f"pos: {train_components['pos_loss']:.9f}, "
              f"mom: {train_components['momentum_loss']:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} "
              f"(vel: {val_components['vel_loss']:.6f}, "
              f"pos: {val_components['pos_loss']:.9f}, "
              f"mom: {val_components['momentum_loss']:.6f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,  # Full loss history
                'val_losses': val_losses,      # Full loss history
                'config': {
                    'model': {
                        'input_node_dim': 7,
                        'input_edge_dim': 3,
                        'hidden_dim': 64,
                        'output_dim': 3,
                        'num_layers': 5,
                        'num_heads': 4,
                        'dropout': 0.1
                    },
                    'training': {
                        'radius': radius,
                        'dt': 0.01
                    }
                }
            }, save_path)
            print(f"  [SAVED] Best model to {save_path}")

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Model saved to: {save_path}")
    print("="*70)


if __name__ == "__main__":
    main()
