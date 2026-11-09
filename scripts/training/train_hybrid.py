"""
Train Hybrid Physics-Guided GNN

Combines MeshGraphNet with GAT-inspired attention mechanisms.
Uses insights from GAT attention analysis to guide training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time
import json

# Import hybrid model
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_gnn import HybridPhysicsGNN


class RigidBodySPHDataset(Dataset):
    """Dataset for SPH simulation with rigid bodies"""

    def __init__(self, data_dir, radius=0.3, cache=True):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))
        self._cache = {} if cache else None

        if len(self.frame_files) < 2:
            raise ValueError(f"Need at least 2 frames in {data_dir}, found {len(self.frame_files)}")

        print(f"  Loaded {len(self.frame_files)} frames from {data_dir}")

    def __len__(self):
        return len(self.frame_files) - 1

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        data_t = np.load(self.frame_files[idx])
        data_t1 = np.load(self.frame_files[idx + 1])

        pos_t = data_t['positions']
        vel_t = data_t['velocities']
        particle_types = data_t['particle_types']
        n_fluid = data_t['n_fluid']

        pos_t1 = data_t1['positions']
        vel_t1 = data_t1['velocities']

        edge_index, edge_attr = self.build_graph(pos_t)

        node_features = np.concatenate([
            pos_t,
            vel_t,
            particle_types[:, np.newaxis]
        ], axis=1)

        sample = {
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
        if self._cache is not None:
            self._cache[idx] = sample
        return sample

    def build_graph(self, positions):
        """Build k-NN graph (vectorised via cKDTree.query_pairs)."""
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type='ndarray')
        if len(pairs) == 0:
            return np.array([[0], [0]]), np.array([[0.0, 0.0, 0.0]])
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
        edge_features = positions[edges[1]] - positions[edges[0]]
        return edges, edge_features


class RigidBodyPhysicsLoss(nn.Module):
    """Physics-aware loss function"""

    def __init__(self, vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=0.01):
        super().__init__()
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.momentum_weight = momentum_weight
        self.dt = dt

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types, particle_mass=1.0):
        is_fluid = particle_types < 0.5

        pred_vel_fluid = pred_vel[is_fluid]
        true_vel_fluid = true_vel[is_fluid]
        vel_loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)

        pos_t_fluid = pos_t[is_fluid]
        pos_t1_fluid = pos_t1[is_fluid]
        predicted_pos_fluid = pos_t_fluid + pred_vel_fluid * self.dt
        pos_loss = torch.mean((predicted_pos_fluid - pos_t1_fluid) ** 2)

        num_fluid = torch.sum(is_fluid).item()
        if num_fluid > 0:
            pred_momentum = torch.sum(pred_vel_fluid, dim=0) / num_fluid
            true_momentum = torch.sum(true_vel_fluid, dim=0) / num_fluid
            momentum_loss = torch.mean((pred_momentum - true_momentum) ** 2)
        else:
            momentum_loss = torch.tensor(0.0, device=pred_vel.device)

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

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for key in components:
            loss_components[key] += components[key]

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
    import argparse

    parser = argparse.ArgumentParser(description='Train Hybrid GNN')
    parser.add_argument('--insights', type=str, default=None,
                       help='Path to GAT insights JSON (optional)')
    parser.add_argument('--boundary-importance', type=float, default=2.0,
                       help='Boundary importance multiplier')
    parser.add_argument('--no-learned-attention', action='store_true',
                       help='Disable learned attention gates')
    parser.add_argument('--no-type-weighting', action='store_true',
                       help='Disable physics-based type weighting')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=5,
                       help='Number of message passing layers')
    parser.add_argument('--save-name', type=str,
                       default='hybrid_rigid_body_sph_best.pt',
                       help='Filename under models/ to save best checkpoint')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    learning_rate = 1e-4
    radius = 0.3

    print("="*70)
    print("TRAINING HYBRID PHYSICS-GUIDED GNN")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Layers: {args.num_layers}")
    print(f"Learned attention: {not args.no_learned_attention}")
    print(f"Type weighting: {not args.no_type_weighting}")

    # Load boundary importance from GAT insights if provided
    if args.insights and Path(args.insights).exists():
        with open(args.insights, 'r') as f:
            insights = json.load(f)

        boundary_ratios = [v['boundary_importance_ratio'] for v in insights.values()]
        boundary_importance = np.mean(boundary_ratios)
        print(f"Boundary importance (from GAT): {boundary_importance:.3f}")
    else:
        boundary_importance = args.boundary_importance
        print(f"Boundary importance (manual): {boundary_importance:.3f}")

    print("="*70)

    # Load datasets
    data_root = Path("datasets/flip_simulations")
    rigid_body_types = ['falling_ball', 'paddle_wave', 'rotating_stirrer',
                        'moving_obstacle', 'double_paddle', 'sphere_bounce']

    rigid_scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("sph_")
        and any(rb_type in d.name for rb_type in rigid_body_types)
    ])

    print(f"\nFound {len(rigid_scenario_dirs)} rigid body scenario directories")

    train_datasets = []
    val_datasets = []

    for i, scenario_dir in enumerate(rigid_scenario_dirs):
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

                print(f"  [{i+1}/{len(rigid_scenario_dirs)}] {scenario_dir.name}: "
                      f"{len(dataset)} sequences (train: {train_size}, val: {val_size})")
        except Exception as e:
            print(f"  [SKIP] {scenario_dir}: {e}")

    if len(train_datasets) == 0:
        print("\nNo datasets loaded!")
        return

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"\nTotal training sequences: {len(train_dataset)}")
    print(f"Total validation sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create hybrid model
    model = HybridPhysicsGNN(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        num_layers=args.num_layers,
        use_learned_attention=not args.no_learned_attention,
        use_type_weighting=not args.no_type_weighting,
        boundary_importance=boundary_importance
    ).to(device)

    print(f"\nHybrid Model Parameters: {model.count_parameters():,}")
    print(f"  Use learned attention: {model.use_learned_attention}")
    print(f"  Use type weighting: {model.use_type_weighting}")
    print(f"  Boundary importance: {model.boundary_importance:.3f}")

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
    save_path = Path("models") / args.save_name
    save_path.parent.mkdir(exist_ok=True)

    train_losses = []
    val_losses = []
    train_components_hist = {'vel_loss': [], 'pos_loss': [], 'momentum_loss': []}
    val_components_hist = {'vel_loss': [], 'pos_loss': [], 'momentum_loss': []}

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    start_time = time.time()

    for epoch in range(args.epochs):
        t_ep = time.time()
        train_loss, train_components = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_components = validate(model, val_loader, criterion, device)
        ep_elapsed = time.time() - t_ep

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for k in train_components_hist:
            train_components_hist[k].append(train_components[k])
            val_components_hist[k].append(val_components[k])

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{args.epochs}  ({ep_elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f} "
              f"(vel: {train_components['vel_loss']:.6f}, "
              f"pos: {train_components['pos_loss']:.9f}, "
              f"mom: {train_components['momentum_loss']:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} "
              f"(vel: {val_components['vel_loss']:.6f}, "
              f"pos: {val_components['pos_loss']:.9f}, "
              f"mom: {val_components['momentum_loss']:.6f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_components_hist': train_components_hist,
                'val_components_hist': val_components_hist,
                'config': {
                    'model': {
                        'input_node_dim': 7,
                        'input_edge_dim': 3,
                        'hidden_dim': args.hidden_dim,
                        'output_dim': 3,
                        'num_layers': args.num_layers,
                        'use_learned_attention': not args.no_learned_attention,
                        'use_type_weighting': not args.no_type_weighting,
                        'boundary_importance': boundary_importance
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
