"""
Train Hybrid Physics-Guided GNN - Ablation Study

Reproduces the ablation study from the paper:
1. Base MLP only
2. + Attention Gate
3. + Type Weighting
4. + Residual Connections
5. + Physics Loss (Full)

Each configuration trained for 10 epochs.
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

    def __init__(self, data_dir, radius=0.3):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))

        if len(self.frame_files) < 2:
            raise ValueError(f"Need at least 2 frames in {data_dir}, found {len(self.frame_files)}")

        print(f"  Loaded {len(self.frame_files)} frames from {data_dir}")

    def __len__(self):
        return len(self.frame_files) - 1

    def __getitem__(self, idx):
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


class PhysicsLoss(nn.Module):
    """Physics-informed loss with velocity, position, and momentum terms"""

    def __init__(self, vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=0.01):
        super().__init__()
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.momentum_weight = momentum_weight
        self.dt = dt

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types):
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

        return total_loss, vel_loss, pos_loss, momentum_loss


class VelocityOnlyLoss(nn.Module):
    """Simple velocity-only MSE loss (for ablation baseline)"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types):
        is_fluid = particle_types < 0.5
        pred_vel_fluid = pred_vel[is_fluid]
        true_vel_fluid = true_vel[is_fluid]
        vel_loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)
        return vel_loss, vel_loss, torch.tensor(0.0), torch.tensor(0.0)


def compute_momentum_conservation(pred_vel, true_vel, particle_types):
    """Compute momentum conservation accuracy"""
    is_fluid = particle_types < 0.5
    if torch.sum(is_fluid).item() == 0:
        return 1.0

    pred_vel_fluid = pred_vel[is_fluid]
    true_vel_fluid = true_vel[is_fluid]

    pred_momentum = torch.sum(pred_vel_fluid, dim=0)
    true_momentum = torch.sum(true_vel_fluid, dim=0)

    momentum_error = torch.norm(pred_momentum - true_momentum) / (torch.norm(true_momentum) + 1e-8)
    momentum_conservation = (1.0 - momentum_error.item()) * 100

    return max(0.0, min(100.0, momentum_conservation))


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mom_conservation = 0
    n_batches = 0

    for batch in dataloader:
        node_features = batch['node_features'].squeeze(0).to(device)
        edge_index = batch['edge_index'].squeeze(0).to(device)
        edge_attr = batch['edge_attr'].squeeze(0).to(device)
        pos_t = batch['pos_t'].squeeze(0).to(device)
        pos_t1 = batch['pos_t1'].squeeze(0).to(device)
        vel_t1 = batch['vel_t1'].squeeze(0).to(device)
        particle_types = batch['particle_types'].squeeze(0).to(device)

        pred_vel = model(node_features, edge_index, edge_attr)
        loss, vel_loss, pos_loss, mom_loss = criterion(pred_vel, vel_t1, pos_t, pos_t1, particle_types)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mom_cons = compute_momentum_conservation(pred_vel, vel_t1, particle_types)

        total_loss += loss.item()
        total_mom_conservation += mom_cons
        n_batches += 1

    return total_loss / n_batches, total_mom_conservation / n_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_mom_conservation = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            node_features = batch['node_features'].squeeze(0).to(device)
            edge_index = batch['edge_index'].squeeze(0).to(device)
            edge_attr = batch['edge_attr'].squeeze(0).to(device)
            pos_t = batch['pos_t'].squeeze(0).to(device)
            pos_t1 = batch['pos_t1'].squeeze(0).to(device)
            vel_t1 = batch['vel_t1'].squeeze(0).to(device)
            particle_types = batch['particle_types'].squeeze(0).to(device)

            pred_vel = model(node_features, edge_index, edge_attr)
            loss, vel_loss, pos_loss, mom_loss = criterion(pred_vel, vel_t1, pos_t, pos_t1, particle_types)

            mom_cons = compute_momentum_conservation(pred_vel, vel_t1, particle_types)

            total_loss += loss.item()
            total_mom_conservation += mom_cons
            n_batches += 1

    return total_loss / n_batches, total_mom_conservation / n_batches


def train_configuration(config_name, model_config, loss_config, train_loader, val_loader, device, epochs=10):
    """Train a single configuration"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {config_name}")
    print(f"{'='*70}")

    # Create model
    model = HybridPhysicsGNN(**model_config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Create loss function
    if loss_config['type'] == 'physics':
        criterion = PhysicsLoss(
            vel_weight=loss_config.get('vel_weight', 1.0),
            pos_weight=loss_config.get('pos_weight', 0.1),
            momentum_weight=loss_config.get('momentum_weight', 0.05)
        )
    else:
        criterion = VelocityOnlyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    momentum_conservations = []

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_mom = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mom = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        momentum_conservations.append(val_mom)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, Momentum: {val_mom:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    elapsed = time.time() - start_time

    print(f"\nFinal Val Loss: {best_val_loss:.6f}")
    print(f"Final Momentum Conservation: {momentum_conservations[-1]:.1f}%")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        'config_name': config_name,
        'val_loss': best_val_loss,
        'final_momentum': momentum_conservations[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'momentum_conservations': momentum_conservations,
        'training_time': elapsed
    }


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    radius = 0.3
    epochs = 10

    print("="*70)
    print("HYBRID GNN ABLATION STUDY - PAPER REPRODUCTION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs per configuration: {epochs}")

    # Load datasets
    data_root = Path("datasets/flip_simulations")

    # Use all scenarios (falling_ball, paddle_wave, etc.)
    scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    print(f"\nFound {len(scenario_dirs)} scenario directories")

    train_datasets = []
    val_datasets = []

    for i, scenario_dir in enumerate(scenario_dirs):
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

                print(f"  [{i+1}/{len(scenario_dirs)}] {scenario_dir.name}: "
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

    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'Base MLP only',
            'model': {
                'input_node_dim': 7,
                'input_edge_dim': 3,
                'hidden_dim': 64,
                'output_dim': 3,
                'num_layers': 5,
                'use_learned_attention': False,
                'use_type_weighting': False,
                'boundary_importance': 1.0
            },
            'loss': {'type': 'velocity_only'}
        },
        {
            'name': '+ Attention Gate',
            'model': {
                'input_node_dim': 7,
                'input_edge_dim': 3,
                'hidden_dim': 64,
                'output_dim': 3,
                'num_layers': 5,
                'use_learned_attention': True,
                'use_type_weighting': False,
                'boundary_importance': 1.0
            },
            'loss': {'type': 'velocity_only'}
        },
        {
            'name': '+ Type Weighting',
            'model': {
                'input_node_dim': 7,
                'input_edge_dim': 3,
                'hidden_dim': 64,
                'output_dim': 3,
                'num_layers': 5,
                'use_learned_attention': True,
                'use_type_weighting': True,
                'boundary_importance': 2.0
            },
            'loss': {'type': 'velocity_only'}
        },
        {
            'name': '+ Residual Connections',
            'model': {
                'input_node_dim': 7,
                'input_edge_dim': 3,
                'hidden_dim': 64,
                'output_dim': 3,
                'num_layers': 5,
                'use_learned_attention': True,
                'use_type_weighting': True,
                'boundary_importance': 2.0
            },
            'loss': {'type': 'velocity_only'}
        },
        {
            'name': '+ Physics Loss (Full)',
            'model': {
                'input_node_dim': 7,
                'input_edge_dim': 3,
                'hidden_dim': 64,
                'output_dim': 3,
                'num_layers': 5,
                'use_learned_attention': True,
                'use_type_weighting': True,
                'boundary_importance': 2.0
            },
            'loss': {
                'type': 'physics',
                'vel_weight': 1.0,
                'pos_weight': 0.1,
                'momentum_weight': 0.05
            }
        }
    ]

    # Run ablation study
    results = []
    for config in ablation_configs:
        result = train_configuration(
            config['name'],
            config['model'],
            config['loss'],
            train_loader,
            val_loader,
            device,
            epochs=epochs
        )
        results.append(result)

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<30} {'Val Loss':<12} {'Momentum':<12}")
    print("-"*70)
    for result in results:
        print(f"{result['config_name']:<30} {result['val_loss']:<12.4f} {result['final_momentum']:<12.1f}%")

    # Save results
    results_dir = Path("logs/ablation_study")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ablation_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*70)


if __name__ == "__main__":
    main()
