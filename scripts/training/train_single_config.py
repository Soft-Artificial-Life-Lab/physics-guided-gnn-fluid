"""
Train Single Configuration - Quick Test
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_gnn import HybridPhysicsGNN


class RigidBodySPHDataset(Dataset):
    def __init__(self, data_dir, radius=0.3):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))
        if len(self.frame_files) < 2:
            raise ValueError(f"Need at least 2 frames")

    def __len__(self):
        return len(self.frame_files) - 1

    def __getitem__(self, idx):
        data_t = np.load(self.frame_files[idx])
        data_t1 = np.load(self.frame_files[idx + 1])
        pos_t = data_t['positions']
        vel_t = data_t['velocities']
        particle_types = data_t['particle_types']
        pos_t1 = data_t1['positions']
        vel_t1 = data_t1['velocities']
        edge_index, edge_attr = self.build_graph(pos_t)
        node_features = np.concatenate([pos_t, vel_t, particle_types[:, np.newaxis]], axis=1)

        return {
            'node_features': torch.FloatTensor(node_features),
            'edge_index': torch.LongTensor(edge_index),
            'edge_attr': torch.FloatTensor(edge_attr),
            'pos_t': torch.FloatTensor(pos_t),
            'pos_t1': torch.FloatTensor(pos_t1),
            'vel_t1': torch.FloatTensor(vel_t1),
            'particle_types': torch.FloatTensor(particle_types),
        }

    def build_graph(self, positions):
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


def compute_momentum_conservation(pred_vel, true_vel, particle_types):
    is_fluid = particle_types < 0.5
    if torch.sum(is_fluid).item() == 0:
        return 100.0
    pred_vel_fluid = pred_vel[is_fluid]
    true_vel_fluid = true_vel[is_fluid]
    pred_momentum = torch.sum(pred_vel_fluid, dim=0)
    true_momentum = torch.sum(true_vel_fluid, dim=0)
    momentum_error = torch.norm(pred_momentum - true_momentum) / (torch.norm(true_momentum) + 1e-8)
    return max(0.0, min(100.0, (1.0 - momentum_error.item()) * 100))


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mom = 0
    n = 0
    for batch in dataloader:
        node_features = batch['node_features'].squeeze(0).to(device)
        edge_index = batch['edge_index'].squeeze(0).to(device)
        edge_attr = batch['edge_attr'].squeeze(0).to(device)
        pos_t = batch['pos_t'].squeeze(0).to(device)
        pos_t1 = batch['pos_t1'].squeeze(0).to(device)
        vel_t1 = batch['vel_t1'].squeeze(0).to(device)
        particle_types = batch['particle_types'].squeeze(0).to(device)

        pred_vel = model(node_features, edge_index, edge_attr)
        is_fluid = particle_types < 0.5
        pred_vel_fluid = pred_vel[is_fluid]
        true_vel_fluid = vel_t1[is_fluid]
        loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mom_cons = compute_momentum_conservation(pred_vel, vel_t1, particle_types)
        total_loss += loss.item()
        total_mom += mom_cons
        n += 1

    return total_loss / n, total_mom / n


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_mom = 0
    n = 0
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
            is_fluid = particle_types < 0.5
            pred_vel_fluid = pred_vel[is_fluid]
            true_vel_fluid = vel_t1[is_fluid]
            loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)

            mom_cons = compute_momentum_conservation(pred_vel, vel_t1, particle_types)
            total_loss += loss.item()
            total_mom += mom_cons
            n += 1

    return total_loss / n, total_mom / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', action='store_true', help='Use attention gate')
    parser.add_argument('--type-weight', action='store_true', help='Use type weighting')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: attention={args.attention}, type_weight={args.type_weight}")

    # Load data
    data_root = Path("datasets/flip_simulations")
    scenario_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and 'dynamic' in d.name])[:5]  # Use first 5 scenarios

    print(f"Loading {len(scenario_dirs)} scenarios...")
    train_datasets = []
    val_datasets = []

    for scenario_dir in scenario_dirs:
        try:
            dataset = RigidBodySPHDataset(scenario_dir, radius=0.3)
            if len(dataset) > 0:
                split_idx = int(0.8 * len(dataset))
                train_subset, val_subset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])
                train_datasets.append(train_subset)
                val_datasets.append(val_subset)
                print(f"  {scenario_dir.name}: {len(dataset)} frames")
        except Exception as e:
            print(f"  Skip {scenario_dir.name}: {e}")

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    print(f"Total: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create model
    model = HybridPhysicsGNN(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5,
        use_learned_attention=args.attention,
        use_type_weighting=args.type_weight,
        boundary_importance=2.0 if args.type_weight else 1.0
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        train_loss, train_mom = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mom = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, Momentum={val_mom:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"\nFinal: Val Loss={best_val_loss:.4f}, Momentum={val_mom:.1f}%")


if __name__ == "__main__":
    main()
