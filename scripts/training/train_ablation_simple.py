"""Simple Ablation Study - One config at a time"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_gnn import HybridPhysicsGNN


class SimpleDataset(Dataset):
    def __init__(self, data_dir, radius=0.3):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))[:50]  # Limit frames

    def __len__(self):
        return max(0, len(self.frame_files) - 1)

    def __getitem__(self, idx):
        data_t = np.load(self.frame_files[idx])
        data_t1 = np.load(self.frame_files[idx + 1])
        pos_t, vel_t = data_t['positions'], data_t['velocities']
        particle_types = data_t['particle_types']
        pos_t1, vel_t1 = data_t1['positions'], data_t1['velocities']

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
        edges, edge_features = [], []
        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, self.radius)
            for j in neighbors:
                if i != j:
                    edges.append([i, j])
                    edge_features.append(positions[j] - positions[i])
        if not edges:
            return np.array([[0], [0]]), np.array([[0, 0, 0]])
        return np.array(edges).T, np.array(edge_features)


def compute_momentum(pred_vel, true_vel, particle_types):
    is_fluid = particle_types < 0.5
    if torch.sum(is_fluid).item() == 0:
        return 100.0
    pred_mom = torch.sum(pred_vel[is_fluid], dim=0)
    true_mom = torch.sum(true_vel[is_fluid], dim=0)
    error = torch.norm(pred_mom - true_mom) / (torch.norm(true_mom) + 1e-8)
    return max(0.0, min(100.0, (1.0 - error.item()) * 100))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_mom, n = 0, 0, 0
    for batch in loader:
        nf = batch['node_features'].squeeze(0).to(device)
        ei = batch['edge_index'].squeeze(0).to(device)
        ea = batch['edge_attr'].squeeze(0).to(device)
        pt = batch['particle_types'].squeeze(0).to(device)
        vel_t1 = batch['vel_t1'].squeeze(0).to(device)

        pred_vel = model(nf, ei, ea)
        is_fluid = pt < 0.5
        loss = torch.mean((pred_vel[is_fluid] - vel_t1[is_fluid]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mom += compute_momentum(pred_vel, vel_t1, pt)
        n += 1
    return total_loss / n, total_mom / n


def validate(model, loader, device):
    model.eval()
    total_loss, total_mom, n = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            nf = batch['node_features'].squeeze(0).to(device)
            ei = batch['edge_index'].squeeze(0).to(device)
            ea = batch['edge_attr'].squeeze(0).to(device)
            pt = batch['particle_types'].squeeze(0).to(device)
            vel_t1 = batch['vel_t1'].squeeze(0).to(device)

            pred_vel = model(nf, ei, ea)
            is_fluid = pt < 0.5
            loss = torch.mean((pred_vel[is_fluid] - vel_t1[is_fluid]) ** 2)

            total_loss += loss.item()
            total_mom += compute_momentum(pred_vel, vel_t1, pt)
            n += 1
    return total_loss / n, total_mom / n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    configs = {
        1: {'name': 'Base MLP only', 'att': False, 'tw': False, 'bi': 1.0},
        2: {'name': '+ Attention Gate', 'att': True, 'tw': False, 'bi': 1.0},
        3: {'name': '+ Type Weighting', 'att': True, 'tw': True, 'bi': 2.0},
        4: {'name': '+ Residual Connections', 'att': True, 'tw': True, 'bi': 2.0},
        5: {'name': '+ Physics Loss (Full)', 'att': True, 'tw': True, 'bi': 2.0},
    }

    cfg = configs[args.config]
    print(f"\n{'='*70}")
    print(f"CONFIG {args.config}: {cfg['name']}")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (first 3 scenarios only for speed)
    data_root = Path("datasets/flip_simulations")
    dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and 'dynamic' in d.name])[:3]

    print(f"Loading {len(dirs)} scenarios...")
    all_data = []
    for d in dirs:
        try:
            dataset = SimpleDataset(d, radius=0.3)
            if len(dataset) > 0:
                all_data.extend([dataset[i] for i in range(len(dataset))])
                print(f"  {d.name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  Skip {d.name}: {e}")

    if not all_data:
        print("No data loaded!")
        return

    # Split
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    print(f"Total: {len(train_data)} train, {len(val_data)} val\n")

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Model
    model = HybridPhysicsGNN(
        input_node_dim=7, input_edge_dim=3, hidden_dim=64, output_dim=3, num_layers=5,
        use_learned_attention=cfg['att'], use_type_weighting=cfg['tw'], boundary_importance=cfg['bi']
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    print("Training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_mom = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mom = validate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d}/{args.epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, Mom={val_mom:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - Config {args.config}: {cfg['name']}")
    print(f"{'='*70}")
    print(f"Val Loss: {best_val_loss:.4f}")
    print(f"Momentum Conservation: {val_mom:.1f}%")
    print(f"Training Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}\n")

    # Save model
    save_dir = Path("models/ablation")
    save_dir.mkdir(exist_ok=True)
    torch.save({
        'config': args.config,
        'config_name': cfg['name'],
        'model_state_dict': model.state_dict(),
        'val_loss': best_val_loss,
        'momentum': val_mom,
    }, save_dir / f"config_{args.config}.pt")
    print(f"Model saved to: {save_dir / f'config_{args.config}.pt'}")


if __name__ == "__main__":
    main()
