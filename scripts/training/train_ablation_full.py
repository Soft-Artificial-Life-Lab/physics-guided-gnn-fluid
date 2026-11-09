"""Full Ablation Study - All datasets, paper reproduction"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import sys
import time
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_gnn import HybridPhysicsGNN


class RigidBodySPHDataset(Dataset):
    """Full dataset loader"""
    def __init__(self, data_dir, radius=0.3):
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.frame_files = sorted(self.data_dir.glob("frame_*.npz"))

        if len(self.frame_files) < 2:
            raise ValueError(f"Need at least 2 frames")

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


class PhysicsLoss(nn.Module):
    """Full physics-informed loss"""
    def __init__(self, vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=0.01):
        super().__init__()
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.momentum_weight = momentum_weight
        self.dt = dt

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types):
        is_fluid = particle_types < 0.5

        # Velocity loss
        pred_vel_fluid = pred_vel[is_fluid]
        true_vel_fluid = true_vel[is_fluid]
        vel_loss = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2)

        # Position consistency loss
        pos_t_fluid = pos_t[is_fluid]
        pos_t1_fluid = pos_t1[is_fluid]
        predicted_pos_fluid = pos_t_fluid + pred_vel_fluid * self.dt
        pos_loss = torch.mean((predicted_pos_fluid - pos_t1_fluid) ** 2)

        # Momentum conservation loss
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

        return total_loss


def compute_momentum_conservation(pred_vel, true_vel, particle_types):
    """Compute momentum conservation percentage"""
    is_fluid = particle_types < 0.5
    if torch.sum(is_fluid).item() == 0:
        return 100.0

    pred_vel_fluid = pred_vel[is_fluid]
    true_vel_fluid = true_vel[is_fluid]

    pred_momentum = torch.sum(pred_vel_fluid, dim=0)
    true_momentum = torch.sum(true_vel_fluid, dim=0)

    momentum_error = torch.norm(pred_momentum - true_momentum) / (torch.norm(true_momentum) + 1e-8)
    momentum_conservation = (1.0 - momentum_error.item()) * 100

    return max(0.0, min(100.0, momentum_conservation))


def train_epoch(model, loader, optimizer, criterion, device, use_physics_loss):
    model.train()
    total_loss, total_mom, n = 0, 0, 0

    for batch in loader:
        nf = batch['node_features'].squeeze(0).to(device)
        ei = batch['edge_index'].squeeze(0).to(device)
        ea = batch['edge_attr'].squeeze(0).to(device)
        pt = batch['particle_types'].squeeze(0).to(device)
        pos_t = batch['pos_t'].squeeze(0).to(device)
        pos_t1 = batch['pos_t1'].squeeze(0).to(device)
        vel_t1 = batch['vel_t1'].squeeze(0).to(device)

        pred_vel = model(nf, ei, ea)

        if use_physics_loss:
            loss = criterion(pred_vel, vel_t1, pos_t, pos_t1, pt)
        else:
            is_fluid = pt < 0.5
            loss = torch.mean((pred_vel[is_fluid] - vel_t1[is_fluid]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mom += compute_momentum_conservation(pred_vel, vel_t1, pt)
        n += 1

    return total_loss / n, total_mom / n


def validate(model, loader, criterion, device, use_physics_loss):
    model.eval()
    total_loss, total_mom, n = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            nf = batch['node_features'].squeeze(0).to(device)
            ei = batch['edge_index'].squeeze(0).to(device)
            ea = batch['edge_attr'].squeeze(0).to(device)
            pt = batch['particle_types'].squeeze(0).to(device)
            pos_t = batch['pos_t'].squeeze(0).to(device)
            pos_t1 = batch['pos_t1'].squeeze(0).to(device)
            vel_t1 = batch['vel_t1'].squeeze(0).to(device)

            pred_vel = model(nf, ei, ea)

            if use_physics_loss:
                loss = criterion(pred_vel, vel_t1, pos_t, pos_t1, pt)
            else:
                is_fluid = pt < 0.5
                loss = torch.mean((pred_vel[is_fluid] - vel_t1[is_fluid]) ** 2)

            total_loss += loss.item()
            total_mom += compute_momentum_conservation(pred_vel, vel_t1, pt)
            n += 1

    return total_loss / n, total_mom / n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, required=True, choices=[1,2,3,4,5])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    configs = {
        1: {'name': 'Base MLP only', 'att': False, 'tw': False, 'bi': 1.0, 'physics': False},
        2: {'name': '+ Attention Gate', 'att': True, 'tw': False, 'bi': 1.0, 'physics': False},
        3: {'name': '+ Type Weighting', 'att': True, 'tw': True, 'bi': 2.0, 'physics': False},
        4: {'name': '+ Residual Connections', 'att': True, 'tw': True, 'bi': 2.0, 'physics': False},
        5: {'name': '+ Physics Loss (Full)', 'att': True, 'tw': True, 'bi': 2.0, 'physics': True},
    }

    cfg = configs[args.config]
    print(f"\n{'='*70}")
    print(f"CONFIG {args.config}: {cfg['name']}")
    print(f"FULL DATASET - PAPER REPRODUCTION")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load ALL scenarios
    data_root = Path("datasets/flip_simulations")
    dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and 'dynamic' in d.name])

    print(f"\nLoading {len(dirs)} scenarios...")
    train_datasets = []
    val_datasets = []

    for i, d in enumerate(dirs):
        try:
            dataset = RigidBodySPHDataset(d, radius=0.3)
            if len(dataset) > 0:
                split_idx = int(0.8 * len(dataset))
                train_size = split_idx
                val_size = len(dataset) - split_idx

                train_subset, val_subset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

                train_datasets.append(train_subset)
                val_datasets.append(val_subset)

                print(f"  [{i+1:2d}/{len(dirs)}] {d.name}: {len(dataset)} frames "
                      f"(train: {train_size}, val: {val_size})")
        except Exception as e:
            print(f"  [SKIP] {d.name}: {e}")

    if not train_datasets:
        print("No data loaded!")
        return

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"\nTotal: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    model = HybridPhysicsGNN(
        input_node_dim=7, input_edge_dim=3, hidden_dim=64, output_dim=3, num_layers=5,
        use_learned_attention=cfg['att'],
        use_type_weighting=cfg['tw'],
        boundary_importance=cfg['bi']
    ).to(device)

    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"  Attention: {cfg['att']}")
    print(f"  Type Weighting: {cfg['tw']}")
    print(f"  Boundary Importance: {cfg['bi']}")
    print(f"  Physics Loss: {cfg['physics']}\n")

    # Loss and optimizer
    criterion = PhysicsLoss() if cfg['physics'] else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    results = {
        'config': args.config,
        'config_name': cfg['name'],
        'train_losses': [],
        'val_losses': [],
        'momentum_conservations': []
    }

    print("Training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_mom = train_epoch(model, train_loader, optimizer, criterion, device, cfg['physics'])
        val_loss, val_mom = validate(model, val_loader, criterion, device, cfg['physics'])
        scheduler.step(val_loss)

        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['momentum_conservations'].append(val_mom)

        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"Train={train_loss:.6f}, Val={val_loss:.6f}, Mom={val_mom:.1f}%, "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    elapsed = time.time() - start_time

    results['best_val_loss'] = best_val_loss
    results['final_momentum'] = results['momentum_conservations'][-1]
    results['training_time'] = elapsed

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - Config {args.config}: {cfg['name']}")
    print(f"{'='*70}")
    print(f"Val Loss: {best_val_loss:.4f}")
    print(f"Momentum Conservation: {results['final_momentum']:.1f}%")
    print(f"Training Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}\n")

    # Save model and results
    save_dir = Path("models/ablation_full")
    save_dir.mkdir(exist_ok=True)

    torch.save({
        'config': args.config,
        'config_name': cfg['name'],
        'model_state_dict': model.state_dict(),
        'results': results,
        'model_config': {
            'input_node_dim': 7,
            'input_edge_dim': 3,
            'hidden_dim': 64,
            'output_dim': 3,
            'num_layers': 5,
            'use_learned_attention': cfg['att'],
            'use_type_weighting': cfg['tw'],
            'boundary_importance': cfg['bi']
        }
    }, save_dir / f"config_{args.config}.pt")

    # Save results JSON
    with open(save_dir / f"config_{args.config}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to: {save_dir / f'config_{args.config}.pt'}")
    print(f"Results saved to: {save_dir / f'config_{args.config}_results.json'}")


if __name__ == "__main__":
    main()
