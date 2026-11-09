"""
Re-validate hybrid_rigid_body_sph_best.pt on newly regenerated sph_* data.

Uses the EXACT loss function and validate() loop from train_hybrid.py.
Only the graph-build step is vectorised (cKDTree.query_pairs instead of the
Python per-particle loop) -- the produced graph (undirected k-NN edges,
edge_attr = dest - src) is mathematically identical.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from src.models.hybrid_gnn import HybridPhysicsGNN
from train_hybrid import RigidBodySPHDataset, RigidBodyPhysicsLoss, validate


class FastRigidBodySPHDataset(RigidBodySPHDataset):
    """Same dataset but with vectorised graph build."""

    def build_graph(self, positions):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(self.radius, output_type='ndarray')
        if len(pairs) == 0:
            return np.array([[0], [0]]), np.array([[0.0, 0.0, 0.0]])
        edges_ij = pairs
        edges_ji = pairs[:, ::-1]
        edges = np.concatenate([edges_ij, edges_ji], axis=0).T
        src, dst = edges[0], edges[1]
        edge_features = positions[dst] - positions[src]
        return edges, edge_features


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = ROOT / "models" / "hybrid_rigid_body_sph_best.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']['model']
    radius = ckpt['config']['training']['radius']
    dt = ckpt['config']['training']['dt']

    print("="*70)
    print("STATIC HYBRID RE-VALIDATION  (regenerated sph_* data)")
    print("="*70)
    print(f"Checkpoint      : {ckpt_path.name}")
    print(f"Recorded best   : epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}")
    print(f"Device          : {device}")
    print("="*70)

    model = HybridPhysicsGNN(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_root = ROOT / "datasets" / "flip_simulations"
    filters = ['sph_falling_ball', 'sph_paddle_wave', 'sph_rotating_stirrer',
               'sph_moving_obstacle', 'sph_double_paddle', 'sph_sphere_bounce']

    scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and any(f in d.name for f in filters)
    ])
    print(f"\nMatched {len(scenario_dirs)} scenarios")

    val_datasets = []
    torch.manual_seed(42)
    for d in scenario_dirs:
        ds = FastRigidBodySPHDataset(d, radius=radius)
        split_idx = int(0.8 * len(ds))
        _, val_sub = torch.utils.data.random_split(
            ds, [split_idx, len(ds) - split_idx]
        )
        val_datasets.append(val_sub)

    val_loader = DataLoader(ConcatDataset(val_datasets),
                            batch_size=1, shuffle=False)
    total = sum(len(v) for v in val_datasets)
    print(f"Total val sequences: {total}")

    criterion = RigidBodyPhysicsLoss(
        vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=dt
    )

    t0 = time.time()
    val_loss, comp = validate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Re-measured val_loss : {val_loss:.6f}")
    print(f"  vel_loss           : {comp['vel_loss']:.6f}")
    print(f"  pos_loss           : {comp['pos_loss']:.9f}")
    print(f"  momentum_loss      : {comp['momentum_loss']:.6f}")
    print(f"Inference            : {elapsed:.1f}s "
          f"({elapsed*1000/total:.1f} ms/sample)")
    print("-"*70)
    print(f"Recorded best        : {ckpt['val_loss']:.6f}")
    print(f"Delta                : {(val_loss - ckpt['val_loss']):+.6f}  "
          f"({(val_loss/ckpt['val_loss']):.2f}x)")
    print("="*70)


if __name__ == "__main__":
    main()
