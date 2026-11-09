"""
3D Multi-view Visualization: Top view + Side view
Generate GIF animations with multiple camera angles
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import torch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gnn import MeshGraphNet
from src.models.hybrid_gnn import HybridPhysicsGNN


def load_models(gnn_path, hybrid_path):
    """Load both GNN and Hybrid models"""
    checkpoint = torch.load(gnn_path, map_location='cpu')
    gnn_model = MeshGraphNet(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5
    )
    gnn_model.load_state_dict(checkpoint['model_state_dict'])
    gnn_model.eval()

    checkpoint = torch.load(hybrid_path, map_location='cpu')
    hybrid_model = HybridPhysicsGNN(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5,
        use_learned_attention=True,
        use_type_weighting=True,
        boundary_importance=2.0
    )
    hybrid_model.load_state_dict(checkpoint['model_state_dict'])
    hybrid_model.eval()

    return gnn_model, hybrid_model


def rollout_prediction(model, initial_frame, num_steps=80, dt=0.01):
    """Rollout model predictions"""
    from scipy.spatial import cKDTree

    positions = initial_frame['positions'].copy()
    velocities = initial_frame['velocities'].copy()
    particle_types = initial_frame['particle_types']
    n_fluid = initial_frame['n_fluid']

    trajectory = [positions.copy()]

    for step in range(num_steps):
        tree = cKDTree(positions)
        edges = []
        edge_features = []

        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, 0.3)
            for j in neighbors:
                if i != j:
                    edges.append([i, j])
                    edge_features.append(positions[j] - positions[i])

        if len(edges) == 0:
            break

        edge_index = torch.LongTensor(np.array(edges).T)
        edge_attr = torch.FloatTensor(np.array(edge_features))
        node_features = torch.FloatTensor(np.concatenate([
            positions, velocities, particle_types[:, np.newaxis]
        ], axis=1))

        with torch.no_grad():
            pred_vel = model(node_features, edge_index, edge_attr)

        velocities[:n_fluid] = pred_vel[:n_fluid].numpy()
        positions[:n_fluid] += velocities[:n_fluid] * dt

        # Clear GPU cache to free memory
        del node_features, edge_index, edge_attr, pred_vel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for dim in range(3):
            mask = positions[:n_fluid, dim] < 0
            positions[:n_fluid][mask, dim] = 0
            velocities[:n_fluid][mask, dim] *= -0.5

            mask = positions[:n_fluid, dim] > 1.0
            positions[:n_fluid][mask, dim] = 1.0
            velocities[:n_fluid][mask, dim] *= -0.5

        trajectory.append(positions.copy())

    return trajectory


def create_multiview_gif(scenario_path, gnn_model, hybrid_model, output_path, num_frames=80):
    """Create GIF with 3D multi-view: Top + Side views for GT, GNN, Hybrid"""

    # Load ground truth
    frame_files = sorted(scenario_path.glob('frame_*.npz'))
    print(f"    Loading {len(frame_files)} GT frames...")

    gt_trajectory = []
    gt_rigid_positions = []

    for i in range(min(num_frames, len(frame_files))):
        data = np.load(frame_files[i])
        gt_trajectory.append(data['positions'])
        n_fluid = data['n_fluid']
        gt_rigid_positions.append(data['positions'][n_fluid:])

    initial_frame = np.load(frame_files[0])
    n_fluid = initial_frame['n_fluid']

    # Get predictions
    print(f"    Rolling out GNN predictions...")
    gnn_trajectory = rollout_prediction(gnn_model, initial_frame, num_frames)

    print(f"    Rolling out Hybrid predictions...")
    hybrid_trajectory = rollout_prediction(hybrid_model, initial_frame, num_frames)

    # Create 3x2 grid: (GT, GNN, Hybrid) x (Top view, Side view)
    fig = plt.figure(figsize=(18, 12))

    # Create 6 subplots: 3 columns (GT, GNN, Hybrid) x 2 rows (Top, Side)
    axes = []
    for row in range(2):  # Top and Side
        for col in range(3):  # GT, GNN, Hybrid
            ax = fig.add_subplot(2, 3, row*3 + col + 1)
            axes.append(ax)

    def update(frame):
        for ax in axes:
            ax.clear()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Top view (XY plane) - Row 0
        for col, (traj, rigid_pos, title) in enumerate([
            (gt_trajectory, gt_rigid_positions, 'Ground Truth'),
            (gnn_trajectory, gt_rigid_positions, 'GNN (Loss: 0.002095)'),
            (hybrid_trajectory, gt_rigid_positions, 'Hybrid (Loss: 0.002946)')
        ]):
            if frame < len(traj):
                ax = axes[col]  # Top row
                pos = traj[frame]
                rigid = rigid_pos[frame] if frame < len(rigid_pos) else pos[n_fluid:]

                # Fluid particles (XY)
                ax.scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=20, alpha=0.6, edgecolors='none')
                # Rigid body (XY)
                ax.scatter(rigid[:, 0], rigid[:, 1],
                          c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=1)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel('X', fontsize=9)
                ax.set_ylabel('Y', fontsize=9)
                ax.set_title(f'{title}\nTop View (XY)', fontsize=10, fontweight='bold')

        # Side view (XZ plane) - Row 1
        for col, (traj, rigid_pos, title) in enumerate([
            (gt_trajectory, gt_rigid_positions, 'Ground Truth'),
            (gnn_trajectory, gt_rigid_positions, 'GNN'),
            (hybrid_trajectory, gt_rigid_positions, 'Hybrid')
        ]):
            if frame < len(traj):
                ax = axes[col + 3]  # Bottom row
                pos = traj[frame]
                rigid = rigid_pos[frame] if frame < len(rigid_pos) else pos[n_fluid:]

                # Fluid particles (XZ)
                ax.scatter(pos[:n_fluid, 0], pos[:n_fluid, 2],
                          c='blue', s=20, alpha=0.6, edgecolors='none')
                # Rigid body (XZ)
                ax.scatter(rigid[:, 0], rigid[:, 2],
                          c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=1)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel('X', fontsize=9)
                ax.set_ylabel('Z', fontsize=9)
                ax.set_title(f'{title}\nSide View (XZ)', fontsize=10, fontweight='bold')

        scenario_title = scenario_path.name.replace('sph_dynamic_', '').replace('_', ' ').title()
        fig.suptitle(f'{scenario_title} - Frame {frame}/{num_frames}\n3D Multi-view: Top (XY) + Side (XZ)',
                    fontsize=14, fontweight='bold', y=0.98)

    print(f"    Creating 3D multi-view animation ({num_frames} frames)...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50)

    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"    Saved: {output_path.name}")


def main():
    """Generate 3D multi-view GIF visualizations"""

    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "datasets" / "flip_simulations"
    model_dir = project_root / "scripts" / "training" / "models"
    gnn_path = model_dir / "gnn_dynamic_two_way_coupling_best.pt"
    hybrid_path = model_dir / "hybrid_dynamic_two_way_coupling_best.pt"
    output_dir = project_root / "visualizations" / "dynamic_two_way_coupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("3D MULTI-VIEW VISUALIZATION")
    print("Top View (XY) + Side View (XZ)")
    print("GT vs GNN vs Hybrid - 80 Frames")
    print("="*70)

    print("\nLoading models...")
    gnn_model, hybrid_model = load_models(gnn_path, hybrid_path)
    print("  GNN loaded (Val Loss: 0.002095)")
    print("  Hybrid loaded (Val Loss: 0.002946)")

    scenarios_to_viz = [
        'flip_dynamic_falling_ball_000',
        'flip_dynamic_floating_ball_000',
        'flip_dynamic_heavy_box_000',
        'flip_dynamic_multi_ball_000',
        'flip_dynamic_bouncing_sphere_000'
    ]

    print(f"\nGenerating 3D multi-view GIFs for {len(scenarios_to_viz)} scenarios...")

    for i, scenario_name in enumerate(scenarios_to_viz, 1):
        scenario_path = data_root / scenario_name
        if not scenario_path.exists():
            print(f"\n[{i}/{len(scenarios_to_viz)}] SKIP: {scenario_name}")
            continue

        output_path = output_dir / f"{scenario_name}_3d_multiview.gif"
        print(f"\n[{i}/{len(scenarios_to_viz)}] {scenario_name}:")

        try:
            create_multiview_gif(scenario_path, gnn_model, hybrid_model,
                               output_path, num_frames=80)
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("3D MULTI-VIEW VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Output: {output_dir}")
    print("  - 5 GIFs with 3D multi-view (Top + Side)")
    print("  - GT vs GNN vs Hybrid comparison")
    print("  - 80 frames each (20fps)")
    print("="*70)


if __name__ == "__main__":
    main()
