"""
Visualize GNN predictions on Dynamic Two-way Coupling data
Generate GIF animations comparing Ground Truth vs GNN Predictions
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


def load_gnn_model(model_path):
    """Load trained GNN model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    model = MeshGraphNet(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def rollout_gnn_prediction(model, initial_frame, num_steps=30, dt=0.01):
    """Rollout GNN predictions for multiple timesteps"""
    from scipy.spatial import cKDTree

    positions = initial_frame['positions'].copy()
    velocities = initial_frame['velocities'].copy()
    particle_types = initial_frame['particle_types']
    n_fluid = initial_frame['n_fluid']

    trajectory = [positions.copy()]

    for step in range(num_steps):
        # Build graph
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

        # Node features
        node_features = torch.FloatTensor(np.concatenate([
            positions,
            velocities,
            particle_types[:, np.newaxis]
        ], axis=1))

        # Predict velocities
        with torch.no_grad():
            pred_vel = model(node_features, edge_index, edge_attr)

        # Update positions (only fluid particles)
        velocities[:n_fluid] = pred_vel[:n_fluid].numpy()
        positions[:n_fluid] += velocities[:n_fluid] * dt

        # Boundary conditions
        for dim in range(3):
            mask = positions[:n_fluid, dim] < 0
            positions[:n_fluid][mask, dim] = 0
            velocities[:n_fluid][mask, dim] *= -0.5

            mask = positions[:n_fluid, dim] > 1.0
            positions[:n_fluid][mask, dim] = 1.0
            velocities[:n_fluid][mask, dim] *= -0.5

        trajectory.append(positions.copy())

    return trajectory


def create_comparison_gif(scenario_path, model, output_path, num_frames=30):
    """Create GIF comparing GT vs GNN predictions"""

    # Load ground truth
    frame_files = sorted(scenario_path.glob('frame_*.npz'))
    initial_frame = np.load(frame_files[0])

    # Get ground truth trajectory
    gt_trajectory = []
    for i in range(min(num_frames, len(frame_files))):
        data = np.load(frame_files[i])
        gt_trajectory.append(data['positions'])

    # Get GNN predictions
    print(f"  Rolling out GNN predictions...")
    gnn_trajectory = rollout_gnn_prediction(model, initial_frame, num_frames)

    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_fluid = initial_frame['n_fluid']

    def update(frame):
        for ax in axes:
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

        # Ground Truth
        if frame < len(gt_trajectory):
            pos = gt_trajectory[frame]
            axes[0].scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=10, alpha=0.6, label='Fluid')
            axes[0].scatter(pos[n_fluid:, 0], pos[n_fluid:, 1],
                          c='red', s=20, alpha=0.8, label='Rigid Body')
            axes[0].set_title(f'Ground Truth (Two-way Coupling)\nFrame {frame}', fontsize=10)
            axes[0].legend(loc='upper right', fontsize=8)

        # GNN Prediction
        if frame < len(gnn_trajectory):
            pos = gnn_trajectory[frame]
            axes[1].scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=10, alpha=0.6, label='Fluid')
            axes[1].scatter(pos[n_fluid:, 0], pos[n_fluid:, 1],
                          c='red', s=20, alpha=0.8, label='Rigid Body')
            axes[1].set_title(f'GNN Prediction (Val Loss: 0.002095)\nFrame {frame}', fontsize=10)
            axes[1].legend(loc='upper right', fontsize=8)

        for ax in axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        fig.suptitle(scenario_path.name.replace('sph_dynamic_', '').replace('_', ' ').title(),
                    fontsize=12, fontweight='bold')

    print(f"  Creating animation...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    """Generate GIF visualizations for dynamic scenarios"""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "datasets" / "flip_simulations"
    model_path = project_root / "scripts" / "training" / "models" / "gnn_dynamic_two_way_coupling_best.pt"
    output_dir = project_root / "visualizations" / "dynamic_two_way_coupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("="*70)
    print("GNN VISUALIZATION FOR DYNAMIC TWO-WAY COUPLING DATA")
    print("="*70)
    print("Loading GNN model...")
    model = load_gnn_model(model_path)
    print("Model loaded! (Best Val Loss: 0.002095)")

    # Select scenarios to visualize
    scenarios_to_viz = [
        'sph_dynamic_falling_ball_000',
        'sph_dynamic_floating_ball_000',
        'sph_dynamic_heavy_box_000',
        'sph_dynamic_multi_ball_000',
        'sph_dynamic_bouncing_sphere_000'
    ]

    print(f"\nGenerating GIFs for {len(scenarios_to_viz)} scenarios...")

    for scenario_name in scenarios_to_viz:
        scenario_path = data_root / scenario_name
        if not scenario_path.exists():
            print(f"  [SKIP] {scenario_name} not found")
            continue

        output_path = output_dir / f"{scenario_name}_gnn.gif"
        print(f"\n{scenario_name}:")

        try:
            create_comparison_gif(scenario_path, model, output_path, num_frames=30)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"GIFs saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
