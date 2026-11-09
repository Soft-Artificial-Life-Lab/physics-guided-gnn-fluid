"""
Visualize GNN/Hybrid predictions on Dynamic Two-way Coupling data
Generate GIF animations comparing Ground Truth vs Model Predictions
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


def load_model(model_path, model_type='gnn'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    if model_type == 'gnn':
        model = MeshGraphNet(
            input_node_dim=7,
            input_edge_dim=3,
            hidden_dim=64,
            output_dim=3,
            num_layers=5
        )
    elif model_type == 'hybrid':
        model = HybridPhysicsGNN(
            input_node_dim=7,
            input_edge_dim=3,
            hidden_dim=64,
            output_dim=3,
            num_layers=5,
            use_learned_attention=True,
            use_type_weighting=True,
            boundary_importance=2.0
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def rollout_prediction(model, initial_frame, num_steps=30, dt=0.01):
    """Rollout model predictions for multiple timesteps"""
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
            if isinstance(model, HybridPhysicsGNN):
                pred_vel = model(node_features, edge_index, edge_attr,
                                torch.FloatTensor(particle_types))
            else:
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


def create_comparison_gif(scenario_path, model_gnn, model_hybrid, output_path,
                          num_frames=30):
    """Create GIF comparing GT, GNN, and Hybrid predictions"""

    # Load ground truth
    frame_files = sorted(scenario_path.glob('frame_*.npz'))
    initial_frame = np.load(frame_files[0])

    # Get ground truth trajectory
    gt_trajectory = []
    for i in range(min(num_frames, len(frame_files))):
        data = np.load(frame_files[i])
        gt_trajectory.append(data['positions'])

    # Get model predictions
    print(f"  Rolling out GNN predictions...")
    gnn_trajectory = rollout_prediction(model_gnn, initial_frame, num_frames)

    print(f"  Rolling out Hybrid predictions...")
    hybrid_trajectory = rollout_prediction(model_hybrid, initial_frame, num_frames)

    # Create animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
                          c='red', s=20, alpha=0.8, label='Boundary')
            axes[0].set_title(f'Ground Truth (Frame {frame})')
            axes[0].legend(loc='upper right', fontsize=8)

        # GNN Prediction
        if frame < len(gnn_trajectory):
            pos = gnn_trajectory[frame]
            axes[1].scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=10, alpha=0.6)
            axes[1].scatter(pos[n_fluid:, 0], pos[n_fluid:, 1],
                          c='red', s=20, alpha=0.8)
            axes[1].set_title(f'GNN Prediction (Frame {frame})')

        # Hybrid Prediction
        if frame < len(hybrid_trajectory):
            pos = hybrid_trajectory[frame]
            axes[2].scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=10, alpha=0.6)
            axes[2].scatter(pos[n_fluid:, 0], pos[n_fluid:, 1],
                          c='red', s=20, alpha=0.8)
            axes[2].set_title(f'Hybrid Prediction (Frame {frame})')

        for ax in axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

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
    model_dir = project_root / "scripts" / "training" / "models"
    output_dir = project_root / "visualizations" / "dynamic_two_way_coupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading trained models...")
    model_gnn = load_model(model_dir / "gnn_dynamic_two_way_coupling_best.pt", 'gnn')
    model_hybrid = load_model(model_dir / "hybrid_dynamic_two_way_coupling_best.pt", 'hybrid')
    print("Models loaded!")

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

        output_path = output_dir / f"{scenario_name}_comparison.gif"
        print(f"\n{scenario_name}:")

        try:
            create_comparison_gif(
                scenario_path,
                model_gnn,
                model_hybrid,
                output_path,
                num_frames=30
            )
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"GIFs saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
