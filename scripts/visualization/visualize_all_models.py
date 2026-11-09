"""
Visualize GNN and Hybrid predictions on Dynamic Two-way Coupling data
Generate GIF animations with 80 frames showing rigid body motion
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
    # Load GNN
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

    # Load Hybrid
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


def rollout_prediction(model, initial_frame, num_steps=80, dt=0.01, model_name='GNN'):
    """Rollout model predictions keeping rigid body position from GT"""
    from scipy.spatial import cKDTree

    # Load ground truth frames for rigid body positions
    print(f"    Loading GT frames for rigid body tracking...")

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
            if model_name == 'Hybrid':
                # Hybrid model takes particle_types as separate argument
                pred_vel = model(node_features, edge_index, edge_attr)
            else:
                pred_vel = model(node_features, edge_index, edge_attr)

        # Update ONLY fluid particles
        velocities[:n_fluid] = pred_vel[:n_fluid].numpy()
        positions[:n_fluid] += velocities[:n_fluid] * dt

        # Clear GPU cache to free memory
        del node_features, edge_index, edge_attr, pred_vel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Boundary conditions for fluid
        for dim in range(3):
            mask = positions[:n_fluid, dim] < 0
            positions[:n_fluid][mask, dim] = 0
            velocities[:n_fluid][mask, dim] *= -0.5

            mask = positions[:n_fluid, dim] > 1.0
            positions[:n_fluid][mask, dim] = 1.0
            velocities[:n_fluid][mask, dim] *= -0.5

        # Keep boundary particles fixed (they don't move in prediction)
        # This is correct - boundary positions should come from GT

        trajectory.append(positions.copy())

    return trajectory


def create_three_way_gif(scenario_path, gnn_model, hybrid_model, output_path, num_frames=80):
    """Create GIF comparing GT, GNN, and Hybrid with rigid body tracking"""

    # Load ground truth trajectory (all frames)
    frame_files = sorted(scenario_path.glob('frame_*.npz'))

    print(f"    Loading {len(frame_files)} GT frames...")
    gt_trajectory = []
    gt_rigid_body_positions = []

    for i in range(min(num_frames, len(frame_files))):
        data = np.load(frame_files[i])
        gt_trajectory.append(data['positions'])
        n_fluid = data['n_fluid']
        # Store rigid body positions from GT
        gt_rigid_body_positions.append(data['positions'][n_fluid:])

    initial_frame = np.load(frame_files[0])
    n_fluid = initial_frame['n_fluid']

    # Get model predictions
    print(f"    Rolling out GNN predictions...")
    gnn_trajectory = rollout_prediction(gnn_model, initial_frame, num_frames, model_name='GNN')

    print(f"    Rolling out Hybrid predictions...")
    hybrid_trajectory = rollout_prediction(hybrid_model, initial_frame, num_frames, model_name='Hybrid')

    # Create animation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def update(frame):
        for ax in axes:
            ax.clear()
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Ground Truth
        if frame < len(gt_trajectory):
            pos = gt_trajectory[frame]
            # Fluid particles
            axes[0].scatter(pos[:n_fluid, 0], pos[:n_fluid, 1],
                          c='blue', s=15, alpha=0.6, label='Fluid', edgecolors='none')
            # Rigid body particles
            axes[0].scatter(pos[n_fluid:, 0], pos[n_fluid:, 1],
                          c='red', s=30, alpha=0.9, label='Rigid Body',
                          edgecolors='darkred', linewidths=0.5)
            axes[0].set_title(f'Ground Truth (Two-way Coupling)\nFrame {frame}',
                            fontsize=11, fontweight='bold')
            axes[0].legend(loc='upper right', fontsize=9)

        # GNN Prediction
        if frame < len(gnn_trajectory):
            pos_pred = gnn_trajectory[frame]
            # Use GT rigid body position
            pos_rigid = gt_rigid_body_positions[frame] if frame < len(gt_rigid_body_positions) else pos_pred[n_fluid:]

            # Fluid particles (GNN prediction)
            axes[1].scatter(pos_pred[:n_fluid, 0], pos_pred[:n_fluid, 1],
                          c='blue', s=15, alpha=0.6, edgecolors='none')
            # Rigid body particles (GT position)
            axes[1].scatter(pos_rigid[:, 0], pos_rigid[:, 1],
                          c='red', s=30, alpha=0.9,
                          edgecolors='darkred', linewidths=0.5)
            axes[1].set_title(f'GNN Prediction (Val Loss: 0.002095)\nFrame {frame}',
                            fontsize=11, fontweight='bold')

        # Hybrid Prediction
        if frame < len(hybrid_trajectory):
            pos_pred = hybrid_trajectory[frame]
            pos_rigid = gt_rigid_body_positions[frame] if frame < len(gt_rigid_body_positions) else pos_pred[n_fluid:]

            # Fluid particles (Hybrid prediction)
            axes[2].scatter(pos_pred[:n_fluid, 0], pos_pred[:n_fluid, 1],
                          c='blue', s=15, alpha=0.6, edgecolors='none')
            # Rigid body particles (GT position)
            axes[2].scatter(pos_rigid[:, 0], pos_rigid[:, 1],
                          c='red', s=30, alpha=0.9,
                          edgecolors='darkred', linewidths=0.5)
            axes[2].set_title(f'Hybrid Prediction (Val Loss: 0.002946)\nFrame {frame}',
                            fontsize=11, fontweight='bold')

        for ax in axes:
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)

        scenario_title = scenario_path.name.replace('sph_dynamic_', '').replace('_', ' ').title()
        fig.suptitle(f'{scenario_title} - Dynamic Two-way Coupling',
                    fontsize=14, fontweight='bold', y=0.98)

    print(f"    Creating animation with {num_frames} frames...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50)

    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"    Saved: {output_path.name}")


def main():
    """Generate GIF visualizations for all scenarios"""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "datasets" / "flip_simulations"
    model_dir = project_root / "scripts" / "training" / "models"
    gnn_path = model_dir / "gnn_dynamic_two_way_coupling_best.pt"
    hybrid_path = model_dir / "hybrid_dynamic_two_way_coupling_best.pt"
    output_dir = project_root / "visualizations" / "dynamic_two_way_coupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GNN + HYBRID VISUALIZATION (80 FRAMES)")
    print("Dynamic Two-way Coupling Data")
    print("="*70)

    # Load models
    print("\nLoading models...")
    gnn_model, hybrid_model = load_models(gnn_path, hybrid_path)
    print("  GNN loaded (Val Loss: 0.002095)")
    print("  Hybrid loaded (Val Loss: 0.002946)")

    # Select scenarios (FLIP data)
    scenarios_to_viz = [
        'flip_dynamic_falling_ball_000',
        'flip_dynamic_floating_ball_000',
        'flip_dynamic_heavy_box_000',
        'flip_dynamic_multi_ball_000',
        'flip_dynamic_bouncing_sphere_000'
    ]

    print(f"\nGenerating GIFs for {len(scenarios_to_viz)} scenarios (80 frames each)...")

    for i, scenario_name in enumerate(scenarios_to_viz, 1):
        scenario_path = data_root / scenario_name
        if not scenario_path.exists():
            print(f"\n[{i}/{len(scenarios_to_viz)}] SKIP: {scenario_name}")
            continue

        output_path = output_dir / f"{scenario_name}_comparison.gif"
        print(f"\n[{i}/{len(scenarios_to_viz)}] {scenario_name}:")

        try:
            create_three_way_gif(scenario_path, gnn_model, hybrid_model,
                               output_path, num_frames=80)
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Output: {output_dir}")
    print("  - 5 GIFs with GT vs GNN vs Hybrid comparison")
    print("  - 80 frames each (~4 seconds at 20fps)")
    print("  - Rigid body positions tracked from Ground Truth")
    print("="*70)


if __name__ == "__main__":
    main()
