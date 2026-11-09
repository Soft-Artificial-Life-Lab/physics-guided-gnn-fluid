"""
3D Perspective Visualization with Rotation
Generates GIFs showing GT, GNN, and Hybrid predictions in true 3D space with rotating camera
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.gnn import MeshGraphNet
from src.models.hybrid_gnn import HybridPhysicsGNN


def load_model(model_path, model_type='gnn'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'gnn':
        model = MeshGraphNet(
            input_node_dim=7,
            input_edge_dim=3,
            hidden_dim=64,
            output_dim=3,
            num_layers=5
        ).to(device)
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
        ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device


def load_scenario_data(scenario_path, num_frames=80):
    """Load all frames from a scenario"""
    frame_files = sorted(scenario_path.glob('frame_*.npz'))
    frames = []

    for i in range(min(num_frames, len(frame_files))):
        data = np.load(frame_files[i])
        frames.append({
            'positions': data['positions'],
            'velocities': data['velocities'],
            'particle_types': data['particle_types'],
            'n_fluid': int(data['n_fluid'])
        })

    return frames


def build_graph(positions, velocities, particle_types, radius=0.3):
    """Build graph for GNN prediction"""
    from scipy.spatial import cKDTree

    n_particles = len(positions)
    tree = cKDTree(positions)

    # Build edges
    edge_list = []
    for i in range(n_particles):
        neighbors = tree.query_ball_point(positions[i], radius)
        for j in neighbors:
            if i != j:
                edge_list.append([i, j])

    if len(edge_list) == 0:
        edge_list = [[0, 0]]

    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    # Node features: [pos(3), vel(3), type(1)]
    node_features = torch.cat([
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(velocities, dtype=torch.float32),
        torch.tensor(particle_types, dtype=torch.float32).unsqueeze(1)
    ], dim=1)

    # Edge features: relative position
    edge_features = positions[edge_index[1]] - positions[edge_index[0]]
    edge_features = torch.tensor(edge_features, dtype=torch.float32)

    return node_features, edge_index, edge_features


def rollout_predictions(model, initial_frame, num_frames, device, graph_radius=0.3):
    """Rollout model predictions autoregressively"""
    trajectory = [initial_frame['positions'].copy()]

    current_pos = initial_frame['positions'].copy()
    current_vel = initial_frame['velocities'].copy()
    particle_types = initial_frame['particle_types']

    with torch.no_grad():
        for _ in range(num_frames - 1):
            # Build graph
            node_features, edge_index, edge_features = build_graph(
                current_pos, current_vel, particle_types, graph_radius
            )

            # Don't add batch dimension - models expect [2, num_edges]
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            edge_features = edge_features.to(device)

            # Predict acceleration
            acceleration = model(node_features, edge_index, edge_features)
            acceleration = acceleration.cpu().numpy()

            # Clear GPU cache to free memory
            del node_features, edge_index, edge_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Update velocity and position
            dt = 0.01
            current_vel = current_vel + acceleration * dt
            current_pos = current_pos + current_vel * dt

            trajectory.append(current_pos.copy())

    return trajectory


def create_3d_perspective_gif(scenario_path, gnn_model, hybrid_model, output_path, num_frames=80):
    """Create 3D perspective visualization with rotating camera"""

    print(f"    Loading {num_frames} GT frames...")
    gt_frames = load_scenario_data(scenario_path, num_frames)
    n_fluid = gt_frames[0]['n_fluid']

    # Get GT trajectory
    gt_trajectory = [frame['positions'] for frame in gt_frames]

    # Get rigid body positions separately
    gt_rigid_body_positions = [frame['positions'][n_fluid:] for frame in gt_frames]

    print(f"    Rolling out GNN predictions...")
    device_gnn = next(gnn_model.parameters()).device
    gnn_trajectory = rollout_predictions(gnn_model, gt_frames[0], num_frames, device_gnn)

    print(f"    Rolling out Hybrid predictions...")
    device_hybrid = next(hybrid_model.parameters()).device
    hybrid_trajectory = rollout_predictions(hybrid_model, gt_frames[0], num_frames, device_hybrid)

    print(f"    Creating 3D perspective animation ({num_frames} frames)...")

    # Create figure with 3 subplots (1 row, 3 columns) - all 3D
    fig = plt.figure(figsize=(24, 8))

    # All 3D subplots
    ax_gt = fig.add_subplot(131, projection='3d')
    ax_gnn = fig.add_subplot(132, projection='3d')
    ax_hybrid = fig.add_subplot(133, projection='3d')

    axes = [ax_gt, ax_gnn, ax_hybrid]

    # Set axis limits - SWAP Y and Z to make Z the vertical axis
    all_positions = np.concatenate(gt_trajectory)
    x_min, x_max = all_positions[:, 0].min() - 0.2, all_positions[:, 0].max() + 0.2
    y_min, y_max = all_positions[:, 2].min() - 0.2, all_positions[:, 2].max() + 0.2  # Z -> Y
    z_min, z_max = all_positions[:, 1].min() - 0.2, all_positions[:, 1].max() + 0.2  # Y -> Z (height)

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Height)')

    def update(frame):
        for ax in axes:
            ax.cla()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z (Height)')

        # Rotate camera: azimuth changes with frame
        elev = 20  # Fixed elevation
        azim = frame * 2  # Rotate 2 degrees per frame (360 degrees in 180 frames)

        # Ground Truth
        if frame < len(gt_trajectory):
            pos = gt_trajectory[frame]
            rigid_pos = gt_rigid_body_positions[frame]

            # Fluid particles (blue) - SWAP Y and Z: (X, Z, Y) -> (X, Y, Z_height)
            ax_gt.scatter(pos[:n_fluid, 0], pos[:n_fluid, 2], pos[:n_fluid, 1],
                         c='blue', s=20, alpha=0.6, label='Fluid')
            # Rigid body (red) - SWAP Y and Z
            ax_gt.scatter(rigid_pos[:, 0], rigid_pos[:, 2], rigid_pos[:, 1],
                         c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=0.5, label='Rigid Body')
            ax_gt.set_title(f'Ground Truth\nFrame {frame}', fontsize=14, fontweight='bold')
            ax_gt.view_init(elev=elev, azim=azim)
            ax_gt.legend()

        # GNN Prediction
        if frame < len(gnn_trajectory):
            pos_pred = gnn_trajectory[frame]
            rigid_pos = gt_rigid_body_positions[frame]  # Use GT rigid body

            # Fluid (GNN prediction) - SWAP Y and Z
            ax_gnn.scatter(pos_pred[:n_fluid, 0], pos_pred[:n_fluid, 2], pos_pred[:n_fluid, 1],
                          c='blue', s=20, alpha=0.6, label='Fluid (GNN)')
            # Rigid body (GT) - SWAP Y and Z
            ax_gnn.scatter(rigid_pos[:, 0], rigid_pos[:, 2], rigid_pos[:, 1],
                          c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=0.5, label='Rigid Body')
            ax_gnn.set_title(f'GNN Prediction\nVal Loss: 0.002095', fontsize=14, fontweight='bold')
            ax_gnn.view_init(elev=elev, azim=azim)
            ax_gnn.legend()

        # Hybrid Prediction
        if frame < len(hybrid_trajectory):
            pos_pred = hybrid_trajectory[frame]
            rigid_pos = gt_rigid_body_positions[frame]  # Use GT rigid body

            # Fluid (Hybrid prediction) - SWAP Y and Z
            ax_hybrid.scatter(pos_pred[:n_fluid, 0], pos_pred[:n_fluid, 2], pos_pred[:n_fluid, 1],
                             c='blue', s=20, alpha=0.6, label='Fluid (Hybrid)')
            # Rigid body (GT) - SWAP Y and Z
            ax_hybrid.scatter(rigid_pos[:, 0], rigid_pos[:, 2], rigid_pos[:, 1],
                             c='red', s=40, alpha=0.9, edgecolors='darkred', linewidths=0.5, label='Rigid Body')
            ax_hybrid.set_title(f'Hybrid Prediction\nVal Loss: 0.002946', fontsize=14, fontweight='bold')
            ax_hybrid.view_init(elev=elev, azim=azim)
            ax_hybrid.legend()

        plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50)

    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"    Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("3D PERSPECTIVE VISUALIZATION (Rotating Camera)")
    print("GT vs GNN vs Hybrid - 80 Frames with 360° Rotation")
    print("=" * 70)
    print()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "datasets" / "flip_simulations"
    model_dir = project_root / "scripts" / "training" / "models"
    output_dir = project_root / "visualizations" / "dynamic_two_way_coupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    gnn_model, _ = load_model(model_dir / "gnn_dynamic_two_way_coupling_best.pt", 'gnn')
    print("  GNN loaded (Val Loss: 0.002095)")

    hybrid_model, _ = load_model(model_dir / "hybrid_dynamic_two_way_coupling_best.pt", 'hybrid')
    print("  Hybrid loaded (Val Loss: 0.002946)")
    print()

    # Get one scenario from each type (FLIP data)
    scenario_types = [
        "flip_dynamic_falling_ball_000",
        "flip_dynamic_floating_ball_000",
        "flip_dynamic_heavy_box_000",
        "flip_dynamic_multi_ball_000",
        "flip_dynamic_bouncing_sphere_000"
    ]

    print(f"Generating 3D perspective GIFs for {len(scenario_types)} scenarios...")
    print()

    for idx, scenario_name in enumerate(scenario_types, 1):
        scenario_path = data_root / scenario_name

        if not scenario_path.exists():
            print(f"[{idx}/{len(scenario_types)}] {scenario_name}: NOT FOUND, skipping...")
            continue

        print(f"[{idx}/{len(scenario_types)}] {scenario_name}:")

        output_path = output_dir / f"{scenario_name}_3d_perspective.gif"

        create_3d_perspective_gif(
            scenario_path,
            gnn_model,
            hybrid_model,
            output_path,
            num_frames=80
        )

        # Clear GPU cache after each scenario to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    print("=" * 70)
    print("3D PERSPECTIVE VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"  - 5 GIFs with 3D rotating perspective")
    print(f"  - GT vs GNN vs Hybrid comparison")
    print(f"  - 80 frames each (20fps)")
    print(f"  - Camera rotates 360° around the scene")
    print("=" * 70)


if __name__ == "__main__":
    main()
