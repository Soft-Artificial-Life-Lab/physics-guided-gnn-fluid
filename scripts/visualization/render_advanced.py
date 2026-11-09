"""
Advanced Visualization with Ground Truth Comparison
- Metaball/surface rendering
- Velocity vectors
- Density-based coloring
- Side-by-side comparison: Ground Truth vs GNN Prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gnn import MeshGraphNet


def build_graph(positions, radius=0.3):
    """Build k-NN graph"""
    tree = cKDTree(positions)
    edges = []
    edge_features = []

    for i, pos in enumerate(positions):
        neighbors = tree.query_ball_point(pos, radius)
        for j in neighbors:
            if i != j:
                edges.append([i, j])
                edge_features.append(positions[j] - positions[i])

    if len(edges) == 0:
        return np.array([[0], [0]]), np.array([[0, 0, 0]])

    return np.array(edges).T, np.array(edge_features)


def compute_density(positions, h=0.1):
    """Compute SPH density for each particle"""
    tree = cKDTree(positions)
    densities = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        neighbors = tree.query_ball_point(pos, h)
        for j in neighbors:
            r = np.linalg.norm(positions[j] - pos)
            if r < h:
                # Poly6 kernel
                densities[i] += (315.0 / (64.0 * np.pi * h**9)) * (h**2 - r**2)**3

    return densities


def create_metaball_field(positions, grid_res=50, h=0.15, threshold=0.5):
    """
    Create metaball scalar field for isosurface rendering

    Args:
        positions: Particle positions (N, 3)
        grid_res: Grid resolution per axis
        h: Influence radius
        threshold: Isosurface threshold

    Returns:
        X, Y, Z: Grid coordinates
        field: Scalar field values
    """
    # Create 3D grid
    x = np.linspace(0, 1, grid_res)
    y = np.linspace(0, 1, grid_res)
    z = np.linspace(0, 1, grid_res)
    X, Y, Z = np.meshgrid(x, y, z)

    # Initialize field
    field = np.zeros((grid_res, grid_res, grid_res))

    # Compute field at each grid point
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                grid_point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])

                # Sum contributions from all particles
                for particle_pos in positions:
                    r = np.linalg.norm(grid_point - particle_pos)
                    if r < h and r > 0:
                        # Metaball function: f(r) = 1 - (r/h)^2
                        field[i, j, k] += max(0, 1 - (r / h) ** 2)

    return X, Y, Z, field


def create_metaball_contour_2d(positions, axis='z', slice_pos=0.5, grid_res=100, h=0.15):
    """
    Create 2D metaball contour for a slice

    Args:
        positions: Particle positions (N, 3)
        axis: Axis perpendicular to slice ('x', 'y', or 'z')
        slice_pos: Position of slice along axis
        grid_res: Grid resolution
        h: Influence radius

    Returns:
        X, Y: Grid coordinates for 2D slice
        field: Scalar field values
    """
    if axis == 'z':
        # XY plane slice
        x = np.linspace(0, 1, grid_res)
        y = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)

        # Filter particles near slice
        slice_mask = np.abs(positions[:, 2] - slice_pos) < h
        relevant_pos = positions[slice_mask]

        field = np.zeros((grid_res, grid_res))
        for i in range(grid_res):
            for j in range(grid_res):
                grid_point = np.array([X[i, j], Y[i, j], slice_pos])
                for particle_pos in relevant_pos:
                    r = np.linalg.norm(grid_point - particle_pos)
                    if r < h and r > 0:
                        field[i, j] += max(0, 1 - (r / h) ** 2)

    elif axis == 'y':
        # XZ plane slice
        x = np.linspace(0, 1, grid_res)
        z = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, z)

        slice_mask = np.abs(positions[:, 1] - slice_pos) < h
        relevant_pos = positions[slice_mask]

        field = np.zeros((grid_res, grid_res))
        for i in range(grid_res):
            for j in range(grid_res):
                grid_point = np.array([X[i, j], slice_pos, Y[i, j]])
                for particle_pos in relevant_pos:
                    r = np.linalg.norm(grid_point - particle_pos)
                    if r < h and r > 0:
                        field[i, j] += max(0, 1 - (r / h) ** 2)

    return X, Y, field


def rollout_simulation_with_ground_truth(
    model,
    initial_fluid_positions,
    initial_fluid_velocities,
    boundary_positions_over_time,
    boundary_velocities_over_time,
    ground_truth_positions,
    ground_truth_velocities,
    num_steps=50,
    device='cuda',
    dt=0.01,
    radius=0.3
):
    """
    Rollout simulation and return both prediction and ground truth

    Returns:
        pred_fluid_history, pred_vel_history,
        gt_fluid_history, gt_vel_history,
        boundary_history
    """
    pred_fluid_history = [initial_fluid_positions.copy()]
    pred_vel_history = [initial_fluid_velocities.copy()]

    gt_fluid_history = [ground_truth_positions[0].copy()]
    gt_vel_history = [ground_truth_velocities[0].copy()]

    current_fluid_pos = initial_fluid_positions.copy()
    current_fluid_vel = initial_fluid_velocities.copy()

    model.eval()
    with torch.no_grad():
        for step in range(min(num_steps, len(ground_truth_positions) - 1)):
            # Get boundary particles
            boundary_pos = boundary_positions_over_time[step]
            boundary_vel = boundary_velocities_over_time[step]

            # Combine fluid and boundary
            all_positions = np.vstack([current_fluid_pos, boundary_pos])
            all_velocities = np.vstack([current_fluid_vel, boundary_vel])

            # Particle types
            n_fluid = len(current_fluid_pos)
            n_boundary = len(boundary_pos)
            particle_types = np.hstack([np.zeros(n_fluid), np.ones(n_boundary)])

            # Build graph
            edge_index, edge_attr = build_graph(all_positions, radius=radius)

            # Node features
            node_features = np.concatenate([
                all_positions,
                all_velocities,
                particle_types[:, np.newaxis]
            ], axis=1)

            # Convert to tensors
            node_features_t = torch.FloatTensor(node_features).to(device)
            edge_index_t = torch.LongTensor(edge_index).to(device)
            edge_attr_t = torch.FloatTensor(edge_attr).to(device)

            # Predict
            pred_vel = model(node_features_t, edge_index_t, edge_attr_t).cpu().numpy()

            # Update only fluid particles
            current_fluid_vel = pred_vel[:n_fluid]
            current_fluid_pos = current_fluid_pos + current_fluid_vel * dt

            # Boundary conditions
            current_fluid_pos = np.clip(current_fluid_pos, 0, 1.0)

            # Store
            pred_fluid_history.append(current_fluid_pos.copy())
            pred_vel_history.append(current_fluid_vel.copy())

            # Ground truth
            gt_fluid_history.append(ground_truth_positions[step + 1].copy())
            gt_vel_history.append(ground_truth_velocities[step + 1].copy())

    boundary_history = [boundary_positions_over_time[0]] + boundary_positions_over_time[:num_steps]

    return (pred_fluid_history, pred_vel_history,
            gt_fluid_history, gt_vel_history,
            boundary_history)


def create_advanced_comparison_gif(
    model_path,
    data_scenario_path,
    output_path="visualizations/advanced_comparison.gif",
    num_steps=60,
    device='cuda',
    visualization_mode='density'  # 'density', 'velocity', 'metaball'
):
    """
    Create advanced visualization comparing ground truth and prediction

    Args:
        visualization_mode: 'density', 'velocity', or 'metaball'
    """
    print("="*70)
    print("ADVANCED VISUALIZATION WITH COMPARISON")
    print("="*70)
    print(f"Mode: {visualization_mode}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint and 'model' in checkpoint['config']:
        config = checkpoint['config']['model']
    else:
        config = {
            'input_node_dim': 7,
            'input_edge_dim': 3,
            'hidden_dim': 64,
            'output_dim': 3,
            'num_layers': 5
        }

    model = MeshGraphNet(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully ({num_params:,} parameters)")

    # Load scenario data
    print(f"\nLoading scenario data from {data_scenario_path}...")
    data_path = Path(data_scenario_path)
    frame_files = sorted(data_path.glob("frame_*.npz"))

    if len(frame_files) == 0:
        raise ValueError(f"No frame files found in {data_scenario_path}")

    # Load all ground truth data
    gt_positions = []
    gt_velocities = []
    boundary_positions_over_time = []
    boundary_velocities_over_time = []

    for frame_file in frame_files[:num_steps + 1]:
        data = np.load(frame_file)
        all_pos = data['positions']
        all_vel = data['velocities']
        particle_types = data['particle_types']
        n_fluid = data['n_fluid']

        gt_positions.append(all_pos[:n_fluid])
        gt_velocities.append(all_vel[:n_fluid])
        boundary_positions_over_time.append(all_pos[n_fluid:])
        boundary_velocities_over_time.append(all_vel[n_fluid:])

    print(f"Loaded {len(gt_positions)} frames")
    print(f"Fluid particles: {len(gt_positions[0])}")
    print(f"Boundary particles: {len(boundary_positions_over_time[0])}")

    # Run simulation
    print(f"\nRunning rollout simulation for {num_steps} steps...")
    (pred_pos_history, pred_vel_history,
     gt_pos_history, gt_vel_history,
     boundary_history) = rollout_simulation_with_ground_truth(
        model,
        gt_positions[0],
        gt_velocities[0],
        boundary_positions_over_time,
        boundary_velocities_over_time,
        gt_positions,
        gt_velocities,
        num_steps=num_steps,
        device=device
    )

    # Create figure with comparison
    print(f"\nCreating animated GIF with mode: {visualization_mode}...")
    fig = plt.figure(figsize=(20, 10))

    # 2 rows x 3 columns: GT top row, Pred bottom row
    axes = []
    for i in range(6):
        if i % 3 == 2:  # 3D views
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(2, 3, i + 1)
        axes.append(ax)

    def update(frame):
        for ax in axes:
            ax.clear()

        # Get data
        gt_pos = gt_pos_history[frame]
        pred_pos = pred_pos_history[frame]
        gt_vel = gt_vel_history[frame]
        pred_vel = pred_vel_history[frame]
        boundary_pos = boundary_history[frame]

        # Remap to Z-up
        gt_pos_r = np.column_stack([gt_pos[:, 0], gt_pos[:, 2], gt_pos[:, 1]])
        pred_pos_r = np.column_stack([pred_pos[:, 0], pred_pos[:, 2], pred_pos[:, 1]])
        gt_vel_r = np.column_stack([gt_vel[:, 0], gt_vel[:, 2], gt_vel[:, 1]])
        pred_vel_r = np.column_stack([pred_vel[:, 0], pred_vel[:, 2], pred_vel[:, 1]])

        if len(boundary_pos) > 0:
            boundary_pos_r = np.column_stack([boundary_pos[:, 0], boundary_pos[:, 2], boundary_pos[:, 1]])
        else:
            boundary_pos_r = np.array([]).reshape(0, 3)

        # Compute densities or metaballs if needed
        if visualization_mode == 'density':
            gt_density = compute_density(gt_pos_r)
            pred_density = compute_density(pred_pos_r)
            gt_colors = gt_density
            pred_colors = pred_density
            cmap = 'viridis'
            vmin, vmax = 0, max(gt_density.max(), pred_density.max())
        elif visualization_mode == 'metaball':
            # Compute metaball contours for 2D views
            gt_X_xy, gt_Y_xy, gt_field_xy = create_metaball_contour_2d(
                gt_pos_r, axis='z', slice_pos=0.1, grid_res=80, h=0.12)
            pred_X_xy, pred_Y_xy, pred_field_xy = create_metaball_contour_2d(
                pred_pos_r, axis='z', slice_pos=0.1, grid_res=80, h=0.12)
            gt_colors = 'blue'
            pred_colors = 'blue'
            cmap = None
            vmin, vmax = None, None
        else:
            gt_colors = 'blue'
            pred_colors = 'blue'
            cmap = None
            vmin, vmax = None, None

        # Row 1: Ground Truth
        # Top view XY
        if visualization_mode == 'metaball':
            # Draw metaball contours
            axes[0].contourf(gt_X_xy, gt_Y_xy, gt_field_xy, levels=10,
                           cmap='Blues', alpha=0.6)
            axes[0].contour(gt_X_xy, gt_Y_xy, gt_field_xy, levels=[0.5],
                          colors='blue', linewidths=2)
        elif visualization_mode == 'density':
            axes[0].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], c=gt_colors, s=20,
                          alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[0].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], c=gt_colors, s=20, alpha=0.7)

        if len(boundary_pos_r) > 0:
            axes[0].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 1],
                          c='red', s=15, marker='s', alpha=0.8)

        # Add velocity vectors if mode is velocity
        if visualization_mode == 'velocity':
            subsample = np.arange(0, len(gt_pos_r), 10)
            axes[0].quiver(gt_pos_r[subsample, 0], gt_pos_r[subsample, 1],
                         gt_vel_r[subsample, 0], gt_vel_r[subsample, 1],
                         color='red', alpha=0.6, width=0.003, scale=10)

        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title(f'Ground Truth - Top View (XY) - Frame {frame}/{len(gt_pos_history)-1}')
        axes[0].grid(True, alpha=0.3)

        # Side view XZ
        if visualization_mode == 'density':
            axes[1].scatter(gt_pos_r[:, 0], gt_pos_r[:, 2], c=gt_colors, s=20,
                          alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[1].scatter(gt_pos_r[:, 0], gt_pos_r[:, 2], c=gt_colors, s=20, alpha=0.7)

        if len(boundary_pos_r) > 0:
            axes[1].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 2],
                          c='red', s=15, marker='s', alpha=0.8)

        if visualization_mode == 'velocity':
            axes[1].quiver(gt_pos_r[subsample, 0], gt_pos_r[subsample, 2],
                         gt_vel_r[subsample, 0], gt_vel_r[subsample, 2],
                         color='red', alpha=0.6, width=0.003, scale=10)

        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z (height)')
        axes[1].set_title('Ground Truth - Side View (XZ)')
        axes[1].grid(True, alpha=0.3)

        # 3D view
        if visualization_mode == 'density':
            axes[2].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], gt_pos_r[:, 2],
                          c=gt_colors, s=20, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[2].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], gt_pos_r[:, 2],
                          c=gt_colors, s=20, alpha=0.6)

        if len(boundary_pos_r) > 0:
            axes[2].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 1], boundary_pos_r[:, 2],
                          c='red', s=25, marker='s', alpha=0.9)

        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].set_zlim(0, 1)
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].set_zlabel('Z (height)')
        axes[2].set_title('Ground Truth - 3D View')
        axes[2].view_init(elev=25, azim=45 + frame * 2)

        # Row 2: GNN Prediction
        # Top view XY
        if visualization_mode == 'density':
            axes[3].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], c=pred_colors, s=20,
                          alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[3].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], c=pred_colors, s=20, alpha=0.7)

        if len(boundary_pos_r) > 0:
            axes[3].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 1],
                          c='red', s=15, marker='s', alpha=0.8)

        if visualization_mode == 'velocity':
            axes[3].quiver(pred_pos_r[subsample, 0], pred_pos_r[subsample, 1],
                         pred_vel_r[subsample, 0], pred_vel_r[subsample, 1],
                         color='red', alpha=0.6, width=0.003, scale=10)

        axes[3].set_xlim(0, 1)
        axes[3].set_ylim(0, 1)
        axes[3].set_xlabel('X')
        axes[3].set_ylabel('Y')
        axes[3].set_title('GNN Prediction - Top View (XY)')
        axes[3].grid(True, alpha=0.3)

        # Side view XZ
        if visualization_mode == 'density':
            axes[4].scatter(pred_pos_r[:, 0], pred_pos_r[:, 2], c=pred_colors, s=20,
                          alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[4].scatter(pred_pos_r[:, 0], pred_pos_r[:, 2], c=pred_colors, s=20, alpha=0.7)

        if len(boundary_pos_r) > 0:
            axes[4].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 2],
                          c='red', s=15, marker='s', alpha=0.8)

        if visualization_mode == 'velocity':
            axes[4].quiver(pred_pos_r[subsample, 0], pred_pos_r[subsample, 2],
                         pred_vel_r[subsample, 0], pred_vel_r[subsample, 2],
                         color='red', alpha=0.6, width=0.003, scale=10)

        axes[4].set_xlim(0, 1)
        axes[4].set_ylim(0, 1)
        axes[4].set_xlabel('X')
        axes[4].set_ylabel('Z (height)')
        axes[4].set_title('GNN Prediction - Side View (XZ)')
        axes[4].grid(True, alpha=0.3)

        # 3D view
        if visualization_mode == 'density':
            axes[5].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], pred_pos_r[:, 2],
                          c=pred_colors, s=20, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[5].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], pred_pos_r[:, 2],
                          c=pred_colors, s=20, alpha=0.6)

        if len(boundary_pos_r) > 0:
            axes[5].scatter(boundary_pos_r[:, 0], boundary_pos_r[:, 1], boundary_pos_r[:, 2],
                          c='red', s=25, marker='s', alpha=0.9)

        axes[5].set_xlim(0, 1)
        axes[5].set_ylim(0, 1)
        axes[5].set_zlim(0, 1)
        axes[5].set_xlabel('X')
        axes[5].set_ylabel('Y')
        axes[5].set_zlabel('Z (height)')
        axes[5].set_title('GNN Prediction - 3D View')
        axes[5].view_init(elev=25, azim=45 + frame * 2)

        plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(gt_pos_history), interval=100)

    # Save as GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)

    print(f"\n[SAVED] Advanced comparison GIF: {output_path}")
    print(f"  Frames: {len(gt_pos_history)}")
    print(f"  Mode: {visualization_mode}")
    print(f"  Resolution: 2000×1000")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Advanced visualization with comparison')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('data_path', type=str, help='Path to scenario data directory')
    parser.add_argument('--output', type=str, default=None, help='Output GIF path')
    parser.add_argument('--mode', type=str, default='density',
                       choices=['density', 'velocity', 'metaball'],
                       help='Visualization mode')
    parser.add_argument('--steps', type=int, default=60, help='Number of simulation steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    if args.output is None:
        scenario_name = Path(args.data_path).name
        args.output = f"visualizations/advanced_{args.mode}_{scenario_name}.gif"

    create_advanced_comparison_gif(
        args.model_path,
        args.data_path,
        output_path=args.output,
        num_steps=args.steps,
        device=args.device,
        visualization_mode=args.mode
    )
