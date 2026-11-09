"""
Advanced Visualization with Ground Truth Comparison (v2)
- Metaball surface rendering on ALL views
- Particle overlay option
- Velocity vectors
- Density-based coloring
- Proper folder structure for output
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import torch
from scipy.spatial import cKDTree
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gnn import MeshGraphNet
from src.models.gat import GraphAttentionNet
from src.models.hybrid_gnn import HybridPhysicsGNN


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
                densities[i] += (315.0 / (64.0 * np.pi * h**9)) * (h**2 - r**2)**3

    return densities


def create_metaball_contour_2d(positions, axis='z', slice_pos=0.5, grid_res=100, h=0.12):
    """Create 2D metaball contour for a slice"""
    if axis == 'z':
        # XY plane slice
        x = np.linspace(0, 1, grid_res)
        y = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)

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
    model, initial_fluid_positions, initial_fluid_velocities,
    boundary_positions_over_time, boundary_velocities_over_time,
    ground_truth_positions, ground_truth_velocities,
    num_steps=50, device='cuda', dt=0.01, radius=0.3
):
    """Rollout simulation and return both prediction and ground truth"""
    pred_fluid_history = [initial_fluid_positions.copy()]
    pred_vel_history = [initial_fluid_velocities.copy()]
    gt_fluid_history = [ground_truth_positions[0].copy()]
    gt_vel_history = [ground_truth_velocities[0].copy()]

    current_fluid_pos = initial_fluid_positions.copy()
    current_fluid_vel = initial_fluid_velocities.copy()

    model.eval()
    with torch.no_grad():
        for step in range(min(num_steps, len(ground_truth_positions) - 1)):
            boundary_pos = boundary_positions_over_time[step]
            boundary_vel = boundary_velocities_over_time[step]

            all_positions = np.vstack([current_fluid_pos, boundary_pos])
            all_velocities = np.vstack([current_fluid_vel, boundary_vel])

            n_fluid = len(current_fluid_pos)
            n_boundary = len(boundary_pos)
            particle_types = np.hstack([np.zeros(n_fluid), np.ones(n_boundary)])

            edge_index, edge_attr = build_graph(all_positions, radius=radius)

            node_features = np.concatenate([
                all_positions, all_velocities,
                particle_types[:, np.newaxis]
            ], axis=1)

            node_features_t = torch.FloatTensor(node_features).to(device)
            edge_index_t = torch.LongTensor(edge_index).to(device)
            edge_attr_t = torch.FloatTensor(edge_attr).to(device)

            pred_vel = model(node_features_t, edge_index_t, edge_attr_t).cpu().numpy()

            current_fluid_vel = pred_vel[:n_fluid]
            current_fluid_pos = current_fluid_pos + current_fluid_vel * dt
            current_fluid_pos = np.clip(current_fluid_pos, 0, 1.0)

            pred_fluid_history.append(current_fluid_pos.copy())
            pred_vel_history.append(current_fluid_vel.copy())
            gt_fluid_history.append(ground_truth_positions[step + 1].copy())
            gt_vel_history.append(ground_truth_velocities[step + 1].copy())

    boundary_history = [boundary_positions_over_time[0]] + boundary_positions_over_time[:num_steps]

    return (pred_fluid_history, pred_vel_history,
            gt_fluid_history, gt_vel_history,
            boundary_history)


def create_advanced_comparison_gif(
    model_path,
    data_scenario_path,
    output_path=None,
    num_steps=60,
    device='cuda',
    visualization_mode='metaball',
    show_particles=True
):
    """
    Create advanced visualization comparing ground truth and prediction

    Args:
        show_particles: Show particles on top of metaball (default: True)
    """
    print("="*70)
    print("ADVANCED VISUALIZATION WITH COMPARISON (V2)")
    print("="*70)
    print(f"Mode: {visualization_mode}")
    print(f"Show particles: {show_particles}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {}).get('model', {
        'input_node_dim': 7, 'input_edge_dim': 3,
        'hidden_dim': 64, 'output_dim': 3, 'num_layers': 5
    })

    model = MeshGraphNet(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {num_params:,} parameters")

    # Load scenario data
    data_path = Path(data_scenario_path)
    frame_files = sorted(data_path.glob("frame_*.npz"))

    gt_positions, gt_velocities = [], []
    boundary_positions_over_time, boundary_velocities_over_time = [], []

    for frame_file in frame_files[:num_steps + 1]:
        data = np.load(frame_file)
        n_fluid = data['n_fluid']
        gt_positions.append(data['positions'][:n_fluid])
        gt_velocities.append(data['velocities'][:n_fluid])
        boundary_positions_over_time.append(data['positions'][n_fluid:])
        boundary_velocities_over_time.append(data['velocities'][n_fluid:])

    print(f"Loaded {len(gt_positions)} frames")

    # Run simulation
    (pred_pos_history, pred_vel_history, gt_pos_history, gt_vel_history,
     boundary_history) = rollout_simulation_with_ground_truth(
        model, gt_positions[0], gt_velocities[0],
        boundary_positions_over_time, boundary_velocities_over_time,
        gt_positions, gt_velocities, num_steps=num_steps, device=device
    )

    # Determine output path based on folder structure
    if output_path is None:
        scenario_name = data_path.name
        # Extract scenario type (e.g., "falling_ball" from "sph_falling_ball_000")
        scenario_type = scenario_name.replace('sph_', '').rsplit('_', 1)[0]

        output_dir = Path("visualizations") / "gnn_rigid_body" / scenario_type
        output_dir.mkdir(parents=True, exist_ok=True)

        particle_suffix = "_with_particles" if show_particles else "_surface_only"
        output_path = output_dir / f"advanced_{visualization_mode}{particle_suffix}.gif"

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    axes = []
    for i in range(6):
        if i % 3 == 2:
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(2, 3, i + 1)
        axes.append(ax)

    def update(frame):
        for ax in axes:
            ax.clear()

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

        # Compute visualization data
        if visualization_mode == 'density':
            gt_density = compute_density(gt_pos_r)
            pred_density = compute_density(pred_pos_r)
            gt_colors, pred_colors = gt_density, pred_density
            cmap, vmin, vmax = 'viridis', 0, max(gt_density.max(), pred_density.max())
            # No metaball data
            gt_X_xy, gt_Y_xy, gt_field_xy = None, None, None
            pred_X_xy, pred_Y_xy, pred_field_xy = None, None, None
            gt_X_xz, gt_Y_xz, gt_field_xz = None, None, None
            pred_X_xz, pred_Y_xz, pred_field_xz = None, None, None
        elif visualization_mode == 'metaball':
            # Compute metaball fields for all views
            gt_X_xy, gt_Y_xy, gt_field_xy = create_metaball_contour_2d(
                gt_pos_r, axis='z', slice_pos=0.1, grid_res=80, h=0.12)
            pred_X_xy, pred_Y_xy, pred_field_xy = create_metaball_contour_2d(
                pred_pos_r, axis='z', slice_pos=0.1, grid_res=80, h=0.12)

            gt_X_xz, gt_Y_xz, gt_field_xz = create_metaball_contour_2d(
                gt_pos_r, axis='y', slice_pos=0.5, grid_res=80, h=0.12)
            pred_X_xz, pred_Y_xz, pred_field_xz = create_metaball_contour_2d(
                pred_pos_r, axis='y', slice_pos=0.5, grid_res=80, h=0.12)

            gt_colors, pred_colors = 'blue', 'blue'
            cmap, vmin, vmax = None, None, None
        else:
            # velocity or other modes
            gt_colors, pred_colors = 'blue', 'blue'
            cmap, vmin, vmax = None, None, None
            # No metaball data
            gt_X_xy, gt_Y_xy, gt_field_xy = None, None, None
            pred_X_xy, pred_Y_xy, pred_field_xy = None, None, None
            gt_X_xz, gt_Y_xz, gt_field_xz = None, None, None
            pred_X_xz, pred_Y_xz, pred_field_xz = None, None, None

        # Helper function to draw view
        def draw_2d_view(ax, pos, vel, colors, X_meta, Y_meta, field_meta,
                        view_name, is_pred=False, axis_names=('X', 'Y')):
            # Draw metaball surface
            if visualization_mode == 'metaball' and X_meta is not None:
                ax.contourf(X_meta, Y_meta, field_meta, levels=10,
                          cmap='Blues', alpha=0.4)
                ax.contour(X_meta, Y_meta, field_meta, levels=[0.5],
                          colors='darkblue', linewidths=2, alpha=0.8)

            # Draw particles
            if show_particles or visualization_mode != 'metaball':
                if visualization_mode == 'density':
                    ax.scatter(pos[:, 0] if 'X' in axis_names[0] else pos[:, 1],
                             pos[:, 1] if len(axis_names) > 1 and 'Y' in axis_names[1] else pos[:, 2],
                             c=colors, s=15, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    ax.scatter(pos[:, 0] if 'X' in axis_names[0] else pos[:, 1],
                             pos[:, 1] if len(axis_names) > 1 and 'Y' in axis_names[1] else pos[:, 2],
                             c=colors, s=15, alpha=0.5)

            # Draw boundary
            if len(boundary_pos_r) > 0:
                ax.scatter(boundary_pos_r[:, 0] if 'X' in axis_names[0] else boundary_pos_r[:, 1],
                         boundary_pos_r[:, 1] if len(axis_names) > 1 and 'Y' in axis_names[1] else boundary_pos_r[:, 2],
                         c='red', s=15, marker='s', alpha=0.8)

            # Draw velocity vectors
            if visualization_mode == 'velocity':
                subsample = np.arange(0, len(pos), 10)
                if 'X' in axis_names[0]:
                    ax.quiver(pos[subsample, 0], pos[subsample, 1] if 'Y' in axis_names[1] else pos[subsample, 2],
                            vel[subsample, 0], vel[subsample, 1] if 'Y' in axis_names[1] else vel[subsample, 2],
                            color='red', alpha=0.6, width=0.003, scale=10)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            prefix = "GNN Prediction" if is_pred else "Ground Truth"
            ax.set_title(f'{prefix} - {view_name}')
            ax.grid(True, alpha=0.3)

        # Row 1: Ground Truth
        draw_2d_view(axes[0], gt_pos_r, gt_vel_r, gt_colors,
                    gt_X_xy, gt_Y_xy, gt_field_xy if visualization_mode == 'metaball' else None,
                    f'Top View (XY) - Frame {frame}/{len(gt_pos_history)-1}',
                    axis_names=('X', 'Y'))

        draw_2d_view(axes[1], gt_pos_r, gt_vel_r, gt_colors,
                    gt_X_xz, gt_Y_xz, gt_field_xz if visualization_mode == 'metaball' else None,
                    'Side View (XZ)', axis_names=('X', 'Z (height)'))

        # 3D view GT
        if visualization_mode == 'density':
            axes[2].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], gt_pos_r[:, 2],
                          c=gt_colors, s=20, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[2].scatter(gt_pos_r[:, 0], gt_pos_r[:, 1], gt_pos_r[:, 2],
                          c=gt_colors, s=20, alpha=0.5 if show_particles else 0.6)

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
        draw_2d_view(axes[3], pred_pos_r, pred_vel_r, pred_colors,
                    pred_X_xy, pred_Y_xy, pred_field_xy if visualization_mode == 'metaball' else None,
                    'Top View (XY)', is_pred=True, axis_names=('X', 'Y'))

        draw_2d_view(axes[4], pred_pos_r, pred_vel_r, pred_colors,
                    pred_X_xz, pred_Y_xz, pred_field_xz if visualization_mode == 'metaball' else None,
                    'Side View (XZ)', is_pred=True, axis_names=('X', 'Z (height)'))

        # 3D view Pred
        if visualization_mode == 'density':
            axes[5].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], pred_pos_r[:, 2],
                          c=pred_colors, s=20, alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[5].scatter(pred_pos_r[:, 0], pred_pos_r[:, 1], pred_pos_r[:, 2],
                          c=pred_colors, s=20, alpha=0.5 if show_particles else 0.6)

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

    print(f"\n[SAVED] {output_path}")
    print(f"  Frames: {len(gt_pos_history)}")
    print(f"  Mode: {visualization_mode}")
    print(f"  Particles: {'shown' if show_particles else 'hidden'}")
    print("="*70)

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mode', type=str, default='metaball',
                       choices=['density', 'velocity', 'metaball'])
    parser.add_argument('--no-particles', action='store_true',
                       help='Hide particles (metaball surface only)')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    create_advanced_comparison_gif(
        args.model_path, args.data_path,
        output_path=args.output,
        num_steps=args.steps,
        device=args.device,
        visualization_mode=args.mode,
        show_particles=not args.no_particles
    )
