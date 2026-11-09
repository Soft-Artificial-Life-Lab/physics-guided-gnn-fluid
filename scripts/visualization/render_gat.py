"""
Visualize GAT (Graph Attention Network) rigid body - fluid interaction
Creates animated GIF showing fluid interacting with moving rigid boundaries using GAT model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import torch
from scipy.spatial import cKDTree
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gat import GraphAttentionNet


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


def rollout_simulation_with_rigid_body_gat(
    model,
    initial_fluid_positions,
    initial_fluid_velocities,
    boundary_positions_over_time,
    boundary_velocities_over_time,
    num_steps=50,
    device='cuda',
    dt=0.01,
    radius=0.3
):
    """
    Rollout simulation with moving rigid bodies using GAT model

    Args:
        model: Trained GAT model
        initial_fluid_positions: Initial fluid particle positions (N_fluid, 3)
        initial_fluid_velocities: Initial fluid particle velocities (N_fluid, 3)
        boundary_positions_over_time: List of boundary positions
        boundary_velocities_over_time: List of boundary velocities
        num_steps: Number of steps to simulate
        device: Device to run on
        dt: Time step
        radius: Graph connection radius

    Returns:
        fluid_positions_history, fluid_velocities_history, boundary_history
    """
    fluid_positions_history = [initial_fluid_positions.copy()]
    fluid_velocities_history = [initial_fluid_velocities.copy()]

    current_fluid_pos = initial_fluid_positions.copy()
    current_fluid_vel = initial_fluid_velocities.copy()

    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            # Get boundary particles at this timestep
            boundary_pos = boundary_positions_over_time[step]
            boundary_vel = boundary_velocities_over_time[step]

            # Combine fluid and boundary particles
            all_positions = np.vstack([current_fluid_pos, boundary_pos])
            all_velocities = np.vstack([current_fluid_vel, boundary_vel])

            # Particle types: 0=fluid, 1=boundary
            n_fluid = len(current_fluid_pos)
            n_boundary = len(boundary_pos)
            particle_types = np.hstack([
                np.zeros(n_fluid),
                np.ones(n_boundary)
            ])

            # Build graph
            edge_index, edge_attr = build_graph(all_positions, radius=radius)

            # Create node features: [pos, vel, particle_type]
            node_features = np.concatenate([
                all_positions,
                all_velocities,
                particle_types[:, np.newaxis]
            ], axis=1)

            # Convert to tensors
            node_features_t = torch.FloatTensor(node_features).to(device)
            edge_index_t = torch.LongTensor(edge_index).to(device)
            edge_attr_t = torch.FloatTensor(edge_attr).to(device)

            # Predict velocities using GAT
            pred_vel = model(node_features_t, edge_index_t, edge_attr_t)
            pred_vel = pred_vel.cpu().numpy()

            # Extract only fluid velocities
            pred_fluid_vel = pred_vel[:n_fluid]

            # Update fluid positions
            current_fluid_pos = current_fluid_pos + pred_fluid_vel * dt
            current_fluid_vel = pred_fluid_vel

            # Apply domain boundary conditions
            mask = current_fluid_pos[:, 2] < 0
            current_fluid_pos[mask, 2] = 0
            current_fluid_vel[mask, 2] = -current_fluid_vel[mask, 2] * 0.5

            for dim in range(3):
                mask = current_fluid_pos[:, dim] < 0
                current_fluid_pos[mask, dim] = 0
                current_fluid_vel[mask, dim] = -current_fluid_vel[mask, dim] * 0.5

                mask = current_fluid_pos[:, dim] > 1.0
                current_fluid_pos[mask, dim] = 1.0
                current_fluid_vel[mask, dim] = -current_fluid_vel[mask, dim] * 0.5

            # Store history
            fluid_positions_history.append(current_fluid_pos.copy())
            fluid_velocities_history.append(current_fluid_vel.copy())

    # Add initial boundary to history
    boundary_history = [boundary_positions_over_time[0]] + boundary_positions_over_time[:num_steps]
    return fluid_positions_history, fluid_velocities_history, boundary_history


def create_gat_rigid_body_gif(
    model_path,
    data_scenario_path,
    output_path="visualizations/gat_rigid_body_interaction.gif",
    num_steps=60,
    device='cuda'
):
    """
    Create animated GIF of GAT rigid body-fluid interaction

    Args:
        model_path: Path to trained GAT model
        data_scenario_path: Path to scenario data directory
        output_path: Where to save GIF
        num_steps: Number of simulation steps
        device: Device to run on
    """
    print("="*70)
    print("GAT RIGID BODY-FLUID INTERACTION VISUALIZATION")
    print("="*70)

    # Load GAT model
    print(f"\nLoading GAT model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint and 'model' in checkpoint['config']:
        config = checkpoint['config']['model']
    else:
        config = {
            'input_node_dim': 7,
            'input_edge_dim': 3,
            'hidden_dim': 64,
            'output_dim': 3,
            'num_layers': 5,
            'num_heads': 4,
            'dropout': 0.1
        }

    model = GraphAttentionNet(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"GAT Model loaded successfully ({model.count_parameters():,} parameters)")

    # Load scenario data
    print(f"\nLoading scenario data from {data_scenario_path}...")
    data_path = Path(data_scenario_path)
    frame_files = sorted(data_path.glob("frame_*.npz"))

    if len(frame_files) == 0:
        raise ValueError(f"No frame files found in {data_scenario_path}")

    # Load initial frame
    data_0 = np.load(frame_files[0])
    all_positions = data_0['positions']
    all_velocities = data_0['velocities']
    particle_types = data_0['particle_types']
    n_fluid = data_0['n_fluid']

    # Separate fluid and boundary
    fluid_pos_init = all_positions[:n_fluid]
    fluid_vel_init = all_velocities[:n_fluid]

    print(f"Initial fluid particles: {n_fluid}")
    print(f"Initial boundary particles: {len(all_positions) - n_fluid}")

    # Load boundary particle trajectories
    boundary_positions_over_time = []
    boundary_velocities_over_time = []

    for i in range(min(num_steps, len(frame_files))):
        data = np.load(frame_files[i])
        n_f = data['n_fluid']
        boundary_positions_over_time.append(data['positions'][n_f:])
        boundary_velocities_over_time.append(data['velocities'][n_f:])

    # Repeat last boundary state if needed
    while len(boundary_positions_over_time) < num_steps:
        boundary_positions_over_time.append(boundary_positions_over_time[-1])
        boundary_velocities_over_time.append(boundary_velocities_over_time[-1])

    # Run GAT simulation
    print(f"\nRunning GAT rollout simulation for {num_steps} steps...")
    fluid_history, _, boundary_history = rollout_simulation_with_rigid_body_gat(
        model,
        fluid_pos_init,
        fluid_vel_init,
        boundary_positions_over_time,
        boundary_velocities_over_time,
        num_steps=num_steps,
        device=device
    )

    # Create animated GIF
    print(f"\nCreating animated GIF...")
    fig = plt.figure(figsize=(16, 12))

    # Create 4 subplots
    ax1 = fig.add_subplot(2, 2, 1)  # XY view
    ax2 = fig.add_subplot(2, 2, 2)  # XZ view
    ax3 = fig.add_subplot(2, 2, 3)  # YZ view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')  # 3D view

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        fluid_pos = fluid_history[frame]
        boundary_pos = boundary_history[frame]

        # Remap axes: [X, Y_old, Z_old] -> [X, Y=Z_old, Z=Y_old]
        # So that Z becomes the height axis (standard Z-up convention)
        fluid_pos_remapped = np.column_stack([fluid_pos[:, 0], fluid_pos[:, 2], fluid_pos[:, 1]])
        if len(boundary_pos) > 0:
            boundary_pos_remapped = np.column_stack([boundary_pos[:, 0], boundary_pos[:, 2], boundary_pos[:, 1]])
        else:
            boundary_pos_remapped = np.array([]).reshape(0, 3)

        # Top view (XY plane, looking down Z-axis)
        ax1.scatter(fluid_pos_remapped[:, 0], fluid_pos_remapped[:, 1], c='blue', s=10, alpha=0.6, label='Fluid (GAT)')
        if len(boundary_pos_remapped) > 0:
            ax1.scatter(boundary_pos_remapped[:, 0], boundary_pos_remapped[:, 1], c='red', s=15, marker='s', alpha=0.8, label='Rigid Body')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Top View (XY) - GAT - Frame {frame}/{len(fluid_history)-1}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Side view (XZ plane, looking along Y-axis)
        ax2.scatter(fluid_pos_remapped[:, 0], fluid_pos_remapped[:, 2], c='blue', s=10, alpha=0.6, label='Fluid (GAT)')
        if len(boundary_pos_remapped) > 0:
            ax2.scatter(boundary_pos_remapped[:, 0], boundary_pos_remapped[:, 2], c='red', s=15, marker='s', alpha=0.8, label='Rigid Body')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z (height)')
        ax2.set_title('Side View (XZ)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Side view (YZ plane, looking along X-axis)
        ax3.scatter(fluid_pos_remapped[:, 1], fluid_pos_remapped[:, 2], c='blue', s=10, alpha=0.6, label='Fluid (GAT)')
        if len(boundary_pos_remapped) > 0:
            ax3.scatter(boundary_pos_remapped[:, 1], boundary_pos_remapped[:, 2], c='red', s=15, marker='s', alpha=0.8, label='Rigid Body')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z (height)')
        ax3.set_title('Side View (YZ)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 3D view (Z-up convention)
        ax4.scatter(fluid_pos_remapped[:, 0], fluid_pos_remapped[:, 1], fluid_pos_remapped[:, 2],
                   c='blue', s=10, alpha=0.5, label='Fluid (GAT)')
        if len(boundary_pos_remapped) > 0:
            ax4.scatter(boundary_pos_remapped[:, 0], boundary_pos_remapped[:, 1], boundary_pos_remapped[:, 2],
                       c='red', s=20, marker='s', alpha=0.9, label='Rigid Body')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_zlim(0, 1)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z (height)')
        ax4.set_title('3D View (GAT)')
        # Better viewing angle: elev=25 (looking down from above), azim rotates
        ax4.view_init(elev=25, azim=45 + frame * 2)
        ax4.legend()

        plt.tight_layout()

    # Create animation
    from matplotlib.animation import PillowWriter
    anim = FuncAnimation(fig, update, frames=len(fluid_history), interval=100)

    # Save as GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)

    print(f"\n[SAVED] GAT GIF animation: {output_path}")
    print(f"  Frames: {len(fluid_history)}")
    print(f"  Duration: {len(fluid_history) / 10:.1f}s")
    print(f"  Fluid particles: {len(fluid_pos_init)}")
    print(f"  Boundary particles: {len(boundary_positions_over_time[0])}")

    plt.close()

    print("\n" + "="*70)
    print("GAT VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/gat_rigid_body_sph_best.pt"

    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    else:
        # Find first rigid body scenario
        data_root = Path("datasets/flip_simulations")
        scenarios = sorted([d for d in data_root.iterdir() if d.is_dir() and 'falling_ball' in d.name])
        if scenarios:
            data_path = str(scenarios[0])
        else:
            print("No scenario data found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"visualizations/gat_interaction_{timestamp}.gif"

    try:
        create_gat_rigid_body_gif(
            model_path=model_path,
            data_scenario_path=data_path,
            output_path=output_path,
            num_steps=60
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to train the GAT model first using train_gat_rigid_bodies.py")
        sys.exit(1)
