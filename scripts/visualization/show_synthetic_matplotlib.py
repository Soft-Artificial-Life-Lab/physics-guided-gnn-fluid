"""
Visualize synthetic data using matplotlib (show inline, not save to GIF)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def visualize_frame_3d(scenario_dir, frame_idx=0, save_path=None, view_angle=None):
    """
    Visualize a single frame in 3D using matplotlib

    Args:
        scenario_dir: Directory containing frame NPZ files
        frame_idx: Frame index to visualize
        save_path: If provided, save to file instead of showing
        view_angle: Tuple (elevation, azimuth) for camera view
    """

    scenario_path = Path(scenario_dir)
    frame_files = sorted(scenario_path.glob("frame_*.npz"))

    if frame_idx >= len(frame_files):
        print(f"Frame {frame_idx} not found (only {len(frame_files)} frames)")
        return

    # Load frame data
    data = np.load(frame_files[frame_idx])
    positions = data['positions']

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot fluid particles (blue)
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
              c='blue', s=30, alpha=0.6, label='Fluid Particles')

    # Plot rigid body (red sphere)
    if data['has_rigid_body']:
        rb_pos = data['rigid_position']
        ax.scatter([rb_pos[0]], [rb_pos[2]], [rb_pos[1]],
                  c='red', s=300, alpha=0.9, marker='o',
                  edgecolors='darkred', linewidths=2,
                  label='Rigid Sphere')

    # Set labels and limits
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z (Height)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Y', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Set view angle
    if view_angle:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    else:
        ax.view_init(elev=25, azim=45)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Title with info
    title = f"{scenario_path.name}\n"
    title += f"Frame {frame_idx}/{len(frame_files)-1}, Time: {data['time']:.2f}s\n"
    title += f"Particles: {data['n_particles']}"

    if data['has_rigid_body']:
        title += f", Sphere Z: {rb_pos[2]:.3f}m"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_multi_frame_comparison(scenario_dir, frame_indices=[0, 20, 40, 60], save_path=None):
    """
    Visualize multiple frames side-by-side

    Args:
        scenario_dir: Directory containing frame NPZ files
        frame_indices: List of frame indices to show
        save_path: If provided, save to file
    """

    scenario_path = Path(scenario_dir)
    frame_files = sorted(scenario_path.glob("frame_*.npz"))

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5))

    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(frame_files):
            continue

        # Load frame data
        data = np.load(frame_files[frame_idx])
        positions = data['positions']

        # Create subplot
        ax = fig.add_subplot(1, len(frame_indices), idx+1, projection='3d')

        # Plot fluid particles
        ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
                  c='blue', s=20, alpha=0.6)

        # Plot rigid body
        if data['has_rigid_body']:
            rb_pos = data['rigid_position']
            ax.scatter([rb_pos[0]], [rb_pos[2]], [rb_pos[1]],
                      c='red', s=200, alpha=0.9, marker='o',
                      edgecolors='darkred', linewidths=2)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.view_init(elev=25, azim=45)

        # Title
        title = f"Frame {frame_idx}\nT={data['time']:.2f}s"
        if data['has_rigid_body']:
            title += f"\nZ={rb_pos[2]:.3f}m"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{scenario_path.name} - Time Evolution",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_trajectory(scenario_dir, save_path=None):
    """
    Visualize particle trajectories and rigid body path

    Args:
        scenario_dir: Directory containing frame NPZ files
        save_path: If provided, save to file
    """

    scenario_path = Path(scenario_dir)
    frame_files = sorted(scenario_path.glob("frame_*.npz"))

    # Collect rigid body positions
    rb_positions = []
    times = []

    for frame_file in frame_files[::4]:  # Every 4th frame
        data = np.load(frame_file)
        if data['has_rigid_body']:
            rb_positions.append(data['rigid_position'])
            times.append(data['time'])

    rb_positions = np.array(rb_positions)

    # Create figure
    fig = plt.figure(figsize=(16, 6))

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(rb_positions[:, 0], rb_positions[:, 2], rb_positions[:, 1],
            'r-', linewidth=2, label='Sphere Path')
    ax1.scatter(rb_positions[0, 0], rb_positions[0, 2], rb_positions[0, 1],
               c='green', s=100, marker='o', label='Start')
    ax1.scatter(rb_positions[-1, 0], rb_positions[-1, 2], rb_positions[-1, 1],
               c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (Height)')
    ax1.set_zlabel('Y')
    ax1.set_title('3D Trajectory', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Height vs time
    ax2 = fig.add_subplot(132)
    ax2.plot(times, rb_positions[:, 2], 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Height (m)', fontweight='bold')
    ax2.set_title('Height vs Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Velocity analysis (approximate)
    velocities = np.diff(rb_positions, axis=0) / np.diff(times)[:, None]
    speed = np.linalg.norm(velocities, axis=1)

    ax3 = fig.add_subplot(133)
    ax3.plot(times[:-1], speed, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Speed (m/s)', fontweight='bold')
    ax3.set_title('Speed vs Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{scenario_path.name} - Rigid Body Analysis",
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    # Example usage - visualize first scenario
    scenario_dir = "datasets/synthetic_test_data/synthetic_scenario_000"

    print("Generating matplotlib visualizations...")
    print()

    # 1. Single frame from different angles
    print("1. Single frame visualization (Frame 40)...")
    visualize_frame_3d(scenario_dir, frame_idx=40,
                      save_path="visualizations/matplotlib/single_frame.png",
                      view_angle=(25, 45))

    # 2. Multi-frame comparison
    print("2. Multi-frame comparison...")
    visualize_multi_frame_comparison(scenario_dir,
                                    frame_indices=[0, 20, 40, 60],
                                    save_path="visualizations/matplotlib/multi_frame.png")

    # 3. Trajectory analysis
    print("3. Trajectory analysis...")
    visualize_trajectory(scenario_dir,
                        save_path="visualizations/matplotlib/trajectory.png")

    print()
    print("All visualizations generated!")
    print("Check: visualizations/matplotlib/")
