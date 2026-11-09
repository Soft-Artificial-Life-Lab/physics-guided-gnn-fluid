"""
Visualize synthetic test data with 3D perspective view
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def visualize_scenario_3d(scenario_dir, output_path, max_frames=80):
    """
    Create 3D rotating perspective visualization

    Args:
        scenario_dir: Directory containing frame NPZ files
        output_path: Path to save GIF
        max_frames: Maximum number of frames to visualize
    """

    scenario_path = Path(scenario_dir)
    frame_files = sorted(scenario_path.glob("frame_*.npz"))[:max_frames]

    if len(frame_files) == 0:
        print(f"No frames found in {scenario_dir}")
        return

    print(f"Visualizing {scenario_path.name}: {len(frame_files)} frames")

    # Load first frame to get info
    data = np.load(frame_files[0])

    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()

        # Load frame data
        data = np.load(frame_files[frame_idx])
        positions = data['positions']

        # Fluid particles
        ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
                  c='blue', s=20, alpha=0.6, label='Fluid')

        # Rigid body (sphere)
        if data['has_rigid_body']:
            rb_pos = data['rigid_position']

            # Draw sphere as a point (large)
            ax.scatter([rb_pos[0]], [rb_pos[2]], [rb_pos[1]],
                      c='red', s=200, alpha=0.8, marker='o', label='Rigid Sphere')

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Z (Height)')
        ax.set_zlabel('Y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # Rotate view
        azim = (frame_idx * 360 / len(frame_files)) % 360
        ax.view_init(elev=25, azim=azim)

        ax.set_title(f"{scenario_path.name}\nFrame {frame_idx}/{len(frame_files)}, Time: {data['time']:.2f}s")
        ax.legend(loc='upper right')

        return ax,

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frame_files), interval=50, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"  Saved: {output_path}")

def visualize_all_scenarios(data_dir, output_dir, num_scenarios=5):
    """
    Visualize multiple scenarios

    Args:
        data_dir: Directory containing scenario folders
        output_dir: Directory to save visualizations
        num_scenarios: Number of scenarios to visualize
    """

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SYNTHETIC DATA VISUALIZATION")
    print("=" * 70)

    scenario_dirs = sorted(data_path.glob("synthetic_scenario_*"))[:num_scenarios]

    for idx, scenario_dir in enumerate(scenario_dirs):
        output_file = output_path / f"synthetic_{scenario_dir.name}.gif"
        visualize_scenario_3d(scenario_dir, output_file)

    print()
    print("=" * 70)
    print(f"VISUALIZATION COMPLETE")
    print(f"  Generated {len(scenario_dirs)} GIFs")
    print(f"  Output: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    visualize_all_scenarios(
        data_dir="datasets/synthetic_test_data",
        output_dir="visualizations/synthetic_test_data",
        num_scenarios=5  # First 5 scenarios
    )
