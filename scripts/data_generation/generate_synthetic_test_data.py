"""
Generate simple synthetic test data for GNN training
This allows us to test the GNN pipeline while we fix the Blender/SPH issues
"""

import numpy as np
from pathlib import Path

def create_falling_sphere_scenario(output_dir, scenario_idx, num_frames=100, num_fluid_particles=1000):
    """
    Create synthetic data: water pool + falling sphere

    - Fluid particles: Stay at bottom (simulating settled water)
    - Rigid sphere: Falls from above with simple physics

    Args:
        output_dir: Output directory
        scenario_idx: Scenario index
        num_frames: Number of frames
        num_fluid_particles: Number of fluid particles
    """

    output_path = Path(output_dir) / f"synthetic_scenario_{scenario_idx:03d}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"SYNTHETIC DATA GENERATION - Scenario {scenario_idx}")
    print(f"=" * 70)

    # Domain
    domain_size = np.array([1.0, 1.0, 1.0])

    # Initial fluid particles (water pool at bottom)
    np.random.seed(scenario_idx)
    fluid_pos = np.random.uniform([0.1, 0.1, 0.1], [0.9, 0.9, 0.3], (num_fluid_particles, 3)).astype(np.float32)

    # Rigid sphere parameters
    sphere_radius = 0.1
    sphere_mass = 1.0
    initial_height = 0.8 + scenario_idx * 0.05  # Vary initial height
    sphere_pos_init = np.array([0.5, 0.5, initial_height], dtype=np.float32)
    sphere_vel_init = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Physics parameters
    dt = 0.01
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

    # Generate frames
    print(f"  Fluid particles: {num_fluid_particles}")
    print(f"  Sphere initial height: {initial_height:.3f}")
    print(f"  Generating {num_frames} frames...")

    sphere_pos = sphere_pos_init.copy()
    sphere_vel = sphere_vel_init.copy()

    for frame in range(num_frames):
        # Simple Euler integration for sphere
        sphere_vel += gravity * dt
        sphere_pos += sphere_vel * dt

        # Ground collision (Z = sphere_radius)
        if sphere_pos[2] < sphere_radius:
            sphere_pos[2] = sphere_radius
            sphere_vel[2] = -sphere_vel[2] * 0.5  # Bounce with damping

        # Fluid particles: add small random motion to simulate water movement
        fluid_vel = np.random.uniform(-0.02, 0.02, (num_fluid_particles, 3)).astype(np.float32)
        fluid_pos_current = fluid_pos + np.random.uniform(-0.005, 0.005, (num_fluid_particles, 3)).astype(np.float32)

        # Keep fluid particles in bounds
        fluid_pos_current = np.clip(fluid_pos_current, [0.05, 0.05, 0.05], [0.95, 0.95, 0.4])

        # Splash effect: push particles away from sphere when it enters water
        if sphere_pos[2] < 0.3:  # Sphere in water
            dist_to_sphere = np.linalg.norm(fluid_pos_current - sphere_pos, axis=1, keepdims=True)
            splash_mask = dist_to_sphere < (sphere_radius * 2)

            if splash_mask.any():
                # Push particles away
                push_dir = fluid_pos_current - sphere_pos
                push_dir = push_dir / (np.linalg.norm(push_dir, axis=1, keepdims=True) + 1e-6)
                fluid_vel[splash_mask.flatten()] += push_dir[splash_mask.flatten()] * 0.5

        # Save frame
        frame_file = output_path / f"frame_{frame:04d}.npz"

        np.savez_compressed(
            frame_file,
            positions=fluid_pos_current,
            velocities=fluid_vel,
            n_particles=num_fluid_particles,
            time=frame * dt,
            has_rigid_body=True,
            rigid_position=sphere_pos.copy(),
            rigid_velocity=sphere_vel.copy(),
            rigid_rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity quaternion
            rigid_angular_velocity=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )

        if (frame + 1) % 20 == 0:
            print(f"  Frame {frame + 1}/{num_frames} - Sphere Z: {sphere_pos[2]:.3f}, Vel: {sphere_vel[2]:.3f}")

    print(f"=" * 70)
    print(f"SCENARIO {scenario_idx} COMPLETE")
    print(f"  Output: {output_path}")
    print(f"=" * 70)
    print()

def generate_all_scenarios(output_dir, num_scenarios=25, num_frames=80):
    """
    Generate multiple test scenarios with variations

    Args:
        output_dir: Output directory
        num_scenarios: Number of scenarios
        num_frames: Frames per scenario
    """

    print("=" * 70)
    print("GENERATING SYNTHETIC TEST DATA")
    print("=" * 70)
    print()

    for idx in range(num_scenarios):
        # Vary parameters across scenarios
        num_particles = 800 + idx * 20  # 800-1280 particles
        create_falling_sphere_scenario(output_dir, idx, num_frames, num_particles)

    print("=" * 70)
    print(f"ALL {num_scenarios} SCENARIOS GENERATED")
    print(f"  Total frames: {num_scenarios * num_frames}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    output_dir = Path("datasets/synthetic_test_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 25 test scenarios
    generate_all_scenarios(output_dir, num_scenarios=25, num_frames=80)
