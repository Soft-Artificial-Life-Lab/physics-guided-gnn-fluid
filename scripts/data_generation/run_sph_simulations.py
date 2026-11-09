"""
Run pySPlisHSPlasH simulations and export particle data to .npz format
"""

import pysplishsplash as sph
import numpy as np
from pathlib import Path
import json
import sys

def run_simulation_and_export(scene_file, output_dir):
    """
    Run SPH simulation using pySPlisHSPlasH and export particle data

    Args:
        scene_file: Path to JSON scene file
        output_dir: Directory to save exported .npz files
    """

    print(f"\n{'='*70}")
    print(f"Running: {scene_file.name}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize simulator
    base = sph.Exec.SimulatorBase()
    base.init(sceneFile=str(scene_file), useCache=False, initialPause=False, useGui=False)

    # Get simulation components
    sim = sph.Simulation.getCurrent()
    time_manager = sph.TimeManager.getCurrent()

    # Get fluid model
    fluid_model = sim.getFluidModel(0)

    # Simulation loop
    frame_count = 0
    export_interval = 0.01  # Export every 0.01 seconds (100 FPS)
    next_export_time = 0.0

    print(f"Starting simulation...")
    print(f"Target: 80 frames at {1.0/export_interval:.0f} FPS")

    while time_manager.getTime() < 0.8:  # 0.8 seconds = 80 frames at 100 FPS
        # Simulation step
        base.timeStep()

        current_time = time_manager.getTime()

        # Export frame
        if current_time >= next_export_time:
            # Get particle data
            positions = fluid_model.getPositions()
            velocities = fluid_model.getVelocities()
            n_particles = fluid_model.numActiveParticles()

            # Get rigid body data
            rigid_bodies = []
            for i in range(sim.numberOfBoundaryModels()):
                bm = sim.getBoundaryModel(i)
                rb = bm.getRigidBodyObject()
                if rb and rb.isDynamic():
                    pos = rb.getPosition()
                    vel = rb.getVelocity()
                    rigid_bodies.append({
                        'position': np.array([pos[0], pos[1], pos[2]]),
                        'velocity': np.array([vel[0], vel[1], vel[2]])
                    })

            # Convert to numpy arrays
            pos_array = np.array([[positions[3*i], positions[3*i+1], positions[3*i+2]]
                                  for i in range(n_particles)])
            vel_array = np.array([[velocities[3*i], velocities[3*i+1], velocities[3*i+2]]
                                  for i in range(n_particles)])

            # Create particle types (0 = fluid, 2 = rigid boundary)
            particle_types = np.zeros(n_particles, dtype=np.int32)

            # Add rigid body particles if any
            if rigid_bodies:
                # For now, save rigid body info separately
                # Later we'll add rigid body particles to match our data format
                pass

            # Save to .npz
            frame_file = output_dir / f"frame_{frame_count:04d}.npz"
            np.savez(
                frame_file,
                positions=pos_array,
                velocities=vel_array,
                particle_types=particle_types,
                n_fluid=n_particles,
                time=current_time
            )

            if frame_count % 10 == 0:
                print(f"  Frame {frame_count:3d} | t={current_time:.3f}s | "
                      f"Particles={n_particles} | "
                      f"Y={pos_array[:, 1].mean():.3f}")

            frame_count += 1
            next_export_time += export_interval

    # Cleanup
    base.cleanup()

    print(f"\n[OK] Exported {frame_count} frames to {output_dir}")
    return frame_count

def main():
    print("=" * 70)
    print("PYSPLISHSPLASH SPH SIMULATION")
    print("Running simulations and exporting particle data")
    print("=" * 70)

    # Paths
    scene_dir = Path("datasets/sph_scenes")
    output_base = Path("datasets/sph_simulations")

    # Get all scene files
    scene_files = sorted(scene_dir.glob("*.json"))

    if not scene_files:
        print(f"ERROR: No scene files found in {scene_dir}")
        return

    print(f"\nFound {len(scene_files)} scene files")
    print()

    # Run each simulation
    total_frames = 0
    completed = 0

    for i, scene_file in enumerate(scene_files, 1):
        # Create output directory based on scene name
        scene_name = scene_file.stem  # e.g., "sph_falling_ball_000"
        output_dir = output_base / scene_name

        try:
            frames = run_simulation_and_export(scene_file, output_dir)
            total_frames += frames
            completed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to run {scene_file.name}: {e}")
            continue

    print()
    print("=" * 70)
    print(f"SIMULATION COMPLETE")
    print(f"  Completed: {completed}/{len(scene_files)} scenarios")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_base}")
    print("=" * 70)

if __name__ == "__main__":
    main()
