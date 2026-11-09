"""
Test single pySPlisHSPlasH simulation with data export
"""

import pysplishsplash as sph
import numpy as np
from pathlib import Path

# Paths
scene_file = Path("datasets/sph_scenes/sph_falling_ball_000.json").absolute()
output_dir = Path("datasets/sph_test_output/sph_falling_ball_000")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Scene: {scene_file}")
print(f"Output: {output_dir}")
print()

# Initialize simulator
print("Initializing simulator...")
base = sph.Exec.SimulatorBase()
base.init(sceneFile=str(scene_file), useCache=False, initialPause=False, useGui=False, outputDir=str(output_dir))

# Get simulation components
sim = sph.Simulation.getCurrent()
time_manager = sph.TimeManager.getCurrent()

print(f"Fluid models: {sim.numberOfFluidModels()}")
print(f"Boundary models: {sim.numberOfBoundaryModels()}")

# Get fluid model
fluid_model = sim.getFluidModel(0)
print(f"Initial particles: {fluid_model.numActiveParticles()}")
print()

# Simulation loop
frame_count = 0
export_interval = 0.01  # 100 FPS
next_export_time = 0.0
max_time = 0.08  # Only 8 frames for testing

print("Running simulation...")

while time_manager.getTime() < max_time:
    # Simulation step
    base.timeStep()

    current_time = time_manager.getTime()

    # Export frame
    if current_time >= next_export_time:
        # Get particle data
        positions = fluid_model.getPositions()
        velocities = fluid_model.getVelocities()
        n_particles = fluid_model.numActiveParticles()

        # Convert to numpy arrays (pySPlisHSPlasH stores as flat array)
        pos_array = np.array([[positions[3*i], positions[3*i+1], positions[3*i+2]]
                              for i in range(n_particles)])
        vel_array = np.array([[velocities[3*i], velocities[3*i+1], velocities[3*i+2]]
                              for i in range(n_particles)])

        # Particle types (all fluid for now)
        particle_types = np.zeros(n_particles, dtype=np.int32)

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

        print(f"Frame {frame_count}: t={current_time:.3f}s, particles={n_particles}, "
              f"Y_mean={pos_array[:, 1].mean():.3f}, Y_range=[{pos_array[:, 1].min():.3f}, {pos_array[:, 1].max():.3f}]")

        frame_count += 1
        next_export_time += export_interval

# Cleanup
base.cleanup()

print()
print(f"Test complete! Exported {frame_count} frames")
print(f"Output: {output_dir}")
