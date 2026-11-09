"""
Generate SPH simulations using pySPlisHSPlasH Python API directly
Avoid JSON scene files to prevent segfault
"""

import numpy as np
from pathlib import Path
import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scene

def create_fluid_block(min_corner, max_corner, particle_radius):
    """
    Create fluid particles in a box

    Args:
        min_corner: [x, y, z] minimum corner
        max_corner: [x, y, z] maximum corner
        particle_radius: SPH particle radius

    Returns:
        numpy array of particle positions (N x 3)
    """

    spacing = 2.0 * particle_radius
    x_range = np.arange(min_corner[0], max_corner[0], spacing)
    y_range = np.arange(min_corner[1], max_corner[1], spacing)
    z_range = np.arange(min_corner[2], max_corner[2], spacing)

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    return positions.astype(np.float32)

def simulate_sph_scenario(scenario_idx, output_dir, num_steps=100):
    """
    Run SPH simulation using Python API

    Args:
        scenario_idx: Scenario index (0-24)
        output_dir: Directory to save results
        num_steps: Number of simulation steps
    """

    print(f"=" * 70)
    print(f"SPH Simulation - Scenario {scenario_idx}")
    print(f"=" * 70)

    # Simulation parameters
    particle_radius = 0.025
    dt = 0.005
    domain_size = (1.0, 1.0, 1.0)

    # Initialize simulator
    base = sph.Exec.SimulatorBase()
    base.init(useGui=False, initialPause=False)

    # Set simulation parameters
    sim = sph.Simulation.getCurrent()
    sim.setParticleRadius(particle_radius)

    # Time step
    time_manager = sph.TimeManager.getCurrent()
    time_manager.setTimeStepSize(dt)

    # Create fluid model
    fluid_model = sim.addFluidModel("fluid", 0, None, None, 1000.0)

    print(f"Initialized simulator")
    print(f"  Particle radius: {particle_radius}")
    print(f"  Time step: {dt}")
    print(f"  Domain: {domain_size}")

    # Create fluid particles
    scenario_type = scenario_idx % 5
    fluid_positions = None

    if scenario_type == 0:  # Dam break
        fluid_positions = create_fluid_block(
            [0.1, 0.1, 0.1],
            [0.4, 0.6, 0.4],
            particle_radius
        )
    elif scenario_type == 1:  # Water column
        fluid_positions = create_fluid_block(
            [0.35, 0.1, 0.35],
            [0.65, 0.7, 0.65],
            particle_radius
        )
    elif scenario_type == 2:  # Wave
        fluid_positions = create_fluid_block(
            [0.1, 0.1, 0.1],
            [0.9, 0.35, 0.5],
            particle_radius
        )
    elif scenario_type == 3:  # Splash
        fluid_positions = create_fluid_block(
            [0.3, 0.5, 0.3],
            [0.7, 0.8, 0.7],
            particle_radius
        )
    else:  # scenario_type == 4: Double column
        pos1 = create_fluid_block([0.2, 0.1, 0.2], [0.4, 0.6, 0.4], particle_radius)
        pos2 = create_fluid_block([0.6, 0.1, 0.6], [0.8, 0.6, 0.8], particle_radius)
        fluid_positions = np.vstack([pos1, pos2])

    num_particles = len(fluid_positions)
    print(f"  Fluid particles: {num_particles}")

    # Add particles to fluid model
    fluid_model.resize(num_particles)
    for i in range(num_particles):
        fluid_model.setPosition(i, fluid_positions[i])
        fluid_model.setVelocity(i, [0.0, 0.0, 0.0])

    # Add rigid body (if scenario allows)
    rigid_body_idx = None
    if scenario_type in [0, 1, 3]:  # Add rigid body for some scenarios
        # Determine rigid body parameters based on scenario
        if scenario_type == 0:  # Falling sphere for dam break
            geom_file = "datasets/geometries/sphere_medium.obj"
            initial_pos = [0.7, 0.5, 0.5]
            initial_vel = [0.0, 0.0, 0.0]
            density = 500.0  # Lighter than water
        elif scenario_type == 1:  # Falling box for water column
            geom_file = "datasets/geometries/box_medium.obj"
            initial_pos = [0.5, 0.7, 0.5]
            initial_vel = [0.0, 0.0, 0.0]
            density = 1500.0  # Heavier than water
        else:  # scenario_type == 3: Floating cylinder
            geom_file = "datasets/geometries/cylinder_medium.obj"
            initial_pos = [0.5, 0.3, 0.5]
            initial_vel = [0.0, -0.5, 0.0]
            density = 800.0  # Slightly lighter than water

        # Try to add rigid body
        try:
            rb_data = Scene.RigidBodyData()
            rb_data.geometryFile = geom_file
            rb_data.translation = initial_pos
            rb_data.velocity = initial_vel
            rb_data.isDynamic = True
            rb_data.density = density
            rb_data.collisionObjectType = Scene.CollisionObjectTypes.RigidBody

            rigid_body_idx = sim.addRigidBody(rb_data)
            print(f"  Rigid body added: {geom_file}")
            print(f"    Position: {initial_pos}, Density: {density}")
        except Exception as e:
            print(f"  Warning: Could not add rigid body: {e}")

    # Setup output
    output_path = Path(output_dir) / f"sph_scenario_{scenario_idx:03d}"
    output_path.mkdir(parents=True, exist_ok=True)

    all_positions = []
    all_velocities = []
    all_rigid_states = []

    # Run simulation
    print(f"\\nRunning simulation for {num_steps} steps...")

    for step in range(num_steps):
        # Step simulation
        base.timeStep()

        # Extract fluid state
        positions = []
        velocities = []

        for i in range(fluid_model.numActiveParticles()):
            pos = fluid_model.getPosition(i)
            vel = fluid_model.getVelocity(i)
            positions.append(pos)
            velocities.append(vel)

        all_positions.append(np.array(positions, dtype=np.float32))
        all_velocities.append(np.array(velocities, dtype=np.float32))

        # Extract rigid body state
        if rigid_body_idx is not None:
            try:
                rb = sim.getRigidBody(rigid_body_idx)
                rb_pos = rb.getPosition()
                rb_vel = rb.getVelocity()
                rb_rot = rb.getRotation()  # Quaternion
                rb_angvel = rb.getAngularVelocity()

                all_rigid_states.append({
                    'position': np.array(rb_pos, dtype=np.float32),
                    'velocity': np.array(rb_vel, dtype=np.float32),
                    'rotation': np.array(rb_rot, dtype=np.float32),
                    'angular_velocity': np.array(rb_angvel, dtype=np.float32)
                })
            except Exception as e:
                print(f"Warning at step {step}: Could not extract rigid body: {e}")
                all_rigid_states.append(None)
        else:
            all_rigid_states.append(None)

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps} - Particles: {len(positions)}")

    # Save results
    print(f"\\nSaving results to {output_path}")

    for step in range(num_steps):
        frame_file = output_path / f"frame_{step:04d}.npz"

        save_dict = {
            'positions': all_positions[step],
            'velocities': all_velocities[step],
            'n_particles': len(all_positions[step]),
            'time': step * dt,
            'has_rigid_body': rigid_body_idx is not None
        }

        if all_rigid_states[step] is not None:
            save_dict.update({
                'rigid_position': all_rigid_states[step]['position'],
                'rigid_velocity': all_rigid_states[step]['velocity'],
                'rigid_rotation': all_rigid_states[step]['rotation'],
                'rigid_angular_velocity': all_rigid_states[step]['angular_velocity']
            })

        np.savez_compressed(frame_file, **save_dict)

    # Cleanup
    base.cleanup()

    print(f"=" * 70)
    print(f"SCENARIO {scenario_idx} COMPLETE")
    print(f"  Saved {num_steps} frames to {output_path}")
    print(f"=" * 70)
    print()

if __name__ == "__main__":
    # Test with single scenario first
    output_dir = Path("datasets/sph_python_api")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scenario 0 (dam break with falling sphere)
    simulate_sph_scenario(0, output_dir, num_steps=100)
