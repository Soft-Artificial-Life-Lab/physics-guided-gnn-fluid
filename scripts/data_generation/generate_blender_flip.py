"""
Generate FLIP fluid simulations with rigid body interaction using Blender
Blender has stable FLIP implementation with good rigid body support
"""

import bpy
import numpy as np
import os
from pathlib import Path

def clear_scene():
    """Clear all objects from scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_domain(size=2.0):
    """
    Create fluid domain

    Args:
        size: Domain size (cubic)

    Returns:
        Domain object
    """

    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size/2))
    domain = bpy.context.active_object
    domain.name = "FluidDomain"

    # Add fluid domain modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

    # Domain settings
    domain_settings = domain.modifiers["Fluid"].domain_settings
    domain_settings.domain_type = 'LIQUID'
    domain_settings.resolution_max = 64  # Resolution
    domain_settings.use_adaptive_domain = False
    domain_settings.gravity = (0, 0, -9.81)

    # Time settings
    domain_settings.timesteps_max = 4
    domain_settings.timesteps_min = 1

    # Particle settings
    domain_settings.particle_radius = 0.025
    domain_settings.use_flip_particles = True

    return domain

def create_fluid_block(location, size):
    """
    Create fluid emitter block

    Args:
        location: (x, y, z) position
        size: (sx, sy, sz) size

    Returns:
        Fluid object
    """

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    fluid = bpy.context.active_object
    fluid.name = "Fluid"
    fluid.scale = size

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    fluid.modifiers["Fluid"].fluid_type = 'FLOW'

    # Flow settings
    flow_settings = fluid.modifiers["Fluid"].flow_settings
    flow_settings.flow_type = 'LIQUID'
    flow_settings.flow_behavior = 'GEOMETRY'
    flow_settings.use_initial_velocity = False

    return fluid

def create_rigid_obstacle(obj_type, location, size, velocity=(0, 0, 0), density=1000.0):
    """
    Create rigid body obstacle

    Args:
        obj_type: 'CUBE', 'UV_SPHERE', 'CYLINDER'
        location: (x, y, z) position
        size: scale or (sx, sy, sz)
        velocity: Initial velocity (x, y, z)
        density: Material density kg/m^3

    Returns:
        Rigid body object
    """

    # Create mesh
    if obj_type == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    elif obj_type == 'UV_SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location)
    elif obj_type == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=location)

    rigid_body = bpy.context.active_object
    rigid_body.name = f"RigidBody_{obj_type}"

    if isinstance(size, tuple):
        rigid_body.scale = size
    else:
        rigid_body.scale = (size, size, size)

    # Add fluid effector
    bpy.ops.object.modifier_add(type='FLUID')
    rigid_body.modifiers["Fluid"].fluid_type = 'EFFECTOR'
    rigid_body.modifiers["Fluid"].effector_settings.effector_type = 'COLLISION'

    # Add rigid body physics
    bpy.ops.rigidbody.object_add()
    rigid_body.rigid_body.type = 'ACTIVE'
    rigid_body.rigid_body.mass = density * (size ** 3 if not isinstance(size, tuple) else size[0] * size[1] * size[2])

    # Set initial velocity using keyframes (Blender 4.5 approach)
    if velocity != (0, 0, 0):
        bpy.context.scene.frame_set(1)
        rigid_body.location = location
        rigid_body.keyframe_insert(data_path="location", frame=1)

        # Set second keyframe with velocity applied
        dt = 1.0 / 24.0  # 24 fps
        new_location = (location[0] + velocity[0] * dt,
                       location[1] + velocity[1] * dt,
                       location[2] + velocity[2] * dt)
        rigid_body.location = new_location
        rigid_body.keyframe_insert(data_path="location", frame=2)

        # Reset to initial position
        rigid_body.location = location

    # Enable collision
    rigid_body.rigid_body.collision_shape = 'MESH'

    return rigid_body

def setup_scenario(scenario_idx, output_dir):
    """
    Setup specific scenario

    Args:
        scenario_idx: Scenario index (0-24)
        output_dir: Output directory

    Returns:
        output_path: Path to save simulation data
    """

    clear_scene()

    scenario_type = scenario_idx % 5

    # Create domain
    domain = create_domain(size=2.0)

    # Create scenario based on type
    if scenario_type == 0:
        # Dam break with falling sphere
        create_fluid_block(location=(-0.5, 0, 0.3), size=(0.4, 0.8, 0.6))
        create_rigid_obstacle('UV_SPHERE', location=(0.5, 0, 0.8), size=0.2,
                            velocity=(0, 0, 0), density=500.0)

    elif scenario_type == 1:
        # Water column with falling box
        create_fluid_block(location=(0, 0, 0.4), size=(0.5, 0.5, 0.8))
        create_rigid_obstacle('CUBE', location=(0, 0, 1.2), size=0.25,
                            velocity=(0, 0, 0), density=1500.0)

    elif scenario_type == 2:
        # Wave with floating cylinder
        create_fluid_block(location=(0, 0, 0.2), size=(1.6, 1.0, 0.4))
        create_rigid_obstacle('CYLINDER', location=(0, 0, 0.5), size=(0.15, 0.15, 0.4),
                            velocity=(0, 0, 0), density=800.0)

    elif scenario_type == 3:
        # Splash with dropping sphere
        create_fluid_block(location=(0, 0, 0.25), size=(1.2, 1.2, 0.5))
        create_rigid_obstacle('UV_SPHERE', location=(0, 0, 1.0), size=0.18,
                            velocity=(0, 0, -2.0), density=1200.0)

    else:  # scenario_type == 4
        # Double column with sinking box
        create_fluid_block(location=(-0.4, -0.4, 0.35), size=(0.4, 0.4, 0.7))
        create_fluid_block(location=(0.4, 0.4, 0.35), size=(0.4, 0.4, 0.7))
        create_rigid_obstacle('CUBE', location=(0, 0, 0.6), size=(0.2, 0.2, 0.15),
                            velocity=(0, 0, 0), density=2000.0)

    # Set output path
    output_path = Path(output_dir) / f"blender_flip_scenario_{scenario_idx:03d}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure cache
    domain.modifiers["Fluid"].domain_settings.cache_directory = str(output_path / "cache")

    return output_path, domain

def run_simulation(num_frames=100):
    """
    Run Blender fluid simulation

    Args:
        num_frames: Number of frames to simulate
    """

    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(1)

    # Bake simulation
    print(f"Baking fluid simulation for {num_frames} frames...")

    # Select domain for baking
    domain = bpy.data.objects.get("FluidDomain")
    if domain:
        bpy.context.view_layer.objects.active = domain
        domain.select_set(True)

        # Bake fluid
        bpy.ops.fluid.bake_data()

        print("Baking complete!")
    else:
        print("ERROR: No fluid domain found!")

def extract_particle_data(output_path, domain, num_frames=100):
    """
    Extract particle positions and velocities from Blender cache

    Args:
        output_path: Directory to save data
        domain: Domain object
        num_frames: Number of frames
    """

    print(f"Extracting particle data from cache...")

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)

        # Get particle system (FLIP particles)
        particle_systems = domain.particle_systems

        if len(particle_systems) == 0:
            print(f"  Frame {frame}: No particles")
            continue

        # Extract fluid particles
        particles = particle_systems[0].particles

        positions = []
        velocities = []

        for p in particles:
            positions.append([p.location.x, p.location.z, p.location.y])  # Swap Y and Z
            velocities.append([p.velocity.x, p.velocity.z, p.velocity.y])

        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)

        # Extract rigid body state
        rigid_bodies = [obj for obj in bpy.data.objects if obj.rigid_body and obj.rigid_body.type == 'ACTIVE']

        rigid_states = []
        for rb in rigid_bodies:
            # Calculate velocity from position difference
            if frame > 1:
                bpy.context.scene.frame_set(frame - 1)
                prev_pos = rb.location.copy()
                bpy.context.scene.frame_set(frame)
                curr_pos = rb.location.copy()
                velocity = [(curr_pos[i] - prev_pos[i]) * 24.0 for i in range(3)]  # 24 fps
            else:
                velocity = [0, 0, 0]

            rigid_states.append({
                'position': np.array([rb.location.x, rb.location.z, rb.location.y], dtype=np.float32),
                'velocity': np.array([velocity[0], velocity[2], velocity[1]], dtype=np.float32),
                'rotation': np.array(rb.rotation_quaternion, dtype=np.float32),
                'angular_velocity': np.array([0, 0, 0], dtype=np.float32)  # Simplified
            })

        # Save frame data
        frame_file = output_path / f"frame_{frame-1:04d}.npz"

        save_dict = {
            'positions': positions,
            'velocities': velocities,
            'n_particles': len(positions),
            'time': (frame - 1) * (1.0 / 24.0),  # Assuming 24 fps
            'has_rigid_body': len(rigid_bodies) > 0
        }

        if len(rigid_bodies) > 0:
            for idx, rb_state in enumerate(rigid_states):
                save_dict[f'rigid_{idx}_position'] = rb_state['position']
                save_dict[f'rigid_{idx}_velocity'] = rb_state['velocity']
                save_dict[f'rigid_{idx}_rotation'] = rb_state['rotation']
                save_dict[f'rigid_{idx}_angular_velocity'] = rb_state['angular_velocity']

        np.savez_compressed(frame_file, **save_dict)

        if (frame) % 20 == 0:
            print(f"  Frame {frame}/{num_frames} - Particles: {len(positions)}")

    print(f"Particle data extraction complete!")

def generate_scenario(scenario_idx, output_dir, num_frames=100):
    """
    Generate single scenario

    Args:
        scenario_idx: Scenario index
        output_dir: Output directory
        num_frames: Number of frames
    """

    print("=" * 70)
    print(f"BLENDER FLIP SIMULATION - Scenario {scenario_idx}")
    print("=" * 70)

    output_path, domain = setup_scenario(scenario_idx, output_dir)

    run_simulation(num_frames)

    extract_particle_data(output_path, domain, num_frames)

    print("=" * 70)
    print(f"SCENARIO {scenario_idx} COMPLETE")
    print(f"  Output: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    # Generate scenario 0
    output_dir = Path("datasets/blender_flip")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_scenario(0, output_dir, num_frames=80)
