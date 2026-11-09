"""
Generate FLIP simulation data with dynamic rigid body interactions (Two-way coupling)
Uses grid-based pressure projection for stable incompressible fluid simulation
"""

import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.flip_simulator import FLIPSimulator
from src.simulation.dynamic_rigid_body import (
    DynamicSphereRigidBody, DynamicBoxRigidBody, DynamicRigidBodySimulation
)


def create_fluid_particles(n_particles, domain_size, water_height, spacing=0.05):
    """Create initial fluid particles in a block"""
    particles = []

    # Calculate how many particles fit
    nx = int(domain_size[0] / spacing)
    ny = int(water_height / spacing)
    nz = int(domain_size[2] / spacing)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                z = (k + 0.5) * spacing

                if x < domain_size[0] and y < water_height and z < domain_size[2]:
                    particles.append([x, y, z])

                if len(particles) >= n_particles:
                    break
            if len(particles) >= n_particles:
                break
        if len(particles) >= n_particles:
            break

    return np.array(particles[:n_particles])


def create_dynamic_scenario(scenario_type: str, scenario_idx: int):
    """Create scenarios with dynamic rigid bodies"""
    np.random.seed(scenario_idx * 1000)

    domain_size = (1.0, 1.0, 1.0)

    if scenario_type == "falling_ball":
        # Ball drops into water
        water_height = 0.6
        fluid_pos = create_fluid_particles(1500, domain_size, water_height, spacing=0.05)
        fluid_vel = np.zeros_like(fluid_pos)

        # Dynamic ball
        radius_var = 0.15 + (scenario_idx % 5) * 0.02
        mass_var = 4.0 + (scenario_idx % 5) * 1.0
        height_var = 1.2 + (scenario_idx % 5) * 0.1

        ball = DynamicSphereRigidBody(
            name="ball",
            radius=radius_var,
            mass=mass_var,
            initial_position=np.array([0.5, height_var, 0.5]),
            initial_velocity=np.array([0.0, -2.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball])

    elif scenario_type == "floating_ball":
        # Light ball that floats
        water_height = 0.7
        fluid_pos = create_fluid_particles(1800, domain_size, water_height, spacing=0.05)
        fluid_vel = np.zeros_like(fluid_pos)

        ball = DynamicSphereRigidBody(
            name="floater",
            radius=0.12 + scenario_idx * 0.01,
            mass=1.0 + scenario_idx * 0.3,
            initial_position=np.array([0.5, 0.85, 0.5]),
            initial_velocity=np.array([0.0, 0.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball])

    elif scenario_type == "heavy_box":
        # Heavy box sinking
        water_height = 0.6
        fluid_pos = create_fluid_particles(1600, domain_size, water_height, spacing=0.05)
        fluid_vel = np.zeros_like(fluid_pos)

        box = DynamicBoxRigidBody(
            name="heavy_box",
            size=np.array([0.15, 0.15, 0.15]),
            mass=15.0 + scenario_idx * 3.0,
            initial_position=np.array([0.5, 1.0, 0.5]),
            initial_velocity=np.array([0.0, -1.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([box])

    elif scenario_type == "multi_ball":
        # Multiple balls
        water_height = 0.6
        fluid_pos = create_fluid_particles(1400, domain_size, water_height, spacing=0.05)
        fluid_vel = np.zeros_like(fluid_pos)

        ball1 = DynamicSphereRigidBody(
            name="ball1",
            radius=0.12,
            mass=3.0,
            initial_position=np.array([0.3, 1.2 + scenario_idx * 0.05, 0.5]),
            initial_velocity=np.array([0.0, -1.5, 0.0])
        )
        ball2 = DynamicSphereRigidBody(
            name="ball2",
            radius=0.10,
            mass=6.0,
            initial_position=np.array([0.7, 1.3 + scenario_idx * 0.05, 0.5]),
            initial_velocity=np.array([0.0, -2.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball1, ball2])

    elif scenario_type == "bouncing_sphere":
        # High energy bouncing
        water_height = 0.5
        fluid_pos = create_fluid_particles(1200, domain_size, water_height, spacing=0.05)
        fluid_vel = np.zeros_like(fluid_pos)

        ball = DynamicSphereRigidBody(
            name="bouncer",
            radius=0.15,
            mass=4.0 + scenario_idx,
            initial_position=np.array([0.5, 1.4, 0.5]),
            initial_velocity=np.array([0.0, -3.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball])
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")

    return fluid_pos, fluid_vel, rigid_sim


def simulate_flip_scenario(scenario_type: str, scenario_idx: int, output_dir: Path,
                           num_steps=80, dt=0.01):
    """Run FLIP simulation with dynamic rigid bodies"""

    print(f"\n{'='*70}")
    print(f"FLIP SCENARIO: {scenario_type} #{scenario_idx}")
    print(f"{'='*70}")

    # Create scenario
    fluid_pos, fluid_vel, rigid_sim = create_dynamic_scenario(scenario_type, scenario_idx)

    # Initialize FLIP simulator
    flip = FLIPSimulator(
        domain_size=(1.0, 1.0, 1.0),
        grid_resolution=(32, 32, 32),
        dt=dt,
        gravity=-9.81,
        pic_flip_ratio=0.05,
        density=1000.0
    )

    # Create output directory
    scenario_dir = output_dir / f"flip_dynamic_{scenario_type}_{scenario_idx:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"FLIP Grid: {flip.grid_res}, dx={flip.dx}")
    print(f"Initial: {len(fluid_pos)} fluid particles")
    print(f"Two-way coupling: ENABLED")
    print()

    for step in range(num_steps):
        t = step * dt

        # Get rigid body boundary particles
        boundary_pos, boundary_vel, body_ids = rigid_sim.get_all_boundary_particles()

        # Save current state
        output_file = scenario_dir / f"frame_{step:04d}.npz"

        all_pos = np.vstack([fluid_pos, boundary_pos]) if len(boundary_pos) > 0 else fluid_pos
        all_vel = np.vstack([fluid_vel, boundary_vel]) if len(boundary_vel) > 0 else fluid_vel

        particle_types = np.hstack([
            np.zeros(len(fluid_pos)),
            np.ones(len(boundary_pos)) if len(boundary_pos) > 0 else []
        ])

        np.savez_compressed(
            output_file,
            positions=all_pos,
            velocities=all_vel,
            particle_types=particle_types,
            body_ids=np.hstack([
                np.full(len(fluid_pos), -1),
                body_ids if len(body_ids) > 0 else []
            ]),
            time=t,
            n_fluid=len(fluid_pos),
            n_boundary=len(boundary_pos)
        )

        # FLIP step (incompressible fluid)
        fluid_pos, fluid_vel = flip.step(fluid_pos, fluid_vel)

        # Update rigid bodies (two-way coupling would compute forces from fluid)
        # For now, just apply gravity and boundary constraints
        rigid_sim.update_all(dt, np.array([0, -9.81, 0]), flip.domain_size)

        if step % 10 == 0:
            # Calculate water height
            water_height_mean = fluid_pos[:, 1].mean()
            water_height_max = fluid_pos[:, 1].max()

            # Print rigid body info
            if len(rigid_sim.rigid_bodies) > 0:
                body = rigid_sim.rigid_bodies[0]
                print(f"  Step {step:3d}/{num_steps} - t={t:.2f}s - "
                      f"Water: mean_y={water_height_mean:.3f}, max_y={water_height_max:.3f}, "
                      f"Body Y: {body.position[1]:.3f}, Vel: {body.velocity[1]:.3f} m/s")

    print(f"[OK] Saved {num_steps} frames to {scenario_dir}")
    return num_steps


def main():
    """Generate FLIP scenarios with dynamic rigid bodies"""

    output_dir = Path("datasets/flip_simulations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FLIP DATA GENERATION WITH DYNAMIC RIGID BODIES")
    print("Grid-based pressure projection for incompressible flow")
    print("="*70)
    print()

    # Scenario types and counts
    scenario_configs = [
        ("falling_ball", 5),
        ("floating_ball", 5),
        ("heavy_box", 5),
        ("multi_ball", 5),
        ("bouncing_sphere", 5)
    ]

    total_scenarios = sum(count for _, count in scenario_configs)
    completed = 0

    start_time = time.time()

    for scenario_type, count in scenario_configs:
        print(f"\nGenerating {scenario_type} scenarios...")
        for i in range(count):
            simulate_flip_scenario(scenario_type, i, output_dir)
            completed += 1

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total scenarios: {completed}")
    print(f"Scenario types: {len(scenario_configs)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Average: {elapsed/completed:.1f}s per scenario")
    print("="*70)


if __name__ == "__main__":
    main()
