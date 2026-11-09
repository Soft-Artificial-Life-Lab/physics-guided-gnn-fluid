"""
Generate SPH simulation data with moving rigid body interactions
"""

import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
import time
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.rigid_body import (
    RigidBodySimulation, SphereRigidBody, BoxRigidBody, CylinderRigidBody,
    linear_motion, oscillating_motion, circular_motion,
    oscillating_velocity, circular_velocity, static_position
)


class SPHSimulatorWithRigidBodies:
    """SPH simulator with rigid body boundary conditions"""

    def __init__(self,
                 domain_size=(1.0, 1.0, 1.0),
                 particle_mass=0.02,
                 rest_density=1000.0,
                 gas_constant=2000.0,
                 viscosity=0.1,
                 dt=0.01,
                 gravity=-9.81,
                 h=0.1):

        self.domain_size = np.array(domain_size)
        self.particle_mass = particle_mass
        self.rest_density = rest_density
        self.gas_constant = gas_constant
        self.viscosity = viscosity
        self.dt = dt
        self.gravity = np.array([0, gravity, 0])
        self.h = h  # Smoothing length
        self.h2 = h * h

        # Precompute kernel coefficients
        self.poly6_coeff = 315.0 / (64.0 * np.pi * h**9)
        self.spiky_grad_coeff = -45.0 / (np.pi * h**6)
        self.visc_lap_coeff = 45.0 / (np.pi * h**6)

    def poly6_kernel(self, r_squared):
        """Poly6 kernel for density"""
        if r_squared < self.h2:
            return self.poly6_coeff * (self.h2 - r_squared)**3
        return 0.0

    def spiky_gradient(self, r_vec):
        """Spiky gradient for pressure"""
        r = np.linalg.norm(r_vec)
        if r < self.h and r > 1e-6:
            return self.spiky_grad_coeff * (self.h - r)**2 * r_vec / r
        return np.zeros(3)

    def viscosity_laplacian(self, r):
        """Viscosity laplacian"""
        if r < self.h:
            return self.visc_lap_coeff * (self.h - r)
        return 0.0

    def compute_density(self, positions, is_fluid_mask):
        """
        Compute density for fluid particles

        Args:
            positions: All particle positions (fluid + boundary)
            is_fluid_mask: Boolean mask indicating fluid particles

        Returns:
            densities: Array of densities for ALL particles
        """
        n = len(positions)
        densities = np.zeros(n)

        tree = KDTree(positions)
        neighbors_list = tree.query_ball_tree(tree, self.h)

        for i in range(n):
            density = 0.0
            for j in neighbors_list[i]:
                r_vec = positions[i] - positions[j]
                r_squared = np.dot(r_vec, r_vec)
                density += self.particle_mass * self.poly6_kernel(r_squared)
            densities[i] = max(density, self.rest_density)  # Prevent zero density

        return densities

    def compute_forces(self, positions, velocities, densities, is_fluid_mask):
        """
        Compute forces on fluid particles
        Boundary particles exert forces on fluid but don't move

        Args:
            positions: All particle positions
            velocities: All particle velocities
            densities: All particle densities
            is_fluid_mask: Boolean mask for fluid particles

        Returns:
            forces: Forces on fluid particles only
        """
        n_fluid = np.sum(is_fluid_mask)
        fluid_indices = np.where(is_fluid_mask)[0]
        forces = np.zeros((n_fluid, 3))

        tree = KDTree(positions)

        for idx, i in enumerate(fluid_indices):
            pressure_force = np.zeros(3)
            viscosity_force = np.zeros(3)

            # Compute pressure from density
            pressure_i = self.gas_constant * (densities[i] - self.rest_density)

            # Find neighbors (both fluid and boundary)
            neighbors = tree.query_ball_point(positions[i], self.h)

            for j in neighbors:
                if i == j:
                    continue

                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)

                if r < self.h and r > 1e-6:
                    # Pressure force
                    pressure_j = self.gas_constant * (densities[j] - self.rest_density)
                    pressure_force += -self.particle_mass * (pressure_i + pressure_j) / (2 * densities[j]) * \
                                     self.spiky_gradient(r_vec)

                    # Viscosity force (only with fluid particles)
                    if is_fluid_mask[j]:
                        vel_diff = velocities[j] - velocities[i]
                        viscosity_force += self.viscosity * self.particle_mass * vel_diff / densities[j] * \
                                          self.viscosity_laplacian(r)

            # Total force
            forces[idx] = pressure_force + viscosity_force + self.gravity * densities[i]

        return forces

    def step(self, fluid_positions, fluid_velocities, boundary_positions, boundary_velocities):
        """
        Single simulation step with rigid boundaries

        Args:
            fluid_positions: (N_fluid, 3)
            fluid_velocities: (N_fluid, 3)
            boundary_positions: (N_boundary, 3)
            boundary_velocities: (N_boundary, 3)

        Returns:
            new_fluid_positions, new_fluid_velocities
        """
        # Combine fluid and boundary particles
        all_positions = np.vstack([fluid_positions, boundary_positions])
        all_velocities = np.vstack([fluid_velocities, boundary_velocities])

        n_fluid = len(fluid_positions)
        is_fluid = np.zeros(len(all_positions), dtype=bool)
        is_fluid[:n_fluid] = True

        # Compute densities for all particles
        densities = self.compute_density(all_positions, is_fluid)

        # Compute forces on fluid particles only
        forces = self.compute_forces(all_positions, all_velocities, densities, is_fluid)

        # Update fluid particles
        accelerations = forces / densities[:n_fluid, np.newaxis]
        new_velocities = fluid_velocities + accelerations * self.dt
        new_positions = fluid_positions + new_velocities * self.dt

        # Boundary conditions (domain walls)
        for dim in range(3):
            # Lower boundary
            mask = new_positions[:, dim] < 0
            new_positions[mask, dim] = 0
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

            # Upper boundary
            mask = new_positions[:, dim] > self.domain_size[dim]
            new_positions[mask, dim] = self.domain_size[dim]
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

        return new_positions, new_velocities


SCENARIO_SEED_OFFSET = {
    "falling_ball": 0, "paddle_wave": 1, "rotating_stirrer": 2,
    "moving_obstacle": 3, "double_paddle": 4, "sphere_bounce": 5,
}


def create_scenario_with_rigid_body(scenario_type: str, scenario_idx: int):
    """Create different scenarios with rigid body interactions"""

    np.random.seed(scenario_idx * 1000 + SCENARIO_SEED_OFFSET[scenario_type] * 100)

    if scenario_type == "falling_ball":
        # Ball drops into water
        n_particles = 800
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.4, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        ball = SphereRigidBody(
            name="ball",
            radius=0.15,
            position_fn=linear_motion([0.5, 1.2, 0.5], [0, -2.0, 0]),
            velocity_fn=lambda t: np.array([0, -2.0, 0])
        )
        rigid_sim = RigidBodySimulation([ball])

    elif scenario_type == "paddle_wave":
        # Oscillating paddle creates waves
        n_particles = 600
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.5, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        paddle = BoxRigidBody(
            name="paddle",
            size=(0.05, 0.3, 0.8),
            position_fn=oscillating_motion([0.2, 0.25, 0.5], [0.15, 0, 0], 1.0),
            velocity_fn=oscillating_velocity([0.15, 0, 0], 1.0)
        )
        rigid_sim = RigidBodySimulation([paddle])

    elif scenario_type == "rotating_stirrer":
        # Rotating cylinder stirs fluid
        n_particles = 700
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.6, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        stirrer = CylinderRigidBody(
            name="stirrer",
            radius=0.08,
            height=0.5,
            position_fn=circular_motion([0.5, 0.3, 0.5], 0.25, 2.0, 'y'),
            velocity_fn=circular_velocity(0.25, 2.0, 'y')
        )
        rigid_sim = RigidBodySimulation([stirrer])

    elif scenario_type == "moving_obstacle":
        # Box moves through fluid
        n_particles = 500
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.5, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        obstacle = BoxRigidBody(
            name="obstacle",
            size=(0.2, 0.2, 0.2),
            position_fn=linear_motion([0.1, 0.25, 0.5], [0.5, 0, 0]),
            velocity_fn=lambda t: np.array([0.5, 0, 0])
        )
        rigid_sim = RigidBodySimulation([obstacle])

    elif scenario_type == "double_paddle":
        # Two paddles oscillating
        n_particles = 600
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.5, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        paddle1 = BoxRigidBody(
            name="paddle1",
            size=(0.05, 0.3, 0.6),
            position_fn=oscillating_motion([0.3, 0.25, 0.5], [0, 0.15, 0], 1.5),
            velocity_fn=oscillating_velocity([0, 0.15, 0], 1.5)
        )
        paddle2 = BoxRigidBody(
            name="paddle2",
            size=(0.05, 0.3, 0.6),
            position_fn=oscillating_motion([0.7, 0.25, 0.5], [0, -0.15, 0], 1.5),
            velocity_fn=oscillating_velocity([0, -0.15, 0], 1.5)
        )
        rigid_sim = RigidBodySimulation([paddle1, paddle2])

    else:  # sphere_bounce
        # Sphere bounces vertically
        n_particles = 600
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.4, 1.0]
        fluid_velocities = np.zeros((n_particles, 3))

        ball = SphereRigidBody(
            name="ball",
            radius=0.12,
            position_fn=oscillating_motion([0.5, 0.6, 0.5], [0, 0.3, 0], 1.0),
            velocity_fn=oscillating_velocity([0, 0.3, 0], 1.0)
        )
        rigid_sim = RigidBodySimulation([ball])

    return fluid_positions, fluid_velocities, rigid_sim


def simulate_scenario(scenario_type: str, scenario_idx: int, output_dir: Path,
                     num_steps=100, dt=0.01):
    """Run simulation with rigid bodies and save data"""

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_type} #{scenario_idx}")
    print(f"{'='*60}")

    # Create scenario
    fluid_pos, fluid_vel, rigid_sim = create_scenario_with_rigid_body(scenario_type, scenario_idx)

    # Initialize simulator
    simulator = SPHSimulatorWithRigidBodies(dt=dt)

    # Create output directory
    scenario_dir = output_dir / f"sph_{scenario_type}_{scenario_idx:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Simulation loop
    print(f"Initial: {len(fluid_pos)} fluid particles")

    for step in range(num_steps):
        t = step * dt

        # Get rigid body particles at current time
        boundary_pos, boundary_vel, body_ids = rigid_sim.get_all_boundary_particles(t)

        # Save current state
        output_file = scenario_dir / f"frame_{step:04d}.npz"

        # Combine for saving (includes boundary info)
        all_pos = np.vstack([fluid_pos, boundary_pos]) if len(boundary_pos) > 0 else fluid_pos
        all_vel = np.vstack([fluid_vel, boundary_vel]) if len(boundary_vel) > 0 else fluid_vel

        particle_types = np.hstack([
            np.zeros(len(fluid_pos)),  # 0 = fluid
            np.ones(len(boundary_pos)) if len(boundary_pos) > 0 else []  # 1 = boundary
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

        # Simulate one step
        if len(boundary_pos) > 0:
            fluid_pos, fluid_vel = simulator.step(fluid_pos, fluid_vel, boundary_pos, boundary_vel)
        else:
            fluid_pos, fluid_vel = simulator.step(fluid_pos, fluid_vel, np.array([]), np.array([]))

        if step % 20 == 0:
            print(f"  Step {step}/{num_steps} - t={t:.2f}s - "
                  f"Fluid: {len(fluid_pos)}, Boundary: {len(boundary_pos)}")

    print(f"[OK] Saved {num_steps} frames to {scenario_dir}")
    return num_steps


def main():
    """Generate diverse SPH scenarios with rigid body interactions"""

    output_dir = Path("datasets/flip_simulations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scenario types with rigid bodies
    scenario_types = [
        "falling_ball",      # Ball drops into water
        "paddle_wave",       # Paddle creates waves
        "rotating_stirrer",  # Rotating cylinder
        "moving_obstacle",   # Box pushes through
        "double_paddle",     # Two oscillating paddles
        "sphere_bounce"      # Bouncing sphere
    ]

    scenarios_per_type = 5
    steps_per_scenario = 80

    total_scenarios = len(scenario_types) * scenarios_per_type
    total_frames = 0

    print(f"\n{'='*70}")
    print(f"SPH DATA GENERATION WITH RIGID BODY INTERACTIONS")
    print(f"{'='*70}")
    print(f"Scenario types: {len(scenario_types)}")
    print(f"Scenarios per type: {scenarios_per_type}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Steps per scenario: {steps_per_scenario}")
    print(f"{'='*70}\n")

    start_time = time.time()

    for scenario_type in scenario_types:
        print(f"\n{'#'*70}")
        print(f"# Generating: {scenario_type.upper()}")
        print(f"{'#'*70}")

        for idx in range(scenarios_per_type):
            frames = simulate_scenario(
                scenario_type,
                idx,
                output_dir,
                num_steps=steps_per_scenario
            )
            total_frames += frames

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Total frames: {total_frames}")
    print(f"Training sequences: {total_frames - total_scenarios} (consecutive pairs)")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
