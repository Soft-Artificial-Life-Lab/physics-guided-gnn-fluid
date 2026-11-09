"""
Generate SPH simulation data with DYNAMIC rigid body interactions (Two-way coupling)
Rigid bodies respond to fluid forces (buoyancy, drag, pressure)
"""

import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
import time
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.dynamic_rigid_body import (
    DynamicSphereRigidBody, DynamicBoxRigidBody, DynamicRigidBodySimulation
)


class SPHSimulatorWithDynamicRigidBodies:
    """SPH simulator with two-way coupling to dynamic rigid bodies"""

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

    def compute_density(self, positions):
        """Compute density for all particles"""
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
            densities[i] = max(density, self.rest_density)

        return densities

    def compute_forces_on_fluid(self, positions, velocities, densities, is_fluid_mask):
        """
        Compute forces on fluid particles from fluid and boundary

        Args:
            positions: All particle positions (fluid + boundary)
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

            pressure_i = self.gas_constant * (densities[i] - self.rest_density)

            neighbors = tree.query_ball_point(positions[i], self.h)

            for j in neighbors:
                if i == j:
                    continue

                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)

                if r < self.h and r > 1e-6:
                    # Pressure force (from both fluid and boundary)
                    pressure_j = self.gas_constant * (densities[j] - self.rest_density)
                    pressure_force += -self.particle_mass * (pressure_i + pressure_j) / (2 * densities[j]) * \
                                     self.spiky_gradient(r_vec)

                    # Viscosity force (only fluid-fluid)
                    if is_fluid_mask[j]:
                        vel_diff = velocities[j] - velocities[i]
                        viscosity_force += self.viscosity * self.particle_mass * vel_diff / densities[j] * \
                                          self.viscosity_laplacian(r)

            forces[idx] = pressure_force + viscosity_force + self.gravity * densities[i]

        return forces

    def compute_forces_on_rigid_bodies(self, positions, densities, is_fluid_mask,
                                       rigid_body_simulation):
        """
        Compute forces from fluid on each rigid body (TWO-WAY COUPLING!)

        Args:
            positions: All particle positions (fluid + boundary)
            densities: All particle densities
            is_fluid_mask: Boolean mask for fluid particles
            rigid_body_simulation: DynamicRigidBodySimulation object

        Returns:
            forces_on_bodies: Dict {body_name: force_vector}
        """
        forces_on_bodies = {body.name: np.zeros(3) for body in rigid_body_simulation.rigid_bodies}

        # Get boundary particle info
        boundary_positions, _, boundary_body_ids = rigid_body_simulation.get_all_boundary_particles()

        if len(boundary_positions) == 0:
            return forces_on_bodies

        n_fluid = np.sum(is_fluid_mask)
        fluid_positions = positions[:n_fluid]

        tree = KDTree(fluid_positions)

        # For each boundary particle, compute force from nearby fluid
        for boundary_idx, boundary_pos in enumerate(boundary_positions):
            body_id = boundary_body_ids[boundary_idx]
            body = rigid_body_simulation.rigid_bodies[body_id]

            # Find nearby fluid particles
            neighbors = tree.query_ball_point(boundary_pos, self.h)

            force_on_this_particle = np.zeros(3)

            for fluid_idx in neighbors:
                r_vec = boundary_pos - fluid_positions[fluid_idx]
                r = np.linalg.norm(r_vec)

                if r < self.h and r > 1e-6:
                    # Pressure force from fluid to boundary
                    # This is the reaction force (Newton's 3rd law)
                    boundary_particle_idx = n_fluid + boundary_idx
                    pressure_boundary = self.gas_constant * (densities[boundary_particle_idx] - self.rest_density)
                    pressure_fluid = self.gas_constant * (densities[fluid_idx] - self.rest_density)

                    # Force FROM fluid TO boundary
                    force = self.particle_mass * (pressure_fluid + pressure_boundary) / (2 * densities[fluid_idx]) * \
                           self.spiky_gradient(r_vec)

                    force_on_this_particle += force

            forces_on_bodies[body.name] += force_on_this_particle

        return forces_on_bodies

    def step(self, fluid_positions, fluid_velocities, rigid_body_simulation):
        """
        Single simulation step with dynamic rigid bodies (two-way coupling)

        Args:
            fluid_positions: (N_fluid, 3)
            fluid_velocities: (N_fluid, 3)
            rigid_body_simulation: DynamicRigidBodySimulation

        Returns:
            new_fluid_positions, new_fluid_velocities
        """
        # Get boundary particles from rigid bodies
        boundary_pos, boundary_vel, _ = rigid_body_simulation.get_all_boundary_particles()

        # Combine fluid and boundary
        all_positions = np.vstack([fluid_positions, boundary_pos]) if len(boundary_pos) > 0 else fluid_positions
        all_velocities = np.vstack([fluid_velocities, boundary_vel]) if len(boundary_vel) > 0 else fluid_velocities

        n_fluid = len(fluid_positions)
        is_fluid = np.zeros(len(all_positions), dtype=bool)
        is_fluid[:n_fluid] = True

        # Compute densities for all particles
        densities = self.compute_density(all_positions)

        # Compute forces on fluid particles
        forces_on_fluid = self.compute_forces_on_fluid(all_positions, all_velocities, densities, is_fluid)

        # TWO-WAY COUPLING: Compute forces on rigid bodies from fluid
        if len(boundary_pos) > 0:
            forces_on_bodies = self.compute_forces_on_rigid_bodies(
                all_positions, densities, is_fluid, rigid_body_simulation
            )

            # Apply forces to rigid bodies
            for body in rigid_body_simulation.rigid_bodies:
                body.apply_force(forces_on_bodies[body.name])

            # Update rigid bodies
            rigid_body_simulation.update_all(self.dt, self.gravity, self.domain_size)

        # Update fluid particles
        accelerations = forces_on_fluid / densities[:n_fluid, np.newaxis]
        new_velocities = fluid_velocities + accelerations * self.dt
        new_positions = fluid_positions + new_velocities * self.dt

        # Boundary conditions for fluid (domain walls)
        for dim in range(3):
            mask = new_positions[:, dim] < 0
            new_positions[mask, dim] = 0
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

            mask = new_positions[:, dim] > self.domain_size[dim]
            new_positions[mask, dim] = self.domain_size[dim]
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

        return new_positions, new_velocities

    def step_with_static_boundary(self, fluid_positions, fluid_velocities, rigid_body_simulation,
                                   static_boundary_pos, static_boundary_vel):
        """
        Single simulation step with both static boundaries and dynamic rigid bodies

        Args:
            fluid_positions: (N_fluid, 3)
            fluid_velocities: (N_fluid, 3)
            rigid_body_simulation: DynamicRigidBodySimulation
            static_boundary_pos: (N_static, 3) static boundary particles
            static_boundary_vel: (N_static, 3) zero velocities

        Returns:
            new_fluid_positions, new_fluid_velocities
        """
        # Get dynamic boundary particles from rigid bodies
        dynamic_boundary_pos, dynamic_boundary_vel, _ = rigid_body_simulation.get_all_boundary_particles()

        # Combine static and dynamic boundaries
        if len(dynamic_boundary_pos) > 0:
            boundary_pos = np.vstack([static_boundary_pos, dynamic_boundary_pos])
            boundary_vel = np.vstack([static_boundary_vel, dynamic_boundary_vel])
        else:
            boundary_pos = static_boundary_pos
            boundary_vel = static_boundary_vel

        # Combine fluid and boundary
        all_positions = np.vstack([fluid_positions, boundary_pos])
        all_velocities = np.vstack([fluid_velocities, boundary_vel])

        n_fluid = len(fluid_positions)
        is_fluid = np.zeros(len(all_positions), dtype=bool)
        is_fluid[:n_fluid] = True

        # Compute densities for all particles
        densities = self.compute_density(all_positions)

        # Compute forces on fluid particles
        forces_on_fluid = self.compute_forces_on_fluid(all_positions, all_velocities, densities, is_fluid)

        # TWO-WAY COUPLING: Compute forces on rigid bodies from fluid (only for dynamic bodies)
        if len(dynamic_boundary_pos) > 0:
            forces_on_bodies = self.compute_forces_on_rigid_bodies(
                all_positions, densities, is_fluid, rigid_body_simulation
            )

            # Apply forces to rigid bodies
            for body in rigid_body_simulation.rigid_bodies:
                body.apply_force(forces_on_bodies[body.name])

            # Update rigid bodies
            rigid_body_simulation.update_all(self.dt, self.gravity, self.domain_size)

        # Update fluid particles
        accelerations = forces_on_fluid / densities[:n_fluid, np.newaxis]
        new_velocities = fluid_velocities + accelerations * self.dt
        new_positions = fluid_positions + new_velocities * self.dt

        # Boundary conditions for fluid (domain walls)
        for dim in range(3):
            mask = new_positions[:, dim] < 0
            new_positions[mask, dim] = 0
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

            mask = new_positions[:, dim] > self.domain_size[dim]
            new_positions[mask, dim] = self.domain_size[dim]
            new_velocities[mask, dim] = -new_velocities[mask, dim] * 0.5

        return new_positions, new_velocities


def create_static_boundary_particles(domain_size=(1.0, 1.0, 1.0), spacing=0.05):
    """Create static boundary particles at the bottom and walls of the domain"""
    boundary_particles = []

    # Bottom plane (Y=0)
    x_points = int(domain_size[0] / spacing) + 1
    z_points = int(domain_size[2] / spacing) + 1

    for i in range(x_points):
        for k in range(z_points):
            x = i * spacing
            z = k * spacing
            if x <= domain_size[0] and z <= domain_size[2]:
                boundary_particles.append([x, 0.0, z])

    # Left wall (X=0)
    y_points = int(domain_size[1] / spacing) + 1
    for j in range(1, y_points):  # Start from 1 to avoid duplicates with bottom
        for k in range(z_points):
            y = j * spacing
            z = k * spacing
            if y <= domain_size[1] and z <= domain_size[2]:
                boundary_particles.append([0.0, y, z])

    # Right wall (X=domain_size[0])
    for j in range(1, y_points):
        for k in range(z_points):
            y = j * spacing
            z = k * spacing
            if y <= domain_size[1] and z <= domain_size[2]:
                boundary_particles.append([domain_size[0], y, z])

    # Front wall (Z=0)
    for i in range(1, x_points - 1):  # Avoid corner duplicates
        for j in range(1, y_points):
            x = i * spacing
            y = j * spacing
            if x <= domain_size[0] and y <= domain_size[1]:
                boundary_particles.append([x, y, 0.0])

    # Back wall (Z=domain_size[2])
    for i in range(1, x_points - 1):
        for j in range(1, y_points):
            x = i * spacing
            y = j * spacing
            if x <= domain_size[0] and y <= domain_size[1]:
                boundary_particles.append([x, y, domain_size[2]])

    return np.array(boundary_particles)


def create_dynamic_scenario(scenario_type: str, scenario_idx: int):
    """Create scenarios with dynamic rigid bodies"""

    # Add variation based on scenario_idx
    np.random.seed(scenario_idx * 1000)

    # Create static boundary particles
    static_boundary = create_static_boundary_particles(domain_size=(1.0, 1.0, 1.0), spacing=0.05)

    if scenario_type == "falling_ball":
        # Ball drops into water with physics
        n_particles = 1500  # Increased particles
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.7, 1.0]  # Height: 0.4 -> 0.7
        fluid_velocities = np.zeros((n_particles, 3))

        # Dynamic ball with mass and initial velocity - vary parameters
        radius_var = 0.15 + (scenario_idx % 5) * 0.02  # 0.15 to 0.23
        mass_var = 4.0 + (scenario_idx % 5) * 1.0  # 4 to 8 kg
        height_var = 1.2 + (scenario_idx % 5) * 0.1  # 1.2 to 1.6

        ball = DynamicSphereRigidBody(
            name="ball",
            radius=radius_var,
            mass=mass_var,
            initial_position=np.array([0.5, height_var, 0.5]),
            initial_velocity=np.array([0.0, -2.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball])

    elif scenario_type == "floating_ball":
        # Ball with buoyancy - floats in water
        n_particles = 1800  # Increased particles
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.8, 1.0]  # Height: 0.6 -> 0.8
        fluid_velocities = np.zeros((n_particles, 3))

        # Light ball that floats
        ball = DynamicSphereRigidBody(
            name="floater",
            radius=0.12 + scenario_idx * 0.01,
            mass=1.0 + scenario_idx * 0.3,  # Light - will float
            initial_position=np.array([0.5, 0.9, 0.5]),  # Start above water: 0.8 -> 0.9
            initial_velocity=np.array([0.0, 0.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([ball])

    elif scenario_type == "heavy_box":
        # Heavy box sinking through water
        n_particles = 1600  # Increased particles
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.7, 1.0]  # Height: 0.5 -> 0.7
        fluid_velocities = np.zeros((n_particles, 3))

        # Dense box
        box = DynamicBoxRigidBody(
            name="heavy_box",
            size=(0.15, 0.15, 0.15),
            mass=15.0 + scenario_idx * 3.0,
            initial_position=np.array([0.5, 1.0, 0.5]),
            initial_velocity=np.array([0.0, -1.0, 0.0])
        )
        rigid_sim = DynamicRigidBodySimulation([box])

    elif scenario_type == "multi_ball":
        # Multiple balls with different properties
        n_particles = 1400  # Increased particles
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.7, 1.0]  # Height: 0.4 -> 0.7
        fluid_velocities = np.zeros((n_particles, 3))

        # Two balls with different masses
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
        # Sphere with high restitution - bounces a lot
        n_particles = 1200  # Increased particles
        fluid_positions = np.random.rand(n_particles, 3) * [1.0, 0.6, 1.0]  # Height: 0.3 -> 0.6
        fluid_velocities = np.zeros((n_particles, 3))

        ball = DynamicSphereRigidBody(
            name="bouncer",
            radius=0.15,
            mass=4.0 + scenario_idx,
            initial_position=np.array([0.5, 1.4, 0.5]),
            initial_velocity=np.array([0.0, -3.0, 0.0])
        )
        # Note: restitution set in boundary constraint (0.5 default)
        rigid_sim = DynamicRigidBodySimulation([ball])

    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")

    return fluid_positions, fluid_velocities, rigid_sim, static_boundary


def simulate_dynamic_scenario(scenario_type: str, scenario_idx: int, output_dir: Path,
                              num_steps=80, dt=0.01):
    """Run simulation with dynamic rigid bodies and save data"""

    print(f"\n{'='*70}")
    print(f"DYNAMIC SCENARIO: {scenario_type} #{scenario_idx}")
    print(f"{'='*70}")

    # Create scenario
    fluid_pos, fluid_vel, rigid_sim, static_boundary = create_dynamic_scenario(scenario_type, scenario_idx)

    # Initialize simulator
    simulator = SPHSimulatorWithDynamicRigidBodies(dt=dt)

    # Create output directory
    scenario_dir = output_dir / f"sph_dynamic_{scenario_type}_{scenario_idx:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initial: {len(fluid_pos)} fluid particles, {len(static_boundary)} static boundary particles")
    print(f"Two-way coupling: ENABLED")

    # Static boundary velocities (all zero)
    static_boundary_vel = np.zeros_like(static_boundary)

    for step in range(num_steps):
        t = step * dt

        # Get rigid body particles
        rigid_boundary_pos, rigid_boundary_vel, body_ids = rigid_sim.get_all_boundary_particles()

        # Combine static and dynamic boundaries
        if len(rigid_boundary_pos) > 0:
            boundary_pos = np.vstack([static_boundary, rigid_boundary_pos])
            boundary_vel = np.vstack([static_boundary_vel, rigid_boundary_vel])
            # Static boundaries have body_id=-2, dynamic boundaries have their actual IDs
            all_body_ids = np.hstack([
                np.full(len(static_boundary), -2),
                body_ids
            ])
        else:
            boundary_pos = static_boundary
            boundary_vel = static_boundary_vel
            all_body_ids = np.full(len(static_boundary), -2)

        # Save current state
        output_file = scenario_dir / f"frame_{step:04d}.npz"

        all_pos = np.vstack([fluid_pos, boundary_pos])
        all_vel = np.vstack([fluid_vel, boundary_vel])

        particle_types = np.hstack([
            np.zeros(len(fluid_pos)),
            np.ones(len(boundary_pos))
        ])

        np.savez_compressed(
            output_file,
            positions=all_pos,
            velocities=all_vel,
            particle_types=particle_types,
            body_ids=np.hstack([
                np.full(len(fluid_pos), -1),
                all_body_ids
            ]),
            time=t,
            n_fluid=len(fluid_pos),
            n_boundary=len(boundary_pos)
        )

        # Simulate one step (TWO-WAY COUPLING!)
        # Pass static boundary to simulator
        fluid_pos, fluid_vel = simulator.step_with_static_boundary(
            fluid_pos, fluid_vel, rigid_sim, static_boundary, static_boundary_vel
        )

        if step % 10 == 0:
            # Print rigid body info
            if len(rigid_sim.rigid_bodies) > 0:
                body = rigid_sim.rigid_bodies[0]
                print(f"  Step {step:3d}/{num_steps} - t={t:.2f}s - "
                      f"Fluid: {len(fluid_pos)}, Body Y: {body.position[1]:.3f}, "
                      f"Body Vel: {body.velocity[1]:.3f} m/s")

    print(f"[OK] Saved {num_steps} frames to {scenario_dir}")
    return num_steps


def main():
    """Generate dynamic SPH scenarios with two-way coupling"""

    output_dir = Path("datasets/flip_simulations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SPH DATA GENERATION WITH DYNAMIC RIGID BODIES (TWO-WAY COUPLING)")
    print("="*70)

    # Full dataset with multiple scenario types
    scenario_types = [
        "falling_ball",      # Ball drops and bounces
        "floating_ball",     # Light ball that floats
        "heavy_box",         # Heavy box sinks
        "multi_ball",        # Multiple balls interacting
        "bouncing_sphere"    # High-energy bouncing
    ]
    num_variations = 5  # 5 variations each = 25 scenarios total

    total_scenarios = 0
    start_time = time.time()

    for scenario_type in scenario_types:
        print(f"\nGenerating {scenario_type} scenarios...")
        for i in range(num_variations):
            simulate_dynamic_scenario(scenario_type, i, output_dir)
            total_scenarios += 1

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total scenarios: {total_scenarios}")
    print(f"Scenario types: {len(scenario_types)}")
    print(f"Variations per type: {num_variations}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Average: {elapsed/total_scenarios:.1f}s per scenario")
    print("="*70)


if __name__ == "__main__":
    main()
