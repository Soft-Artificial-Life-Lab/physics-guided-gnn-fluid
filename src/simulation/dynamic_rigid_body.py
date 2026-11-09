"""
Dynamic Rigid Body with Two-Way Coupling
Physics-based rigid body that responds to fluid forces
"""

import numpy as np
from typing import Tuple, Callable
from src.simulation.rigid_body import SphereRigidBody, BoxRigidBody, CylinderRigidBody


class DynamicSphereRigidBody:
    """
    Sphere rigid body with physics dynamics
    Responds to forces from fluid (two-way coupling)
    """

    def __init__(self, name: str, radius: float, mass: float,
                 initial_position: np.ndarray, initial_velocity: np.ndarray,
                 particle_spacing: float = 0.05):
        """
        Args:
            name: Identifier
            radius: Sphere radius
            mass: Mass in kg
            initial_position: Initial center position (3,)
            initial_velocity: Initial velocity (3,)
            particle_spacing: Spacing for boundary particles
        """
        self.name = name
        self.radius = radius
        self.mass = mass
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.force = np.zeros(3)

        # Sample boundary particles on surface
        self.particle_offsets = self._sample_sphere_surface(particle_spacing)
        self.n_particles = len(self.particle_offsets)

        print(f"DynamicSphere '{name}': mass={mass:.2f}kg, radius={radius:.3f}m, "
              f"{self.n_particles} boundary particles")

    def _sample_sphere_surface(self, particle_spacing: float) -> np.ndarray:
        """Sample sphere surface using Fibonacci sphere"""
        surface_area = 4 * np.pi * self.radius**2
        n_particles = int(surface_area / (particle_spacing**2))

        particles = []
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle

        for i in range(n_particles):
            y = 1 - (i / float(n_particles - 1)) * 2  # y from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)

            theta = phi * i

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            particles.append([x * self.radius, y * self.radius, z * self.radius])

        return np.array(particles)

    def apply_force(self, force: np.ndarray):
        """Apply force to rigid body"""
        self.force += np.array(force)

    def update(self, dt: float, gravity: np.ndarray = np.array([0, -9.81, 0])):
        """
        Update position and velocity using accumulated forces

        Args:
            dt: Time step
            gravity: Gravity vector (default: -9.81 m/s^2 in Y)
        """
        # Compute acceleration (F = ma)
        self.acceleration = self.force / self.mass + gravity

        # Update velocity (v = v0 + a*dt)
        self.velocity += self.acceleration * dt

        # Update position (x = x0 + v*dt)
        self.position += self.velocity * dt

        # Reset force for next step
        self.force = np.zeros(3)

    def get_boundary_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current boundary particle positions and velocities

        Returns:
            positions: (N, 3)
            velocities: (N, 3) - all particles have same velocity (rigid body)
        """
        # Transform particle offsets to current position
        positions = self.particle_offsets + self.position

        # All particles have the same velocity (rigid body constraint)
        velocities = np.tile(self.velocity, (self.n_particles, 1))

        return positions, velocities

    def apply_boundary_constraint(self, domain_size: np.ndarray, restitution: float = 0.5):
        """
        Apply domain boundary constraints (bounce off walls)

        Args:
            domain_size: (3,) domain size [width, height, depth]
            restitution: Coefficient of restitution (0=inelastic, 1=elastic)
        """
        for dim in range(3):
            # Lower boundary
            if self.position[dim] - self.radius < 0:
                self.position[dim] = self.radius
                self.velocity[dim] = -self.velocity[dim] * restitution

            # Upper boundary
            if self.position[dim] + self.radius > domain_size[dim]:
                self.position[dim] = domain_size[dim] - self.radius
                self.velocity[dim] = -self.velocity[dim] * restitution


class DynamicBoxRigidBody:
    """Box rigid body with physics dynamics"""

    def __init__(self, name: str, size: Tuple[float, float, float], mass: float,
                 initial_position: np.ndarray, initial_velocity: np.ndarray,
                 particle_spacing: float = 0.05):
        """
        Args:
            name: Identifier
            size: (width, height, depth)
            mass: Mass in kg
            initial_position: Initial center position
            initial_velocity: Initial velocity
        """
        self.name = name
        self.size = np.array(size)
        self.mass = mass
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.force = np.zeros(3)

        # Sample boundary particles
        self.particle_offsets = self._sample_box_surface(particle_spacing)
        self.n_particles = len(self.particle_offsets)

        print(f"DynamicBox '{name}': mass={mass:.2f}kg, size={size}, "
              f"{self.n_particles} boundary particles")

    def _sample_box_surface(self, particle_spacing: float) -> np.ndarray:
        """Sample box surface"""
        particles = []
        w, h, d = self.size

        # Number of particles per dimension
        nx = int(w / particle_spacing) + 1
        ny = int(h / particle_spacing) + 1
        nz = int(d / particle_spacing) + 1

        # Six faces of the box
        # Bottom and top
        for i in np.linspace(-w/2, w/2, nx):
            for k in np.linspace(-d/2, d/2, nz):
                particles.append([i, -h/2, k])
                particles.append([i, h/2, k])

        # Front and back
        for i in np.linspace(-w/2, w/2, nx):
            for j in np.linspace(-h/2, h/2, ny)[1:-1]:
                particles.append([i, j, -d/2])
                particles.append([i, j, d/2])

        # Left and right
        for k in np.linspace(-d/2, d/2, nz)[1:-1]:
            for j in np.linspace(-h/2, h/2, ny)[1:-1]:
                particles.append([-w/2, j, k])
                particles.append([w/2, j, k])

        return np.array(particles)

    def apply_force(self, force: np.ndarray):
        """Apply force to rigid body"""
        self.force += np.array(force)

    def update(self, dt: float, gravity: np.ndarray = np.array([0, -9.81, 0])):
        """Update position and velocity"""
        self.acceleration = self.force / self.mass + gravity
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.force = np.zeros(3)

    def get_boundary_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current boundary particles"""
        positions = self.particle_offsets + self.position
        velocities = np.tile(self.velocity, (self.n_particles, 1))
        return positions, velocities

    def apply_boundary_constraint(self, domain_size: np.ndarray, restitution: float = 0.5):
        """Apply domain boundary constraints"""
        half_size = self.size / 2
        for dim in range(3):
            if self.position[dim] - half_size[dim] < 0:
                self.position[dim] = half_size[dim]
                self.velocity[dim] = -self.velocity[dim] * restitution
            if self.position[dim] + half_size[dim] > domain_size[dim]:
                self.position[dim] = domain_size[dim] - half_size[dim]
                self.velocity[dim] = -self.velocity[dim] * restitution


class DynamicRigidBodySimulation:
    """Manages dynamic rigid bodies with two-way coupling"""

    def __init__(self, rigid_bodies: list):
        """
        Args:
            rigid_bodies: List of DynamicSphereRigidBody or DynamicBoxRigidBody
        """
        self.rigid_bodies = rigid_bodies
        self.body_ids = {body.name: i for i, body in enumerate(rigid_bodies)}

    def get_all_boundary_particles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all boundary particles

        Returns:
            positions: (N_boundary, 3)
            velocities: (N_boundary, 3)
            body_ids: (N_boundary,) - which rigid body each particle belongs to
        """
        all_positions = []
        all_velocities = []
        all_body_ids = []

        for body in self.rigid_bodies:
            pos, vel = body.get_boundary_particles()
            all_positions.append(pos)
            all_velocities.append(vel)
            all_body_ids.append(np.full(len(pos), self.body_ids[body.name]))

        if all_positions:
            return (np.vstack(all_positions),
                   np.vstack(all_velocities),
                   np.hstack(all_body_ids))
        else:
            return np.array([]), np.array([]), np.array([])

    def update_all(self, dt: float, gravity: np.ndarray, domain_size: np.ndarray):
        """Update all rigid bodies"""
        for body in self.rigid_bodies:
            body.update(dt, gravity)
            body.apply_boundary_constraint(domain_size)
