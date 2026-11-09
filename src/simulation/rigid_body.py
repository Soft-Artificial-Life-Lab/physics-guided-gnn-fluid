"""
Rigid Body Utilities for SPH-GNN Fluid Simulation
Implements moving rigid body boundary conditions
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Callable


class RigidBody:
    """Represents a rigid body with boundary particles"""

    def __init__(self, name: str, position_fn: Callable, velocity_fn: Callable = None):
        """
        Args:
            name: Identifier for this rigid body
            position_fn: Function(t) -> (x, y, z) returning center position at time t
            velocity_fn: Function(t) -> (vx, vy, vz) returning velocity at time t
        """
        self.name = name
        self.position_fn = position_fn
        self.velocity_fn = velocity_fn or (lambda t: np.array([0.0, 0.0, 0.0]))
        self.boundary_particles = None
        self.particle_offsets = None  # Relative positions from center

    def sample_surface(self, particle_spacing: float = 0.05):
        """Sample boundary particles on the surface"""
        raise NotImplementedError("Subclass must implement sample_surface")

    def get_particles_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get particle positions and velocities at time t

        Returns:
            positions: (N, 3) array
            velocities: (N, 3) array
        """
        center = self.position_fn(t)
        velocity = self.velocity_fn(t)

        # Transform particle offsets to current position
        positions = self.particle_offsets + center

        # All particles have the same velocity (rigid body)
        velocities = np.tile(velocity, (len(positions), 1))

        return positions, velocities


class SphereRigidBody(RigidBody):
    """Spherical rigid body"""

    def __init__(self, name: str, radius: float, position_fn: Callable, velocity_fn: Callable = None):
        super().__init__(name, position_fn, velocity_fn)
        self.radius = radius
        self.sample_surface()

    def sample_surface(self, particle_spacing: float = 0.05):
        """Sample sphere surface using Fibonacci sphere algorithm"""
        # Estimate number of particles
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

        self.particle_offsets = np.array(particles)
        print(f"Sphere '{self.name}': {len(particles)} boundary particles (r={self.radius})")


class BoxRigidBody(RigidBody):
    """Box-shaped rigid body"""

    def __init__(self, name: str, size: Tuple[float, float, float],
                 position_fn: Callable, velocity_fn: Callable = None):
        super().__init__(name, position_fn, velocity_fn)
        self.size = np.array(size)  # (width, height, depth)
        self.sample_surface()

    def sample_surface(self, particle_spacing: float = 0.05):
        """Sample box surface"""
        particles = []
        w, h, d = self.size

        # Number of particles per dimension
        nx = int(w / particle_spacing) + 1
        ny = int(h / particle_spacing) + 1
        nz = int(d / particle_spacing) + 1

        # Six faces of the box
        # Bottom and top (y = -h/2 and y = h/2)
        for i in np.linspace(-w/2, w/2, nx):
            for k in np.linspace(-d/2, d/2, nz):
                particles.append([i, -h/2, k])  # Bottom
                particles.append([i, h/2, k])   # Top

        # Front and back (z = -d/2 and z = d/2)
        for i in np.linspace(-w/2, w/2, nx):
            for j in np.linspace(-h/2, h/2, ny)[1:-1]:  # Avoid corners
                particles.append([i, j, -d/2])  # Front
                particles.append([i, j, d/2])   # Back

        # Left and right (x = -w/2 and x = w/2)
        for k in np.linspace(-d/2, d/2, nz)[1:-1]:  # Avoid corners
            for j in np.linspace(-h/2, h/2, ny)[1:-1]:
                particles.append([-w/2, j, k])  # Left
                particles.append([w/2, j, k])   # Right

        self.particle_offsets = np.array(particles)
        print(f"Box '{self.name}': {len(particles)} boundary particles (size={self.size})")


class CylinderRigidBody(RigidBody):
    """Cylindrical rigid body"""

    def __init__(self, name: str, radius: float, height: float,
                 position_fn: Callable, velocity_fn: Callable = None):
        super().__init__(name, position_fn, velocity_fn)
        self.radius = radius
        self.height = height
        self.sample_surface()

    def sample_surface(self, particle_spacing: float = 0.05):
        """Sample cylinder surface"""
        particles = []

        # Circular caps
        n_radial = int(self.radius / particle_spacing) + 1
        n_angular = int(2 * np.pi * self.radius / particle_spacing)

        for y in [-self.height/2, self.height/2]:  # Top and bottom caps
            for r in np.linspace(0, self.radius, n_radial):
                n_theta = max(1, int(2 * np.pi * r / particle_spacing))
                for theta in np.linspace(0, 2*np.pi, n_theta, endpoint=False):
                    x = r * np.cos(theta)
                    z = r * np.sin(theta)
                    particles.append([x, y, z])

        # Cylindrical side
        n_height = int(self.height / particle_spacing) + 1
        for theta in np.linspace(0, 2*np.pi, n_angular, endpoint=False):
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            for y in np.linspace(-self.height/2, self.height/2, n_height)[1:-1]:
                particles.append([x, y, z])

        self.particle_offsets = np.array(particles)
        print(f"Cylinder '{self.name}': {len(particles)} boundary particles (r={self.radius}, h={self.height})")


class RigidBodySimulation:
    """Manages multiple rigid bodies in simulation"""

    def __init__(self, rigid_bodies: List[RigidBody], particle_spacing: float = 0.05):
        self.rigid_bodies = rigid_bodies
        self.particle_spacing = particle_spacing
        self.body_ids = {body.name: i for i, body in enumerate(rigid_bodies)}

    def get_all_boundary_particles(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all boundary particles at time t

        Returns:
            positions: (N_boundary, 3)
            velocities: (N_boundary, 3)
            body_ids: (N_boundary,) - which rigid body each particle belongs to
        """
        all_positions = []
        all_velocities = []
        all_body_ids = []

        for body in self.rigid_bodies:
            pos, vel = body.get_particles_at_time(t)
            all_positions.append(pos)
            all_velocities.append(vel)
            all_body_ids.append(np.full(len(pos), self.body_ids[body.name]))

        if all_positions:
            return (np.vstack(all_positions),
                   np.vstack(all_velocities),
                   np.hstack(all_body_ids))
        else:
            return np.array([]), np.array([]), np.array([])

    def combine_with_fluid(self, fluid_pos: np.ndarray, fluid_vel: np.ndarray, t: float):
        """
        Combine fluid and boundary particles

        Returns:
            positions: (N_total, 3)
            velocities: (N_total, 3)
            particle_types: (N_total,) - 0 for fluid, 1 for boundary
            body_ids: (N_total,) - body ID for boundary particles, -1 for fluid
        """
        boundary_pos, boundary_vel, boundary_ids = self.get_all_boundary_particles(t)

        if len(boundary_pos) == 0:
            # No rigid bodies
            n_fluid = len(fluid_pos)
            return (fluid_pos, fluid_vel,
                   np.zeros(n_fluid),
                   np.full(n_fluid, -1))

        # Combine
        positions = np.vstack([fluid_pos, boundary_pos])
        velocities = np.vstack([fluid_vel, boundary_vel])

        # Particle types: 0 = fluid, 1 = boundary
        particle_types = np.hstack([
            np.zeros(len(fluid_pos)),
            np.ones(len(boundary_pos))
        ])

        # Body IDs: -1 = fluid, >= 0 = boundary body ID
        body_ids = np.hstack([
            np.full(len(fluid_pos), -1),
            boundary_ids
        ])

        return positions, velocities, particle_types, body_ids


# Predefined motion functions
def static_position(center: np.ndarray):
    """Returns a function for static position"""
    return lambda t: np.array(center)


def linear_motion(start: np.ndarray, velocity: np.ndarray):
    """Returns a function for constant velocity motion"""
    return lambda t: np.array(start) + np.array(velocity) * t


def oscillating_motion(center: np.ndarray, amplitude: np.ndarray, frequency: float):
    """Returns a function for sinusoidal oscillation"""
    def motion(t):
        return np.array(center) + np.array(amplitude) * np.sin(2 * np.pi * frequency * t)
    return motion


def circular_motion(center: np.ndarray, radius: float, angular_velocity: float, axis: str = 'z'):
    """Returns a function for circular motion around an axis"""
    def motion(t):
        angle = angular_velocity * t
        if axis == 'z':
            return center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        elif axis == 'y':
            return center + np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
        elif axis == 'x':
            return center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
    return motion


def oscillating_velocity(amplitude: np.ndarray, frequency: float):
    """Returns velocity function for oscillating motion"""
    def velocity(t):
        return np.array(amplitude) * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
    return velocity


def circular_velocity(radius: float, angular_velocity: float, axis: str = 'z'):
    """Returns velocity function for circular motion"""
    def velocity(t):
        angle = angular_velocity * t
        if axis == 'z':
            return np.array([
                -radius * angular_velocity * np.sin(angle),
                radius * angular_velocity * np.cos(angle),
                0
            ])
        elif axis == 'y':
            return np.array([
                -radius * angular_velocity * np.sin(angle),
                0,
                radius * angular_velocity * np.cos(angle)
            ])
        elif axis == 'x':
            return np.array([
                0,
                -radius * angular_velocity * np.sin(angle),
                radius * angular_velocity * np.cos(angle)
            ])
    return velocity


if __name__ == "__main__":
    # Example usage
    print("=== Rigid Body Boundary Particle Generation ===\n")

    # Create a falling sphere
    sphere = SphereRigidBody(
        name="falling_ball",
        radius=0.2,
        position_fn=linear_motion([0.5, 1.0, 0.5], [0, -0.5, 0]),
        velocity_fn=lambda t: np.array([0, -0.5, 0])
    )

    # Create an oscillating paddle
    paddle = BoxRigidBody(
        name="paddle",
        size=(0.3, 0.05, 0.6),
        position_fn=oscillating_motion([0.5, 0.3, 0.5], [0.3, 0, 0], 0.5),
        velocity_fn=oscillating_velocity([0.3, 0, 0], 0.5)
    )

    # Create a rotating cylinder
    cylinder = CylinderRigidBody(
        name="stirrer",
        radius=0.1,
        height=0.4,
        position_fn=circular_motion([0.5, 0.5, 0.5], 0.2, 1.0, 'z'),
        velocity_fn=circular_velocity(0.2, 1.0, 'z')
    )

    # Test at different times
    print("\n=== Testing at t=0.0 ===")
    pos0, vel0 = sphere.get_particles_at_time(0.0)
    print(f"Sphere: {len(pos0)} particles, center at {sphere.position_fn(0.0)}, vel={sphere.velocity_fn(0.0)}")

    print("\n=== Testing at t=1.0 ===")
    pos1, vel1 = sphere.get_particles_at_time(1.0)
    print(f"Sphere: {len(pos1)} particles, center at {sphere.position_fn(1.0)}, vel={sphere.velocity_fn(1.0)}")

    # Create simulation with multiple bodies
    sim = RigidBodySimulation([sphere, paddle, cylinder])

    # Simulate fluid particles
    fluid_pos = np.random.rand(100, 3)
    fluid_vel = np.random.randn(100, 3) * 0.1

    # Combine with rigid bodies
    all_pos, all_vel, types, ids = sim.combine_with_fluid(fluid_pos, fluid_vel, t=0.5)

    print(f"\n=== Combined System at t=0.5 ===")
    print(f"Total particles: {len(all_pos)}")
    print(f"  Fluid particles: {np.sum(types == 0)}")
    print(f"  Boundary particles: {np.sum(types == 1)}")
    print(f"    - Sphere: {np.sum(ids == 0)}")
    print(f"    - Paddle: {np.sum(ids == 1)}")
    print(f"    - Stirrer: {np.sum(ids == 2)}")
