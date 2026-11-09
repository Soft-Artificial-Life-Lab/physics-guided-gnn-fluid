"""
v2 SPH data generator with DYNAMIC rigid bodies (method 2).

Differences from v1 (`generate_sph_with_rigid_bodies.py`):
  + Free-fall scenarios (`falling_ball`, `sphere_bounce`) use
    DynamicSphereRigidBody — gravity, floor/wall collision (restitution),
    AND two-way fluid<->rigid SPH force coupling.
  + Driven scenarios (`paddle_wave`, `rotating_stirrer`, `moving_obstacle`,
    `double_paddle`) keep their kinematic motion BUT receive a floor-clamp
    safety check and adjusted center heights so rigid never crosses y=0.
  + Architecture (HybridPhysicsGNN) and dataset I/O format are UNCHANGED.

Output format is byte-compatible with v1: each frame_*.npz has
  positions (N,3), velocities (N,3), particle_types (N,), body_ids (N,),
  time, n_fluid, n_boundary

Output path:
  datasets/flip_simulations_v2_20260504_063347/sph_<scenario>_<idx>/frame_*.npz

The original v1 generator and v1 dataset are NOT touched.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.simulation.rigid_body import (
    SphereRigidBody, BoxRigidBody, CylinderRigidBody,
    linear_motion, oscillating_motion, circular_motion,
    oscillating_velocity, circular_velocity, static_position,
    RigidBodySimulation,
)
from src.simulation.dynamic_rigid_body import (
    DynamicSphereRigidBody, DynamicBoxRigidBody, DynamicRigidBodySimulation,
)


TIMESTAMP = "20260504_063347"
OUTPUT_ROOT = ROOT / "datasets" / f"flip_simulations_v2_{TIMESTAMP}"


# =============================================================================
# SPH simulator with two-way fluid<->rigid coupling
# =============================================================================
class SPHSimulatorV2:
    """SPH with two-way coupling for dynamic rigid bodies."""

    def __init__(self, domain_size=(1.0, 1.0, 1.0), particle_mass=0.02,
                 rest_density=1000.0, gas_constant=2000.0, viscosity=0.1,
                 dt=0.01, gravity=-9.81, h=0.1):
        self.domain_size = np.array(domain_size, dtype=float)
        self.particle_mass = particle_mass
        self.rest_density = rest_density
        self.gas_constant = gas_constant
        self.viscosity = viscosity
        self.dt = dt
        self.gravity = np.array([0.0, gravity, 0.0])
        self.h = h
        self.h2 = h * h

        self.poly6_coeff = 315.0 / (64.0 * np.pi * h ** 9)
        self.spiky_grad_coeff = -45.0 / (np.pi * h ** 6)
        self.visc_lap_coeff = 45.0 / (np.pi * h ** 6)

    # --- SPH kernels (same form as v1) ---
    def poly6(self, r2):
        return self.poly6_coeff * (self.h2 - r2) ** 3 if r2 < self.h2 else 0.0

    def spiky_grad(self, r_vec):
        r = np.linalg.norm(r_vec)
        if 1e-6 < r < self.h:
            return self.spiky_grad_coeff * (self.h - r) ** 2 * r_vec / r
        return np.zeros(3)

    def visc_lap(self, r):
        return self.visc_lap_coeff * (self.h - r) if r < self.h else 0.0

    def compute_density(self, positions):
        n = len(positions)
        densities = np.zeros(n)
        tree = KDTree(positions)
        nbrs = tree.query_ball_tree(tree, self.h)
        for i in range(n):
            d = 0.0
            for j in nbrs[i]:
                r2 = float(np.dot(positions[i] - positions[j], positions[i] - positions[j]))
                d += self.particle_mass * self.poly6(r2)
            densities[i] = max(d, self.rest_density)
        return densities, nbrs

    def compute_forces_two_way(self, positions, velocities, densities, neighbors,
                               n_fluid, gravity_on_fluid=True):
        """Compute SPH forces on ALL particles (fluid + rigid).

        Pairs counted:
          - fluid <- fluid : pressure + viscosity (standard SPH)
          - fluid <- rigid : pressure (boundary force from rigid on fluid)
          - rigid <- fluid : pressure (back-reaction, two-way coupling)
          - rigid <- rigid : SKIPPED (would create spurious internal forces;
                              the rigid body constraint is enforced by the
                              DynamicRigidBody class, not by SPH)
        """
        n = len(positions)
        forces = np.zeros((n, 3))
        for i in range(n):
            is_fluid_i = i < n_fluid
            pforce = np.zeros(3)
            vforce = np.zeros(3)
            for j in neighbors[i]:
                if j == i:
                    continue
                is_fluid_j = j < n_fluid
                # rigid<-rigid pair: skip (do not let SPH stress the rigid body)
                if not is_fluid_i and not is_fluid_j:
                    continue
                rvec = positions[i] - positions[j]
                # pressure
                pressure_i = self.gas_constant * (densities[i] - self.rest_density)
                pressure_j = self.gas_constant * (densities[j] - self.rest_density)
                avg_p = (pressure_i + pressure_j) * 0.5
                pforce += -self.particle_mass * avg_p / densities[j] * self.spiky_grad(rvec)
                # viscosity (only fluid<-fluid pair contributes)
                if is_fluid_i and is_fluid_j:
                    r = float(np.linalg.norm(rvec))
                    vforce += (self.viscosity * self.particle_mass
                               * (velocities[j] - velocities[i]) / densities[j]
                               * self.visc_lap(r))
            f = pforce + vforce
            if is_fluid_i and gravity_on_fluid:
                f += self.gravity * densities[i]
            forces[i] = f
        return forces

    def step(self, fluid_pos, fluid_vel, dyn_sim,
             kin_boundary_pos=None, kin_boundary_vel=None):
        """One integration step.

        Args:
            fluid_pos: (N_fluid, 3)
            fluid_vel: (N_fluid, 3)
            dyn_sim: DynamicRigidBodySimulation or None
            kin_boundary_pos / kin_boundary_vel: optional kinematic rigid
                particles (already at the current frame). If provided, they
                act as boundary for the fluid but do NOT receive reaction
                forces (driven motion).

        Returns:
            new_fluid_pos, new_fluid_vel
        """
        # Gather rigid boundary particles (dynamic + kinematic)
        if dyn_sim is not None:
            d_pos, d_vel, d_ids = dyn_sim.get_all_boundary_particles()
        else:
            d_pos = np.zeros((0, 3)); d_vel = np.zeros((0, 3)); d_ids = np.zeros(0, dtype=int)

        k_pos = kin_boundary_pos if kin_boundary_pos is not None else np.zeros((0, 3))
        k_vel = kin_boundary_vel if kin_boundary_vel is not None else np.zeros((0, 3))

        rigid_pos = np.vstack([d_pos, k_pos]) if (len(d_pos) + len(k_pos)) else np.zeros((0, 3))
        rigid_vel = np.vstack([d_vel, k_vel]) if (len(d_vel) + len(k_vel)) else np.zeros((0, 3))

        all_pos = np.vstack([fluid_pos, rigid_pos]) if len(rigid_pos) else fluid_pos
        all_vel = np.vstack([fluid_vel, rigid_vel]) if len(rigid_vel) else fluid_vel
        n_fluid = len(fluid_pos)

        densities, nbrs = self.compute_density(all_pos)
        forces = self.compute_forces_two_way(all_pos, all_vel, densities, nbrs, n_fluid)

        # Update fluid (existing kinematics)
        accels = forces[:n_fluid] / densities[:n_fluid, np.newaxis]
        new_vel = fluid_vel + accels * self.dt
        new_pos = fluid_pos + new_vel * self.dt
        # Domain wall reflection on fluid
        for dim in range(3):
            mask = new_pos[:, dim] < 0
            new_pos[mask, dim] = 0.0
            new_vel[mask, dim] = -new_vel[mask, dim] * 0.5
            mask = new_pos[:, dim] > self.domain_size[dim]
            new_pos[mask, dim] = self.domain_size[dim]
            new_vel[mask, dim] = -new_vel[mask, dim] * 0.5

        # Two-way coupling: aggregate rigid forces back to dynamic bodies
        if dyn_sim is not None and len(d_pos):
            d_forces = forces[n_fluid:n_fluid + len(d_pos)]
            for body_idx, body in enumerate(dyn_sim.rigid_bodies):
                mask = (d_ids == body_idx)
                if mask.any():
                    body.apply_force(d_forces[mask].sum(axis=0))
            dyn_sim.update_all(self.dt, gravity=self.gravity, domain_size=self.domain_size)

        return new_pos, new_vel


# =============================================================================
# Scenario factories
# =============================================================================
SCENARIO_SEED_OFFSET = {
    "falling_ball": 0, "paddle_wave": 1, "rotating_stirrer": 2,
    "moving_obstacle": 3, "double_paddle": 4, "sphere_bounce": 5,
}


def _safe_kinematic_sim(scenario_type, scenario_idx):
    """Driven scenarios: keep prescribed motion but raise center y where needed."""
    np.random.seed(scenario_idx * 1000 + SCENARIO_SEED_OFFSET[scenario_type] * 100)

    if scenario_type == "paddle_wave":
        n = 600
        fp = np.random.rand(n, 3) * [1.0, 0.5, 1.0]
        fv = np.zeros((n, 3))
        paddle = BoxRigidBody(
            name="paddle", size=(0.05, 0.3, 0.8),
            position_fn=oscillating_motion([0.2, 0.25, 0.5], [0.15, 0, 0], 1.0),
            velocity_fn=oscillating_velocity([0.15, 0, 0], 1.0),
        )
        return fp, fv, RigidBodySimulation([paddle])

    if scenario_type == "rotating_stirrer":
        n = 700
        fp = np.random.rand(n, 3) * [1.0, 0.6, 1.0]
        fv = np.zeros((n, 3))
        stirrer = CylinderRigidBody(
            name="stirrer", radius=0.08, height=0.5,
            position_fn=circular_motion([0.5, 0.3, 0.5], 0.25, 2.0, 'y'),
            velocity_fn=circular_velocity(0.25, 2.0, 'y'),
        )
        return fp, fv, RigidBodySimulation([stirrer])

    if scenario_type == "moving_obstacle":
        n = 500
        fp = np.random.rand(n, 3) * [1.0, 0.5, 1.0]
        fv = np.zeros((n, 3))
        obs = BoxRigidBody(
            name="obstacle", size=(0.2, 0.2, 0.2),
            position_fn=linear_motion([0.1, 0.25, 0.5], [0.5, 0, 0]),
            velocity_fn=lambda t: np.array([0.5, 0, 0]),
        )
        return fp, fv, RigidBodySimulation([obs])

    if scenario_type == "double_paddle":
        # v1: center y=0.25, amplitude 0.15 → bottom = -0.05  (5cm penetration)
        # v2: raise center y to 0.30, keep amplitude 0.15 → bottom = 0.0   (no penetration)
        n = 600
        fp = np.random.rand(n, 3) * [1.0, 0.5, 1.0]
        fv = np.zeros((n, 3))
        p1 = BoxRigidBody(
            name="paddle1", size=(0.05, 0.3, 0.6),
            position_fn=oscillating_motion([0.3, 0.30, 0.5], [0, 0.15, 0], 1.5),
            velocity_fn=oscillating_velocity([0, 0.15, 0], 1.5),
        )
        p2 = BoxRigidBody(
            name="paddle2", size=(0.05, 0.3, 0.6),
            position_fn=oscillating_motion([0.7, 0.30, 0.5], [0, -0.15, 0], 1.5),
            velocity_fn=oscillating_velocity([0, -0.15, 0], 1.5),
        )
        return fp, fv, RigidBodySimulation([p1, p2])

    raise ValueError(scenario_type)


def _dynamic_sim(scenario_type, scenario_idx):
    """Free-fall scenarios with gravity + collision + two-way coupling."""
    np.random.seed(scenario_idx * 1000 + SCENARIO_SEED_OFFSET[scenario_type] * 100)

    if scenario_type == "falling_ball":
        n = 800
        fp = np.random.rand(n, 3) * [1.0, 0.4, 1.0]
        fv = np.zeros((n, 3))
        ball = DynamicSphereRigidBody(
            name="ball", radius=0.15, mass=2.0,
            initial_position=np.array([0.5, 1.0, 0.5]),
            initial_velocity=np.array([0.0, 0.0, 0.0]),
            particle_spacing=0.05,
        )
        return fp, fv, DynamicRigidBodySimulation([ball])

    if scenario_type == "sphere_bounce":
        # v1: oscillating from y=0.6 ± 0.3 (often above water)
        # v2: drop with small upward impulse so ball bounces on water surface
        n = 600
        fp = np.random.rand(n, 3) * [1.0, 0.4, 1.0]
        fv = np.zeros((n, 3))
        ball = DynamicSphereRigidBody(
            name="ball", radius=0.12, mass=1.0,
            initial_position=np.array([0.5, 0.5, 0.5]),   # just above water surface
            initial_velocity=np.array([0.0, -0.5, 0.0]),  # gentle downward
            particle_spacing=0.05,
        )
        return fp, fv, DynamicRigidBodySimulation([ball])

    raise ValueError(scenario_type)


def make_scenario(scenario_type, scenario_idx):
    """Returns (fluid_pos, fluid_vel, dyn_or_kin_sim, is_dynamic)."""
    if scenario_type in ("falling_ball", "sphere_bounce"):
        fp, fv, sim = _dynamic_sim(scenario_type, scenario_idx)
        return fp, fv, sim, True
    fp, fv, sim = _safe_kinematic_sim(scenario_type, scenario_idx)
    return fp, fv, sim, False


# =============================================================================
# Simulation loop & saving
# =============================================================================
def simulate_one(scenario_type, scenario_idx, num_steps=80, dt=0.01):
    out_dir = OUTPUT_ROOT / f"sph_{scenario_type}_{scenario_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fluid_pos, fluid_vel, sim, is_dynamic = make_scenario(scenario_type, scenario_idx)
    simulator = SPHSimulatorV2(dt=dt)

    print(f"\n[{scenario_type}#{scenario_idx}] dynamic={is_dynamic}  "
          f"fluid={len(fluid_pos)}  output={out_dir.name}")

    t = 0.0
    for step in range(num_steps):
        if is_dynamic:
            d_pos, d_vel, d_ids = sim.get_all_boundary_particles()
            k_pos = k_vel = None
            boundary_ids = d_ids
            boundary_pos = d_pos
            boundary_vel = d_vel
        else:
            b_pos, b_vel, b_ids = sim.get_all_boundary_particles(t)
            # Floor safety clamp for kinematic bodies (defence in depth):
            #  any rigid particle that ended up < 0 in y is pulled up to 0.
            if len(b_pos):
                b_pos = b_pos.copy()
                b_pos[:, 1] = np.maximum(b_pos[:, 1], 0.0)
            k_pos = b_pos; k_vel = b_vel
            boundary_pos = b_pos; boundary_vel = b_vel; boundary_ids = b_ids

        # Save the current frame
        if len(boundary_pos):
            all_pos = np.vstack([fluid_pos, boundary_pos])
            all_vel = np.vstack([fluid_vel, boundary_vel])
            ptypes = np.concatenate([np.zeros(len(fluid_pos)), np.ones(len(boundary_pos))])
            body_ids_all = np.concatenate([
                -np.ones(len(fluid_pos), dtype=int),
                np.asarray(boundary_ids, dtype=int),
            ])
        else:
            all_pos, all_vel = fluid_pos, fluid_vel
            ptypes = np.zeros(len(fluid_pos))
            body_ids_all = -np.ones(len(fluid_pos), dtype=int)

        np.savez(
            out_dir / f"frame_{step:04d}.npz",
            positions=all_pos, velocities=all_vel,
            particle_types=ptypes, body_ids=body_ids_all,
            time=np.array(t), n_fluid=np.array(len(fluid_pos)),
            n_boundary=np.array(len(boundary_pos)),
        )

        # Step forward
        if is_dynamic:
            fluid_pos, fluid_vel = simulator.step(
                fluid_pos, fluid_vel, dyn_sim=sim,
            )
        else:
            fluid_pos, fluid_vel = simulator.step(
                fluid_pos, fluid_vel, dyn_sim=None,
                kin_boundary_pos=k_pos, kin_boundary_vel=k_vel,
            )
        t += dt

    print(f"  done -> {out_dir}")


SCENARIOS = [
    "falling_ball", "paddle_wave", "rotating_stirrer",
    "moving_obstacle", "double_paddle", "sphere_bounce",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--single", action="store_true",
                        help="quick test: one sim of one scenario only")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")

    t0 = time.time()
    if args.single:
        simulate_one(args.scenarios[0], 0, num_steps=args.num_steps, dt=args.dt)
    else:
        for sc in args.scenarios:
            for idx in range(args.n_runs):
                simulate_one(sc, idx, num_steps=args.num_steps, dt=args.dt)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
