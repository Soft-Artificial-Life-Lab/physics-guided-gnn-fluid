"""
v3.2 SPH data generator — single-way coupling (rigid -> fluid only).

Background:
  v3.1 fixed fluid pressure (mass=0.10, K=150, rest_density=400, no
  density clamp, pressure clamped to non-negative).  The fluid behaved
  correctly (no collapse, splash, sloshing).
  However the v2-style two-way coupling — aggregating SPH pressure
  forces from rigid surface particles back to the rigid body's CoM —
  produced unbounded ball velocities (10,000+ m/s, ball teleporting
  between domain corners).  Root cause: SPH pressure forces designed
  for fluid-fluid pairs saturate when applied to sparser rigid surface
  particles, and aggregating ~500 pairs onto a single rigid body gives
  net forces ~10⁶ N.

v3.2 fix (this file):
  Remove the fluid -> rigid back-reaction loop.  Rigid bodies still:
    - feel gravity (DynamicSphereRigidBody.update applies it)
    - bounce off floor / walls (apply_boundary_constraint, restitution=0.5)
    - act as boundary particles for the SPH fluid (rigid -> fluid)
  Fluid no longer applies any reaction force to rigid (single-way).

  This matches:
    - 3d_fluid AGENTS.md policy: "단일 방향 상호작용을 유지합니다"
    - The paper's wording (no explicit two-way claim, all metrics fluid-only)
    - Pfaff 2021 / Sanchez-Gonzalez 2020 literature standard

  Fluid SPH params (mass / K / rest_density / no clamp + pressure clamp)
  are unchanged from v3.1.

Output format / rigid-body class / architecture / preprocessing:
  UNCHANGED.

Output path:
  datasets/flip_simulations_v3p2_20260506_043045/sph_<scenario>_<idx>/frame_*.npz

The v1, v2, v3, v3.1 generators and datasets are NOT touched.
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


TIMESTAMP = "20260506_043045"
OUTPUT_ROOT = ROOT / "datasets" / f"flip_simulations_v3p2_{TIMESTAMP}"
# v3.2 = v3.1 fluid params (mass/K/rest unchanged) + single-way coupling
PARTICLE_MASS = 0.10
GAS_CONSTANT  = 150.0
REST_DENSITY  = 400.0


# =============================================================================
# SPH simulator with two-way fluid<->rigid coupling
# =============================================================================
class SPHSimulatorV3p2:
    """Single-way SPH (rigid -> fluid) with pressure-clamped EoS."""

    def __init__(self, domain_size=(1.0, 1.0, 1.0), particle_mass=PARTICLE_MASS,
                 rest_density=REST_DENSITY, gas_constant=GAS_CONSTANT, viscosity=0.1,
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
        # v3.1: NO clamp on density.  Honest measured density.
        # Tiny floor (1e-3) just to avoid division by zero downstream.
        n = len(positions)
        densities = np.zeros(n)
        tree = KDTree(positions)
        nbrs = tree.query_ball_tree(tree, self.h)
        for i in range(n):
            d = 0.0
            for j in nbrs[i]:
                r2 = float(np.dot(positions[i] - positions[j], positions[i] - positions[j]))
                d += self.particle_mass * self.poly6(r2)
            densities[i] = max(d, 1e-3)
        return densities, nbrs

    def compute_forces_one_way(self, positions, velocities, densities, neighbors,
                               n_fluid, gravity_on_fluid=True):
        """Compute SPH forces ONLY on fluid particles (single-way coupling).

        For each fluid particle i:
          - sum pressure contributions from fluid AND rigid neighbors j
            (rigid acts as boundary)
          - sum viscosity contributions from fluid neighbors only
          - add gravity term

        Rigid particles get zero force here; their motion is handled by the
        DynamicRigidBody class (gravity + floor/wall collision).
        """
        n = len(positions)
        forces = np.zeros((n, 3))
        for i in range(n_fluid):           # fluid only — skip rigid surface particles
            pforce = np.zeros(3)
            vforce = np.zeros(3)
            for j in neighbors[i]:
                if j == i:
                    continue
                rvec = positions[i] - positions[j]
                # pressure clamped to non-negative (free-surface friendly)
                pressure_i = max(0.0, self.gas_constant * (densities[i] - self.rest_density))
                pressure_j = max(0.0, self.gas_constant * (densities[j] - self.rest_density))
                avg_p = (pressure_i + pressure_j) * 0.5
                pforce += -self.particle_mass * avg_p / densities[j] * self.spiky_grad(rvec)
                # viscosity contributes only between fluid pairs
                if j < n_fluid:
                    r = float(np.linalg.norm(rvec))
                    vforce += (self.viscosity * self.particle_mass
                               * (velocities[j] - velocities[i]) / densities[j]
                               * self.visc_lap(r))
            f = pforce + vforce
            if gravity_on_fluid:
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
        forces = self.compute_forces_one_way(all_pos, all_vel, densities, nbrs, n_fluid)

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

        # v3.2: SINGLE-WAY coupling.
        # Rigid bodies are advanced with gravity + boundary collision only.
        # No reaction force from fluid is applied to rigid (matches paper /
        # 3d_fluid AGENTS.md / Pfaff 2021 / Sanchez-Gonzalez 2020).
        if dyn_sim is not None:
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
    simulator = SPHSimulatorV3p2(dt=dt)

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
