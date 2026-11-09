"""
FLIP (Fluid-Implicit Particle) Simulator with Rigid Body Coupling
Uses grid-based pressure projection for incompressible fluid simulation
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class FLIPSimulator:
    """
    FLIP simulator with:
    - Grid-based pressure projection (incompressibility)
    - Particle advection
    - PIC/FLIP blending
    - Rigid body coupling
    """

    def __init__(self,
                 domain_size=(1.0, 1.0, 1.0),
                 grid_resolution=(32, 32, 32),
                 dt=0.01,
                 gravity=-9.81,
                 pic_flip_ratio=0.05,
                 density=1000.0):
        """
        Args:
            domain_size: Physical size of simulation domain
            grid_resolution: Number of grid cells in each dimension
            dt: Time step
            gravity: Gravity acceleration (Y direction)
            pic_flip_ratio: Blending ratio (0=pure FLIP, 1=pure PIC)
            density: Fluid density (kg/m^3)
        """
        self.domain_size = np.array(domain_size)
        self.grid_res = np.array(grid_resolution)
        self.dt = dt
        self.gravity = np.array([0, gravity, 0])
        self.pic_flip_ratio = pic_flip_ratio
        self.density = density

        # Grid spacing
        self.dx = self.domain_size / self.grid_res

        # MAC (Marker-And-Cell) grid for velocities
        # u: x-velocity at cell faces (grid_res[0]+1, grid_res[1], grid_res[2])
        # v: y-velocity at cell faces (grid_res[0], grid_res[1]+1, grid_res[2])
        # w: z-velocity at cell faces (grid_res[0], grid_res[1], grid_res[2]+1)
        self.u = np.zeros((self.grid_res[0]+1, self.grid_res[1], self.grid_res[2]))
        self.v = np.zeros((self.grid_res[0], self.grid_res[1]+1, self.grid_res[2]))
        self.w = np.zeros((self.grid_res[0], self.grid_res[1], self.grid_res[2]+1))

        # Old velocities for FLIP
        self.u_old = np.zeros_like(self.u)
        self.v_old = np.zeros_like(self.v)
        self.w_old = np.zeros_like(self.w)

        # Cell types: 0=air, 1=fluid, 2=solid
        self.cell_type = np.zeros(self.grid_res, dtype=int)

        # Pressure
        self.pressure = np.zeros(self.grid_res)

    def particles_to_grid(self, positions, velocities):
        """Transfer particle velocities to MAC grid"""
        # Reset grids
        self.u_old = self.u.copy()
        self.v_old = self.v.copy()
        self.w_old = self.w.copy()

        self.u.fill(0)
        self.v.fill(0)
        self.w.fill(0)

        # Weight grids
        u_weight = np.zeros_like(self.u)
        v_weight = np.zeros_like(self.v)
        w_weight = np.zeros_like(self.w)

        # Cell type: fluid cells containing particles
        self.cell_type.fill(0)

        for pos, vel in zip(positions, velocities):
            # Grid indices
            gi = np.floor(pos / self.dx).astype(int)
            gi = np.clip(gi, 0, self.grid_res - 1)

            # Mark cell as fluid
            self.cell_type[gi[0], gi[1], gi[2]] = 1

            # Transfer to u-grid (x-velocity)
            u_pos = pos / self.dx - np.array([0.0, 0.5, 0.5])
            u_i = np.floor(u_pos).astype(int)
            u_i = np.clip(u_i, [0, 0, 0], [self.grid_res[0], self.grid_res[1]-1, self.grid_res[2]-1])
            u_frac = u_pos - u_i

            # Trilinear weights
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (u_i[0]+di < self.u.shape[0] and
                            u_i[1]+dj < self.u.shape[1] and
                            u_i[2]+dk < self.u.shape[2]):
                            w = ((1-di) + di*u_frac[0]) * \
                                ((1-dj) + dj*u_frac[1]) * \
                                ((1-dk) + dk*u_frac[2]) if di+dj+dk > 0 else \
                                (1-u_frac[0]) * (1-u_frac[1]) * (1-u_frac[2])
                            self.u[u_i[0]+di, u_i[1]+dj, u_i[2]+dk] += w * vel[0]
                            u_weight[u_i[0]+di, u_i[1]+dj, u_i[2]+dk] += w

            # Transfer to v-grid (y-velocity)
            v_pos = pos / self.dx - np.array([0.5, 0.0, 0.5])
            v_i = np.floor(v_pos).astype(int)
            v_i = np.clip(v_i, [0, 0, 0], [self.grid_res[0]-1, self.grid_res[1], self.grid_res[2]-1])
            v_frac = v_pos - v_i

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (v_i[0]+di < self.v.shape[0] and
                            v_i[1]+dj < self.v.shape[1] and
                            v_i[2]+dk < self.v.shape[2]):
                            w = ((1-di) + di*v_frac[0]) * \
                                ((1-dj) + dj*v_frac[1]) * \
                                ((1-dk) + dk*v_frac[2]) if di+dj+dk > 0 else \
                                (1-v_frac[0]) * (1-v_frac[1]) * (1-v_frac[2])
                            self.v[v_i[0]+di, v_i[1]+dj, v_i[2]+dk] += w * vel[1]
                            v_weight[v_i[0]+di, v_i[1]+dj, v_i[2]+dk] += w

            # Transfer to w-grid (z-velocity)
            w_pos = pos / self.dx - np.array([0.5, 0.5, 0.0])
            w_i = np.floor(w_pos).astype(int)
            w_i = np.clip(w_i, [0, 0, 0], [self.grid_res[0]-1, self.grid_res[1]-1, self.grid_res[2]])
            w_frac = w_pos - w_i

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (w_i[0]+di < self.w.shape[0] and
                            w_i[1]+dj < self.w.shape[1] and
                            w_i[2]+dk < self.w.shape[2]):
                            w_val = ((1-di) + di*w_frac[0]) * \
                                    ((1-dj) + dj*w_frac[1]) * \
                                    ((1-dk) + dk*w_frac[2]) if di+dj+dk > 0 else \
                                    (1-w_frac[0]) * (1-w_frac[1]) * (1-w_frac[2])
                            self.w[w_i[0]+di, w_i[1]+dj, w_i[2]+dk] += w_val * vel[2]
                            w_weight[w_i[0]+di, w_i[1]+dj, w_i[2]+dk] += w_val

        # Normalize by weights
        mask = u_weight > 1e-6
        self.u[mask] /= u_weight[mask]
        mask = v_weight > 1e-6
        self.v[mask] /= v_weight[mask]
        mask = w_weight > 1e-6
        self.w[mask] /= w_weight[mask]

    def apply_gravity(self):
        """Apply gravity to v-grid (y-velocity)"""
        self.v += self.gravity[1] * self.dt

    def apply_boundary_conditions(self):
        """Set boundary velocities to zero"""
        self.u[0, :, :] = 0
        self.u[-1, :, :] = 0
        self.v[:, 0, :] = 0
        self.v[:, -1, :] = 0
        self.w[:, :, 0] = 0
        self.w[:, :, -1] = 0

    def pressure_projection(self):
        """Solve for pressure and make velocity divergence-free"""
        # Count fluid cells
        fluid_cells = np.where(self.cell_type == 1)
        n_fluid = len(fluid_cells[0])

        if n_fluid == 0:
            return

        # Build linear system: Ap = -div(u)
        # Create cell index mapping
        cell_index = np.full(self.grid_res, -1, dtype=int)
        for idx, (i, j, k) in enumerate(zip(*fluid_cells)):
            cell_index[i, j, k] = idx

        A = lil_matrix((n_fluid, n_fluid))
        b = np.zeros(n_fluid)

        scale = self.dt / (self.density * self.dx[0]**2)

        for idx, (i, j, k) in enumerate(zip(*fluid_cells)):
            # Divergence
            div = (self.u[i+1, j, k] - self.u[i, j, k]) / self.dx[0] + \
                  (self.v[i, j+1, k] - self.v[i, j, k]) / self.dx[1] + \
                  (self.w[i, j, k+1] - self.w[i, j, k]) / self.dx[2]
            b[idx] = -div / scale

            # Laplacian stencil
            diag = 0

            # -X neighbor
            if i > 0 and self.cell_type[i-1, j, k] == 1:
                A[idx, cell_index[i-1, j, k]] = 1
                diag += 1
            elif i > 0:
                diag += 1  # Solid boundary

            # +X neighbor
            if i < self.grid_res[0]-1 and self.cell_type[i+1, j, k] == 1:
                A[idx, cell_index[i+1, j, k]] = 1
                diag += 1
            elif i < self.grid_res[0]-1:
                diag += 1

            # -Y neighbor
            if j > 0 and self.cell_type[i, j-1, k] == 1:
                A[idx, cell_index[i, j-1, k]] = 1
                diag += 1
            elif j > 0:
                diag += 1

            # +Y neighbor
            if j < self.grid_res[1]-1 and self.cell_type[i, j+1, k] == 1:
                A[idx, cell_index[i, j+1, k]] = 1
                diag += 1
            elif j < self.grid_res[1]-1:
                diag += 1

            # -Z neighbor
            if k > 0 and self.cell_type[i, j, k-1] == 1:
                A[idx, cell_index[i, j, k-1]] = 1
                diag += 1
            elif k > 0:
                diag += 1

            # +Z neighbor
            if k < self.grid_res[2]-1 and self.cell_type[i, j, k+1] == 1:
                A[idx, cell_index[i, j, k+1]] = 1
                diag += 1
            elif k < self.grid_res[2]-1:
                diag += 1

            A[idx, idx] = -diag

        # Solve system
        A_csr = A.tocsr()
        pressure_solution = spsolve(A_csr, b)

        # Update pressure field
        self.pressure.fill(0)
        for idx, (i, j, k) in enumerate(zip(*fluid_cells)):
            self.pressure[i, j, k] = pressure_solution[idx]

        # Apply pressure gradient to velocities
        for i, j, k in zip(*fluid_cells):
            # Update u
            if i > 0:
                self.u[i, j, k] -= scale * (self.pressure[i, j, k] -
                                            (self.pressure[i-1, j, k] if self.cell_type[i-1, j, k] == 1 else 0))
            if i < self.grid_res[0]:
                self.u[i+1, j, k] += scale * (self.pressure[i, j, k] -
                                              (self.pressure[i+1, j, k] if i+1 < self.grid_res[0] and self.cell_type[i+1, j, k] == 1 else 0))

            # Update v
            if j > 0:
                self.v[i, j, k] -= scale * (self.pressure[i, j, k] -
                                            (self.pressure[i, j-1, k] if self.cell_type[i, j-1, k] == 1 else 0))
            if j < self.grid_res[1]:
                self.v[i, j+1, k] += scale * (self.pressure[i, j, k] -
                                              (self.pressure[i, j+1, k] if j+1 < self.grid_res[1] and self.cell_type[i, j+1, k] == 1 else 0))

            # Update w
            if k > 0:
                self.w[i, j, k] -= scale * (self.pressure[i, j, k] -
                                            (self.pressure[i, j, k-1] if self.cell_type[i, j, k-1] == 1 else 0))
            if k < self.grid_res[2]:
                self.w[i, j, k+1] += scale * (self.pressure[i, j, k] -
                                              (self.pressure[i, j, k+1] if k+1 < self.grid_res[2] and self.cell_type[i, j, k+1] == 1 else 0))

    def grid_to_particles(self, positions, velocities):
        """Transfer velocities from grid to particles (PIC/FLIP blending)"""
        new_velocities = np.zeros_like(velocities)

        for idx, pos in enumerate(positions):
            # Interpolate velocity from grid (PIC)
            vel_pic = self.interpolate_velocity(pos)

            # Interpolate velocity change (FLIP)
            vel_old = self.interpolate_velocity_old(pos)
            vel_new = self.interpolate_velocity(pos)
            vel_flip = velocities[idx] + (vel_new - vel_old)

            # Blend PIC and FLIP
            new_velocities[idx] = (self.pic_flip_ratio * vel_pic +
                                  (1 - self.pic_flip_ratio) * vel_flip)

        return new_velocities

    def interpolate_velocity(self, pos):
        """Interpolate velocity at particle position"""
        vel = np.zeros(3)

        # U component
        u_pos = pos / self.dx - np.array([0.0, 0.5, 0.5])
        vel[0] = self._trilinear_interpolate(self.u, u_pos)

        # V component
        v_pos = pos / self.dx - np.array([0.5, 0.0, 0.5])
        vel[1] = self._trilinear_interpolate(self.v, v_pos)

        # W component
        w_pos = pos / self.dx - np.array([0.5, 0.5, 0.0])
        vel[2] = self._trilinear_interpolate(self.w, w_pos)

        return vel

    def interpolate_velocity_old(self, pos):
        """Interpolate old velocity at particle position"""
        vel = np.zeros(3)

        u_pos = pos / self.dx - np.array([0.0, 0.5, 0.5])
        vel[0] = self._trilinear_interpolate(self.u_old, u_pos)

        v_pos = pos / self.dx - np.array([0.5, 0.0, 0.5])
        vel[1] = self._trilinear_interpolate(self.v_old, v_pos)

        w_pos = pos / self.dx - np.array([0.5, 0.5, 0.0])
        vel[2] = self._trilinear_interpolate(self.w_old, w_pos)

        return vel

    def _trilinear_interpolate(self, grid, pos):
        """Trilinear interpolation"""
        i = np.floor(pos).astype(int)
        f = pos - i

        # Clamp indices
        i = np.clip(i, [0, 0, 0], np.array(grid.shape) - 2)

        # 8 corner values
        c000 = grid[i[0], i[1], i[2]]
        c100 = grid[i[0]+1, i[1], i[2]]
        c010 = grid[i[0], i[1]+1, i[2]]
        c110 = grid[i[0]+1, i[1]+1, i[2]]
        c001 = grid[i[0], i[1], i[2]+1]
        c101 = grid[i[0]+1, i[1], i[2]+1]
        c011 = grid[i[0], i[1]+1, i[2]+1]
        c111 = grid[i[0]+1, i[1]+1, i[2]+1]

        # Interpolate
        c00 = c000 * (1-f[0]) + c100 * f[0]
        c01 = c001 * (1-f[0]) + c101 * f[0]
        c10 = c010 * (1-f[0]) + c110 * f[0]
        c11 = c011 * (1-f[0]) + c111 * f[0]

        c0 = c00 * (1-f[1]) + c10 * f[1]
        c1 = c01 * (1-f[1]) + c11 * f[1]

        return c0 * (1-f[2]) + c1 * f[2]

    def advect_particles(self, positions, velocities):
        """Advect particles with velocities and enforce boundary conditions"""
        new_positions = positions + velocities * self.dt

        # Boundary conditions: keep particles strictly inside [eps, domain_size - eps]
        # This prevents particles from getting stuck exactly at boundaries
        eps = 1e-4
        for dim in range(3):
            # Clamp to safe interior region
            new_positions[:, dim] = np.clip(new_positions[:, dim], eps, self.domain_size[dim] - eps)

        return new_positions

    def step(self, positions, velocities):
        """
        Single FLIP simulation step

        Returns:
            new_positions, new_velocities
        """
        # 1. Particles to grid
        self.particles_to_grid(positions, velocities)

        # 2. Apply forces (gravity)
        self.apply_gravity()

        # 3. Boundary conditions
        self.apply_boundary_conditions()

        # 4. Pressure projection (incompressibility)
        self.pressure_projection()

        # 5. Grid to particles
        new_velocities = self.grid_to_particles(positions, velocities)

        # 6. Advect particles
        new_positions = self.advect_particles(positions, new_velocities)

        return new_positions, new_velocities
