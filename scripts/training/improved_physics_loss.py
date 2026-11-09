"""
Improved Physics-Aware Loss for Rigid-Body SPH Fluid Prediction.

Fixes three issues identified in RigidBodyPhysicsLoss (train_hybrid.py):

1. pos_loss scale bug
   Original : mean((x_t + v̂·dt - x_{t+1})²)  = dt² × (velocity error)²
              With dt=0.01 this made pos_loss ≈ 10⁻⁴ × vel_loss → pos_weight=0.1 gave
              ~10⁻⁵ effective contribution. Practically zero.
   Fixed    : mean((v̂ - Δx_true/dt)²) — compared at velocity scale, same magnitude as vel_loss.

2. momentum_loss is now relative (normalized by true momentum magnitude)
   Original : mean((p̂_mean - p_mean)²)  — dominated by noise when ground truth mean ≈ 0.
   Fixed    : ‖p̂ - p_true‖² / (‖p_true‖² + ε)  — dimensionless conservation ratio.

3. Added energy conservation (kinetic energy)
   New term : |KE_pred - KE_true| / (KE_true + ε)   (dimensionless)

All three physics terms apply to fluid particles only (particle_types < 0.5).
"""
import torch
import torch.nn as nn


class ImprovedPhysicsLoss(nn.Module):
    def __init__(self,
                 vel_weight: float = 1.0,
                 pos_weight: float = 0.1,
                 momentum_weight: float = 0.05,
                 energy_weight: float = 0.05,
                 dt: float = 0.01,
                 eps: float = 1e-8):
        super().__init__()
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.momentum_weight = momentum_weight
        self.energy_weight = energy_weight
        self.dt = dt
        self.eps = eps

    def forward(self, pred_vel, true_vel, pos_t, pos_t1, particle_types,
                particle_mass: float = 1.0):
        is_fluid = particle_types < 0.5

        v_hat = pred_vel[is_fluid]
        v_true = true_vel[is_fluid]
        x_t = pos_t[is_fluid]
        x_t1 = pos_t1[is_fluid]

        # 1. Velocity MSE — standard supervised signal
        vel_loss = torch.mean((v_hat - v_true) ** 2)

        # 2. Position-consistency at velocity scale
        #    Δx_true/dt is the average velocity implied by the true trajectory
        #    over [t, t+dt]. Comparing v_hat to that gives a true kinematic error.
        dx_true = x_t1 - x_t
        implied_vel = dx_true / self.dt
        pos_loss = torch.mean((v_hat - implied_vel) ** 2)

        # 3. Momentum conservation, normalised by velocity scale (stable)
        #    Original ‖p̂-p_true‖²/(‖p_true‖²+ε) blows up when mean velocity
        #    ≈ 0 (denominator near ε). Use mean‖v_true‖² as a denominator that
        #    is always meaningful, and clamp to keep early-training stable.
        n_fluid = v_hat.shape[0]
        if n_fluid > 0:
            p_hat = v_hat.sum(dim=0) / n_fluid
            p_true = v_true.sum(dim=0) / n_fluid
            v_scale = torch.mean(v_true ** 2) + self.eps
            momentum_loss = torch.clamp(
                torch.sum((p_hat - p_true) ** 2) / v_scale,
                max=10.0,
            )
        else:
            momentum_loss = torch.tensor(0.0, device=pred_vel.device)

        # 4. Relative kinetic-energy conservation (mass=1 per particle)
        KE_hat = 0.5 * particle_mass * torch.sum(v_hat ** 2)
        KE_true = 0.5 * particle_mass * torch.sum(v_true ** 2)
        energy_loss = torch.clamp(
            torch.abs(KE_hat - KE_true) / (KE_true + self.eps),
            max=10.0,
        )

        total = (self.vel_weight * vel_loss
                 + self.pos_weight * pos_loss
                 + self.momentum_weight * momentum_loss
                 + self.energy_weight * energy_loss)

        return total, {
            'vel_loss': vel_loss.item(),
            'pos_loss': pos_loss.item(),
            'momentum_loss': momentum_loss.item(),
            'energy_loss': energy_loss.item(),
        }
