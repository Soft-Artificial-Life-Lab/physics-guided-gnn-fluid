"""
100-step auto-regressive rollout validation for all 9 trained models
(GNN/GAT/Hybrid × Vanilla/Physics-aware/Improved).

For each (model, scenario):
  - Initialise from frame_0000.npz (positions, velocities)
  - Rollout 100 steps. At each step:
      * Build k-NN graph from current positions
      * Predict velocities
      * Integrate fluid positions   x_{t+1} = x_t + v̂·dt
      * Snap boundary state to ground truth (rigid bodies follow known motion)
  - Per step: log fluid vel-MSE, pos-MSE, kinetic-energy ratio, momentum-drift
              (vs ground truth, where available — first 79 steps).

Logs written to logs/rollout_100step.log (text) and
logs/rollout_100step_metrics.npz (numerical arrays per (model, scenario)).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.gnn import MeshGraphNet
from src.models.gat import GraphAttentionNet
from src.models.hybrid_gnn import HybridPhysicsGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.01
RADIUS = 0.3
NUM_STEPS = 100
EPS = 1e-8

CKPTS = {
    'GNN': {
        'Vanilla'      : 'models/gnn_vanilla_100ep_best.pt',
        'Physics-aware': 'models/gnn_rigid_body_sph_100ep_best.pt',
        'Improved'     : 'models/gnn_improved_100ep_best.pt',
    },
    'GAT': {
        'Vanilla'      : 'models/gat_vanilla_100ep_best.pt',
        'Physics-aware': 'models/gat_rigid_body_sph_100ep_best.pt',
        'Improved'     : 'models/gat_improved_100ep_best.pt',
    },
    'Hybrid': {
        'Vanilla'      : 'models/hybrid_vanilla_100ep_best.pt',
        'Physics-aware': 'models/hybrid_rigid_body_sph_100ep_best.pt',
        'Improved'     : 'models/hybrid_improved_100ep_best.pt',
    },
}

SCENARIOS = [
    'sph_falling_ball_000',
    'sph_paddle_wave_000',
    'sph_rotating_stirrer_000',
    'sph_moving_obstacle_000',
    'sph_double_paddle_000',
    'sph_sphere_bounce_000',
]


def build_model(arch: str):
    if arch == 'GNN':
        return MeshGraphNet(input_node_dim=7, input_edge_dim=3,
                            hidden_dim=64, output_dim=3, num_layers=5).to(DEVICE)
    if arch == 'GAT':
        return GraphAttentionNet(input_node_dim=7, input_edge_dim=3,
                                 hidden_dim=64, output_dim=3, num_layers=5,
                                 num_heads=4, dropout=0.1).to(DEVICE)
    if arch == 'Hybrid':
        return HybridPhysicsGNN(input_node_dim=7, input_edge_dim=3,
                                hidden_dim=64, output_dim=3, num_layers=5,
                                use_learned_attention=True, use_type_weighting=True,
                                boundary_importance=2.0).to(DEVICE)
    raise ValueError(arch)


def load_model(arch: str, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model(arch)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model


def build_graph(positions: np.ndarray, radius: float):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return np.array([[0], [0]]), np.array([[0.0, 0.0, 0.0]])
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    edge_features = positions[edges[1]] - positions[edges[0]]
    return edges, edge_features


def load_scenario(scenario_dir: Path, max_frames: int):
    files = sorted(scenario_dir.glob("frame_*.npz"))[:max_frames]
    frames = []
    for fp in files:
        d = np.load(fp)
        frames.append({
            'positions': d['positions'],
            'velocities': d['velocities'],
            'particle_types': d['particle_types'],
        })
    return frames


@torch.no_grad()
def rollout(model, frames, num_steps: int):
    """
    Auto-regressive rollout. Returns lists of per-step metrics.
    """
    pos = frames[0]['positions'].copy()
    vel = frames[0]['velocities'].copy()
    types = frames[0]['particle_types']
    is_fluid = types < 0.5

    metrics = {
        'vel_mse': [], 'pos_mse': [],
        'KE_ratio': [], 'momentum_drift': [],
        'gt_available': [],
    }

    for k in range(num_steps):
        # Build graph + predict
        edge_index, edge_attr = build_graph(pos, RADIUS)
        nf = np.concatenate([pos, vel, types[:, None]], axis=1)
        nf_t = torch.FloatTensor(nf).to(DEVICE)
        ei_t = torch.LongTensor(edge_index).to(DEVICE)
        ea_t = torch.FloatTensor(edge_attr).to(DEVICE)
        v_pred = model(nf_t, ei_t, ea_t).cpu().numpy()

        # Update fluid only; boundary snapped to GT if available
        new_vel = vel.copy()
        new_vel[is_fluid] = v_pred[is_fluid]
        new_pos = pos.copy()
        new_pos[is_fluid] = pos[is_fluid] + new_vel[is_fluid] * DT

        gt_idx = k + 1
        if gt_idx < len(frames):
            new_pos[~is_fluid] = frames[gt_idx]['positions'][~is_fluid]
            new_vel[~is_fluid] = frames[gt_idx]['velocities'][~is_fluid]
            gt_pos = frames[gt_idx]['positions'][is_fluid]
            gt_vel = frames[gt_idx]['velocities'][is_fluid]
            pred_vel_f = new_vel[is_fluid]
            pred_pos_f = new_pos[is_fluid]
            metrics['vel_mse'].append(float(np.mean((pred_vel_f - gt_vel) ** 2)))
            metrics['pos_mse'].append(float(np.mean((pred_pos_f - gt_pos) ** 2)))
            KE_pred = 0.5 * float(np.sum(pred_vel_f ** 2))
            KE_true = 0.5 * float(np.sum(gt_vel ** 2))
            metrics['KE_ratio'].append(KE_pred / (KE_true + EPS))
            mom_pred = pred_vel_f.sum(axis=0)
            mom_true = gt_vel.sum(axis=0)
            metrics['momentum_drift'].append(float(np.linalg.norm(mom_pred - mom_true)))
            metrics['gt_available'].append(True)
        else:
            # Beyond ground truth: still continue, log only intrinsic metrics
            pred_vel_f = new_vel[is_fluid]
            metrics['vel_mse'].append(float('nan'))
            metrics['pos_mse'].append(float('nan'))
            KE_pred = 0.5 * float(np.sum(pred_vel_f ** 2))
            metrics['KE_ratio'].append(float('nan'))
            metrics['momentum_drift'].append(float('nan'))
            metrics['gt_available'].append(False)

        pos = new_pos
        vel = new_vel

    return metrics


def main():
    log_path = ROOT / 'logs' / 'rollout_100step.log'
    npz_path = ROOT / 'logs' / 'rollout_100step_metrics.npz'
    log_path.parent.mkdir(exist_ok=True)

    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(msg)

    log('=' * 76)
    log('100-STEP AUTO-REGRESSIVE ROLLOUT VALIDATION')
    log('=' * 76)
    log(f'Device          : {DEVICE}')
    log(f'Steps per rollout: {NUM_STEPS}')
    log(f'Scenarios       : {len(SCENARIOS)}')
    log(f'Models          : 9 (3 architectures × 3 loss kinds)')
    log('=' * 76)

    all_metrics = {}  # key (arch, kind, scenario) -> metrics dict

    t0 = time.time()
    for arch, paths in CKPTS.items():
        for kind, ckpt_path in paths.items():
            log(f'\n--- Loading {arch} / {kind} ({ckpt_path}) ---')
            model = load_model(arch, str(ROOT / ckpt_path))
            for sc in SCENARIOS:
                sc_dir = ROOT / 'datasets' / 'flip_simulations' / sc
                frames = load_scenario(sc_dir, max_frames=NUM_STEPS + 1)
                gt_avail = min(len(frames) - 1, NUM_STEPS)
                t_run = time.time()
                metrics = rollout(model, frames, NUM_STEPS)
                dt_run = time.time() - t_run

                vm = np.array(metrics['vel_mse'])
                pm = np.array(metrics['pos_mse'])
                ke = np.array(metrics['KE_ratio'])
                md = np.array(metrics['momentum_drift'])

                # Aggregate (only over GT-available steps)
                vm_valid = vm[~np.isnan(vm)]
                pm_valid = pm[~np.isnan(pm)]
                ke_valid = ke[~np.isnan(ke)]
                md_valid = md[~np.isnan(md)]

                summary = (f'  {sc:30s} '
                           f'gt_steps={gt_avail:3d}  '
                           f'vel_mse_mean={vm_valid.mean():.6f}  '
                           f'pos_mse_mean={pm_valid.mean():.6f}  '
                           f'KE_ratio_last={ke_valid[-1]:.3f}  '
                           f'mom_drift_last={md_valid[-1]:.4f}  '
                           f'({dt_run:.1f}s)')
                log(summary)

                all_metrics[f'{arch}|{kind}|{sc}'] = metrics

    # Overall summary across scenarios per (arch, kind)
    log('\n' + '=' * 76)
    log('OVERALL  (mean over 6 scenarios, only GT-available steps)')
    log('=' * 76)
    log(f"{'Model':<8}{'Loss':<16}{'vel_mse':>12}{'pos_mse':>12}"
        f"{'|KE-1|_last':>14}{'mom_drift_last':>16}")
    log('-' * 76)
    for arch in CKPTS:
        for kind in CKPTS[arch]:
            vms, pms, kes, mds = [], [], [], []
            for sc in SCENARIOS:
                m = all_metrics[f'{arch}|{kind}|{sc}']
                vm = np.array(m['vel_mse'])
                pm = np.array(m['pos_mse'])
                ke = np.array(m['KE_ratio'])
                md = np.array(m['momentum_drift'])
                vms.append(np.nanmean(vm))
                pms.append(np.nanmean(pm))
                ke_v = ke[~np.isnan(ke)]
                md_v = md[~np.isnan(md)]
                kes.append(abs(ke_v[-1] - 1.0))
                mds.append(md_v[-1])
            log(f'{arch:<8}{kind:<16}{np.mean(vms):>12.6f}{np.mean(pms):>12.6f}'
                f'{np.mean(kes):>14.4f}{np.mean(mds):>16.4f}')

    elapsed = time.time() - t0
    log('\n' + '=' * 76)
    log(f'DONE in {elapsed:.1f}s')
    log(f'Text log : {log_path}')
    log(f'Numerical: {npz_path}')
    log('=' * 76)

    # Save text log
    log_path.write_text('\n'.join(log_lines), encoding='utf-8')

    # Save numerical metrics
    save_dict = {}
    for k, m in all_metrics.items():
        for kk, vv in m.items():
            save_dict[f'{k}|{kk}'] = np.array(vv)
    np.savez_compressed(npz_path, **save_dict)


if __name__ == '__main__':
    main()
