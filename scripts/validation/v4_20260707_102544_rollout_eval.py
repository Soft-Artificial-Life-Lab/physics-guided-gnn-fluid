"""
v4 autoregressive rollout evaluation — paper §3.6.2 protocol, v4 I/O.

Version ID : v4_20260707_102544

Runs an N-step autoregressive rollout of a HybridPhysicsGNNv4 checkpoint on
one or more sim directories and records the paper's four per-step metrics:

  vel_mse_k   = mean ||v_hat - v_GT||^2          (fluid only)
  pos_mse_k   = mean ||x_pred - x_GT||^2         (fluid only)
  KE_ratio_k  = KE_pred / (KE_true + eps)        (1.0 = conserved)
  mom_drift_k = ||p_hat - p_GT||                 (mean-velocity drift)

Protocol (identical to the v3p2 evaluation except for the model interface):
  - The first `history` frames are taken from GT to seed the velocity history
    (GNS convention), rollout starts at t = history-1.
  - Each step: build graph at current positions -> predict delta -> integrate
    fluid with Forward Euler -> snap rigid particles to their GT trajectory.

Usage:
  python scripts/validation/v4_20260707_102544_rollout_eval.py \
      --ckpt models/hybrid_v4_full_<ts>_v4_20260707_102544.pt \
      --scenarios all --steps 100
  # single scenario, save predicted trajectory for visualisation
  python scripts/validation/v4_20260707_102544_rollout_eval.py \
      --ckpt <ckpt.pt> --scenarios sph_falling_ball_000 --steps 100 --save-traj
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4, V4_VERSION_ID

DATASET_ID = "v3p2_20260506_043045"
DATA_DIR = ROOT / "datasets" / f"flip_simulations_{DATASET_ID}"
DT = 0.01
EPS = 1e-8


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = HybridPhysicsGNNv4(
        history=cfg["history"],
        input_edge_dim=cfg["input_edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        num_layers=cfg["num_layers"],
        use_learned_attention=cfg["use_learned_attention"],
        use_type_weighting=cfg["use_type_weighting"],
        boundary_importance=cfg["boundary_importance"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])   # weights + normaliser buffers
    model.eval()
    return model, ckpt


def load_frames(sim_dir):
    files = sorted(Path(sim_dir).glob("frame_*.npz"))
    pos, vel, types = [], [], None
    for f in files:
        d = np.load(f)
        pos.append(d["positions"].astype(np.float32))
        vel.append(d["velocities"].astype(np.float32))
        if types is None:
            types = d["particle_types"].astype(np.float32)
    return np.stack(pos), np.stack(vel), types


def build_graph(positions, radius):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    return torch.from_numpy(edges)


@torch.no_grad()
def rollout(model, pos_gt, vel_gt, types, steps, radius, device):
    """Returns per-step metric dict lists and the predicted trajectory."""
    C = model.history
    n_frames = pos_gt.shape[0]
    steps = min(steps, n_frames - C)          # need GT up to start+steps
    start = C - 1                              # last seeded frame index

    fluid = types < 0.5
    fluid_t = torch.from_numpy(fluid).to(device)
    types_t = torch.from_numpy(types).to(device)

    # Seed state from GT
    x = torch.from_numpy(pos_gt[start].copy()).to(device)
    hist = torch.from_numpy(vel_gt[:C].copy()).to(device)        # (C, N, 3)

    metrics = {"vel_mse": [], "pos_mse": [], "ke_ratio": [], "mom_drift": []}
    traj_pos, traj_vel = [x.cpu().numpy().copy()], [hist[-1].cpu().numpy().copy()]

    for k in range(steps):
        t_next = start + k + 1

        edge_index = build_graph(x.cpu().numpy(), radius).to(device)
        rel = x[edge_index[1]] - x[edge_index[0]]
        ea = torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)

        N = x.shape[0]
        nf = torch.cat(
            [hist.permute(1, 0, 2).reshape(N, 3 * C), types_t.unsqueeze(-1)],
            dim=-1)
        v_hat = model.predict_next_velocity(nf, edge_index, ea)

        # Integrate fluid, snap rigid to GT (paper protocol step 3-4)
        v_gt_next = torch.from_numpy(vel_gt[t_next]).to(device)
        x_gt_next = torch.from_numpy(pos_gt[t_next]).to(device)
        v_new = torch.where(fluid_t.unsqueeze(-1), v_hat, v_gt_next)
        x_new = torch.where(fluid_t.unsqueeze(-1), x + v_hat * DT, x_gt_next)

        # Metrics (fluid only)
        vf_hat, vf_gt = v_new[fluid_t], v_gt_next[fluid_t]
        xf_pred, xf_gt = x_new[fluid_t], x_gt_next[fluid_t]
        ke_pred = 0.5 * torch.sum(vf_hat ** 2)
        ke_true = 0.5 * torch.sum(vf_gt ** 2)
        metrics["vel_mse"].append(torch.mean((vf_hat - vf_gt) ** 2).item())
        metrics["pos_mse"].append(torch.mean((xf_pred - xf_gt) ** 2).item())
        metrics["ke_ratio"].append((ke_pred / (ke_true + EPS)).item())
        metrics["mom_drift"].append(
            (vf_hat.mean(0) - vf_gt.mean(0)).norm().item())

        # Advance state
        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new
        traj_pos.append(x.cpu().numpy().copy())
        traj_vel.append(v_new.cpu().numpy().copy())

    return metrics, np.stack(traj_pos), np.stack(traj_vel), steps


def main():
    p = argparse.ArgumentParser(description="v4 rollout evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scenarios", type=str, default="all",
                   help="'all', a comma list of sim dir names, or 'one_per_type'")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--save-traj", action="store_true",
                   help="save predicted trajectories as .npz next to the report")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    print(f"Loaded {args.ckpt}")
    print(f"  version={ckpt.get('version')}  epoch={ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss'):.5f}  params={model.count_parameters():,}")

    data_root = Path(args.data_dir)
    all_dirs = sorted(d for d in data_root.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    if args.scenarios == "all":
        sim_dirs = all_dirs
    elif args.scenarios == "one_per_type":
        seen, sim_dirs = set(), []
        for d in all_dirs:
            key = d.name.rsplit("_", 1)[0]
            if key not in seen:
                seen.add(key)
                sim_dirs.append(d)
    else:
        names = {s.strip() for s in args.scenarios.split(",")}
        sim_dirs = [d for d in all_dirs if d.name in names]
    print(f"Evaluating {len(sim_dirs)} scenario(s), {args.steps} steps\n")

    out_dir = ROOT / "logs" / V4_VERSION_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).stem

    report = {"ckpt": str(args.ckpt), "version": V4_VERSION_ID,
              "steps": args.steps, "scenarios": {}}

    for sd in sim_dirs:
        pos_gt, vel_gt, types = load_frames(sd)
        t0 = time.time()
        metrics, traj_pos, traj_vel, n_done = rollout(
            model, pos_gt, vel_gt, types, args.steps, args.radius, device)
        last = {k: v[-1] for k, v in metrics.items()}
        report["scenarios"][sd.name] = {
            "steps_done": n_done,
            "per_step": metrics,
            "final": last,
            "final_ke_dev": abs(last["ke_ratio"] - 1.0),
        }
        print(f"  {sd.name:<28} ({n_done} steps, {time.time() - t0:.1f}s)  "
              f"vel_mse={last['vel_mse']:.5f}  pos_mse={last['pos_mse']:.5f}  "
              f"|KE-1|={abs(last['ke_ratio'] - 1.0):.4f}  "
              f"mom_drift={last['mom_drift']:.5f}")

        if args.save_traj:
            np.savez_compressed(
                out_dir / f"rollout_{tag}_{sd.name}.npz",
                positions=traj_pos, velocities=traj_vel,
                particle_types=types)

    # Aggregate
    finals = [s["final"] for s in report["scenarios"].values()]
    if finals:
        agg = {k: float(np.mean([f[k] for f in finals])) for k in finals[0]}
        agg["ke_dev"] = float(np.mean(
            [abs(f["ke_ratio"] - 1.0) for f in finals]))
        report["aggregate_final"] = agg
        print(f"\nAGGREGATE (mean of final step over {len(finals)} scenarios)")
        print(f"  vel_mse={agg['vel_mse']:.5f}  pos_mse={agg['pos_mse']:.5f}  "
              f"|KE_ratio-1|={agg['ke_dev']:.4f}  mom_drift={agg['mom_drift']:.5f}")

    report_path = out_dir / f"rollout_report_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
