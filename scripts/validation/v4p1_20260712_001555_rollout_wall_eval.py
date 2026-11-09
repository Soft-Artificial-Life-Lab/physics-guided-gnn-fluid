"""
v4.1 rollout evaluation — wall-boundary models, paper metrics + BC compliance.

Version ID : v4p1_20260712_001555

Rollout protocol identical to v4 (§3.6.2) except every frame is augmented with
the same static wall particles used in training (type 1, zero velocity,
snapped like rigid bodies) and the graph excludes wall-wall edges.

Reports, aggregated over reps x scenarios (final step + per-step curves):
  - vel_mse / pos_mse / |KE_ratio-1| / mom_drift   (fluid only, vs GT)
  - domain violation fraction (fluid outside the GT bbox + tol) — the metric
    v4.1 is designed to fix (v4 full100: ~7.9%, backbone: ~11.6% at end).

Usage:
  python scripts/validation/v4p1_20260712_001555_rollout_wall_eval.py \
      --tags wall_hybrid wall_backbone --reps 3 --steps 100
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))
sys.path.append(str(ROOT / "scripts" / "validation"))

from v4_20260707_102544_rollout_eval import DATA_DIR, DT, load_frames
from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

v4p1_train = importlib.import_module("v4p1_20260712_001555_train_wall")
V4P1_VERSION_ID = v4p1_train.V4P1_VERSION_ID
EPS = 1e-8


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in ckpt["model_config"].items() if k != "version"}
    model = HybridPhysicsGNNv4(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_graph_no_wallwall(positions, radius, n_orig):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) > 0:
        keep = ~((pairs[:, 0] >= n_orig) & (pairs[:, 1] >= n_orig))
        pairs = pairs[keep]
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    return torch.from_numpy(edges)


@torch.no_grad()
def rollout(model, pos_gt, vel_gt, types_orig, walls, steps, radius, device,
            domain_tol=0.02):
    C = model.history
    steps = min(steps, pos_gt.shape[0] - C)
    start = C - 1
    n_orig = pos_gt.shape[1]
    n_w = len(walls)

    types = np.concatenate([types_orig, np.ones(n_w, np.float32)])
    fluid = types < 0.5
    fluid_t = torch.from_numpy(fluid).to(device)
    types_t = torch.from_numpy(types).to(device)
    walls_t = torch.from_numpy(walls).to(device)
    bbox_lo = pos_gt.reshape(-1, 3).min(axis=0)
    bbox_hi = pos_gt.reshape(-1, 3).max(axis=0)

    def aug_pos(p):        # (N_orig,3) GT frame -> with walls
        return torch.cat([torch.from_numpy(p).to(device), walls_t], dim=0)

    def aug_vel(v):
        return torch.cat([torch.from_numpy(v).to(device),
                          torch.zeros(n_w, 3, device=device)], dim=0)

    x = aug_pos(pos_gt[start].copy())
    hist = torch.stack([aug_vel(vel_gt[i].copy()) for i in range(C)], dim=0)

    metrics = {k: [] for k in ("vel_mse", "pos_mse", "ke_ratio", "mom_drift",
                               "domain_frac")}
    for k in range(steps):
        t_next = start + k + 1
        edge_index = build_graph_no_wallwall(
            x.cpu().numpy(), radius, n_orig).to(device)
        rel = x[edge_index[1]] - x[edge_index[0]]
        ea = torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)
        N = x.shape[0]
        nf = torch.cat(
            [hist.permute(1, 0, 2).reshape(N, 3 * C), types_t.unsqueeze(-1)],
            dim=-1)
        v_hat = model.predict_next_velocity(nf, edge_index, ea)

        v_gt_next = aug_vel(vel_gt[t_next])
        x_gt_next = aug_pos(pos_gt[t_next])
        v_new = torch.where(fluid_t.unsqueeze(-1), v_hat, v_gt_next)
        x_new = torch.where(fluid_t.unsqueeze(-1), x + v_hat * DT, x_gt_next)

        vf_hat, vf_gt = v_new[fluid_t], v_gt_next[fluid_t]
        xf_pred, xf_gt = x_new[fluid_t], x_gt_next[fluid_t]
        ke_pred = 0.5 * torch.sum(vf_hat ** 2)
        ke_true = 0.5 * torch.sum(vf_gt ** 2)
        metrics["vel_mse"].append(torch.mean((vf_hat - vf_gt) ** 2).item())
        metrics["pos_mse"].append(torch.mean((xf_pred - xf_gt) ** 2).item())
        metrics["ke_ratio"].append((ke_pred / (ke_true + EPS)).item())
        metrics["mom_drift"].append(
            (vf_hat.mean(0) - vf_gt.mean(0)).norm().item())

        fp = xf_pred.cpu().numpy()
        outside = ((fp < bbox_lo - domain_tol) |
                   (fp > bbox_hi + domain_tol)).any(axis=1)
        metrics["domain_frac"].append(float(outside.mean()))

        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new
    return metrics, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--wall-spacing", type=float, default=0.1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_dirs = sorted(d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    gt = {sd.name: load_frames(sd) for sd in sim_dirs}
    walls = {sd.name: v4p1_train.make_wall_particles(sd, spacing=args.wall_spacing)
             for sd in sim_dirs}

    out = {"version": V4P1_VERSION_ID, "reps": args.reps, "models": {}}
    for tag in args.tags:
        matches = sorted((ROOT / "models").glob(
            f"hybrid_{tag}_*_{V4P1_VERSION_ID}.pt"))
        if not matches:
            print(f"[skip] {tag}")
            continue
        model, meta = load_model(matches[-1], device)
        t0 = time.time()

        rep_aggs, curves = [], []
        for rep in range(args.reps):
            finals, curve = [], []
            for sd in sim_dirs:
                pos_gt, vel_gt, types = gt[sd.name]
                m, _ = rollout(model, pos_gt, vel_gt, types, walls[sd.name],
                               args.steps, args.radius, device)
                finals.append({k: v[-1] for k, v in m.items()})
                curve.append(m)
            rep_aggs.append({
                "vel_mse": float(np.mean([f["vel_mse"] for f in finals])),
                "pos_mse": float(np.mean([f["pos_mse"] for f in finals])),
                "ke_dev": float(np.mean([abs(f["ke_ratio"] - 1) for f in finals])),
                "mom_drift": float(np.mean([f["mom_drift"] for f in finals])),
                "domain_frac": float(np.mean([f["domain_frac"] for f in finals])),
            })
            curves.append(curve)

        stats = {k: {"mean": float(np.mean([r[k] for r in rep_aggs])),
                     "std": float(np.std([r[k] for r in rep_aggs]))}
                 for k in rep_aggs[0]}
        # mean per-step domain curve (rep0, scenario-mean) for figures
        n = min(len(c["domain_frac"]) for c in curves[0])
        dom_curve = np.mean([c["domain_frac"][:n] for c in curves[0]], axis=0)
        out["models"][tag] = {
            "ckpt": matches[-1].name, "epoch": meta.get("epoch"),
            "val_loss": meta.get("val_loss"),
            "reps": rep_aggs, "stats": stats,
            "domain_curve": dom_curve.tolist(),
        }
        s = stats
        print(f"{tag:<14} ({time.time()-t0:.0f}s)  "
              f"vel={s['vel_mse']['mean']:.3f}±{s['vel_mse']['std']:.3f}  "
              f"pos={s['pos_mse']['mean']:.4f}  "
              f"|KE-1|={s['ke_dev']['mean']:.4f}  "
              f"mom={s['mom_drift']['mean']:.4f}  "
              f"domain={s['domain_frac']['mean']*100:.2f}%")

    out_path = ROOT / "logs" / V4P1_VERSION_ID / "rollout_wall_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
