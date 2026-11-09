"""
v5 rollout evaluation on the public GNS Water-3D validation split.

Version ID : v5_20260820_022909
Mirrors the v4.1 evaluator (scripts/validation/v4p1_20260712_001555_rollout_wall_eval.py)
so the numbers are directly comparable with our own-data results, with three
dataset-specific changes:

  * DT = 1.0            — GNS stores velocity as the per-step position delta.
  * radius, bounds      — taken from the Water-3D metadata (0.035, [0.125, 0.875]^3)
                          rather than inferred from the GT bounding box, because
                          the true container is known here.
  * walls               — restored from the checkpoint (None for --boundary none
                          runs, in which case the model has no boundary input at
                          all and domain escape is expected).

Reported per scenario, then averaged:
  vel_mse, pos_mse, |KE ratio|, momentum drift, domain-escape fraction.

Usage:
  python scripts/validation/v5_20260820_022909_rollout_water3d.py \
      --ckpt models/hybrid_w3d_wall_<ts>_v5_20260820_022909.pt --steps 400
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

from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

V5_VERSION_ID = "v5_20260820_022909"
DATA_DIR = ROOT / "datasets" / "gns_water3d_v5"
DT = 1.0
EPS = 1e-12


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in ckpt["model_config"].items() if k != "version"}
    model = HybridPhysicsGNNv4(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_graph(positions, radius, n_orig, has_walls):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if has_walls and len(pairs) > 0:
        keep = ~((pairs[:, 0] >= n_orig) & (pairs[:, 1] >= n_orig))
        pairs = pairs[keep]
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    return torch.from_numpy(edges)


@torch.no_grad()
def rollout(model, pos_gt, walls, steps, radius, bounds, device, domain_tol=0.0,
            clamp=False):
    """Autoregressive rollout. Fluid is predicted; walls stay static."""
    C = model.history
    n_orig = pos_gt.shape[1]
    steps = min(steps, pos_gt.shape[0] - C) if steps else pos_gt.shape[0] - C
    start = C - 1

    # GNS convention: velocity is the per-step position delta.
    vel_gt = np.zeros_like(pos_gt)
    vel_gt[1:] = pos_gt[1:] - pos_gt[:-1]
    vel_gt[0] = vel_gt[1]

    n_w = 0 if walls is None else len(walls)
    has_walls = n_w > 0
    types = np.concatenate([np.zeros(n_orig, np.float32),
                            np.ones(n_w, np.float32)]) if has_walls \
        else np.zeros(n_orig, np.float32)
    fluid_t = torch.from_numpy(types < 0.5).to(device)
    types_t = torch.from_numpy(types).to(device)
    walls_t = (torch.from_numpy(walls).to(device) if has_walls else None)

    def aug_pos(p):
        t = torch.from_numpy(p).to(device)
        return torch.cat([t, walls_t], dim=0) if has_walls else t

    def aug_vel(v):
        t = torch.from_numpy(v).to(device)
        return torch.cat([t, torch.zeros(n_w, 3, device=device)], dim=0) \
            if has_walls else t

    lo = np.asarray(bounds[0], np.float32) - domain_tol
    hi = np.asarray(bounds[1], np.float32) + domain_tol

    x = aug_pos(pos_gt[start].copy())
    hist = torch.stack([aug_vel(vel_gt[i].copy()) for i in range(C)], dim=0)

    metrics = {k: [] for k in ("vel_mse", "pos_mse", "ke_ratio", "mom_drift",
                               "domain_frac")}
    for k in range(steps):
        t_next = start + k + 1
        edge_index = build_graph(x.cpu().numpy(), radius, n_orig, has_walls).to(device)
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
        if clamp:
            # trivial post-hoc boundary handling: project escaped fluid back
            lo_c = torch.tensor(bounds[0], dtype=x_new.dtype, device=x_new.device)
            hi_c = torch.tensor(bounds[1], dtype=x_new.dtype, device=x_new.device)
            x_cl = torch.max(torch.min(x_new, hi_c), lo_c)
            x_new = torch.where(fluid_t.unsqueeze(-1), x_cl, x_new)

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
        outside = ((fp < lo) | (fp > hi)).any(axis=1)
        metrics["domain_frac"].append(float(outside.mean()))

        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new
    return metrics, steps


def main():
    p = argparse.ArgumentParser(description="v5 Water-3D rollout evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=400,
                   help="rollout length (0 = full trajectory)")
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--limit-trajs", type=int, default=0)
    p.add_argument("--domain-tol", type=float, default=0.01,
                   help="escape tolerance beyond metadata bounds; GT particles "
                        "legitimately compress ~0.007 past them, so 0.01 makes "
                        "GT escape exactly 0%% (tol=0 overcounts by 7-9%%)")
    p.add_argument("--clamp", action="store_true",
                   help="project escaped fluid back onto the bounding box "
                        "after every step (trivial post-hoc baseline)")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    radius = meta["default_connectivity_radius"]
    bounds = np.asarray(meta["bounds"], dtype=np.float32).T          # (2, 3)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    model, ckpt = load_model(ckpt_path, device)
    walls = ckpt.get("walls")
    if walls is not None:
        walls = np.asarray(walls, dtype=np.float32)
    radius = ckpt.get("radius", radius)

    files = sorted((DATA_DIR / args.split).glob("traj_*.npz"))
    if args.limit_trajs:
        files = files[: args.limit_trajs]

    bar = "=" * 70
    print("\n" + bar)
    print("V5 WATER-3D ROLLOUT  ckpt=%s" % ckpt_path.name)
    print("  split=%s trajs=%d  radius=%g  DT=%g  walls=%s"
          % (args.split, len(files), radius, DT,
             "none" if walls is None else str(len(walls))))
    print("  train val_loss=%.5f @ epoch %s"
          % (ckpt.get("val_loss", float("nan")), ckpt.get("epoch", "?")))
    print(bar)

    per_traj, curves = [], []
    t0 = time.time()
    for f in files:
        pos_gt = np.load(f)["positions"].astype(np.float32)
        m, n = rollout(model, pos_gt, walls, args.steps, radius, bounds, device,
                       domain_tol=args.domain_tol,
                       clamp=args.clamp)
        final = {k: v[-1] for k, v in m.items()}
        per_traj.append({"traj": f.stem, "steps": n, "final": final})
        curves.append(m)
        print("  %s: steps=%d  vel=%.3e pos=%.3e |KE-1|=%.3f mom=%.3e "
              "escape=%.2f%%"
              % (f.stem, n, final["vel_mse"], final["pos_mse"],
                 abs(final["ke_ratio"] - 1.0), final["mom_drift"],
                 final["domain_frac"] * 100))

    def agg(key, transform=lambda v: v):
        vals = [transform(t["final"][key]) for t in per_traj]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "vel_mse": agg("vel_mse"),
        "pos_mse": agg("pos_mse"),
        "ke_dev": agg("ke_ratio", lambda v: abs(v - 1.0)),
        "mom_drift": agg("mom_drift"),
        "domain_frac": agg("domain_frac"),
    }
    n_min = min(len(c["domain_frac"]) for c in curves)
    dom_curve = np.mean([c["domain_frac"][:n_min] for c in curves], axis=0)
    vel_curve = np.mean([c["vel_mse"][:n_min] for c in curves], axis=0)
    ke_curve = np.mean([c["ke_ratio"][:n_min] for c in curves], axis=0)

    print(bar)
    print("MEAN over %d trajectories (final step):" % len(per_traj))
    for k, v in summary.items():
        print("  %-12s %.5g +/- %.3g" % (k, v["mean"], v["std"]))
    print("  domain escape: %.2f%%" % (summary["domain_frac"]["mean"] * 100))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))

    out = Path(args.out) if args.out else (
        ROOT / "logs" / V5_VERSION_ID / ("rollout_%s.json" % ckpt_path.stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": V5_VERSION_ID, "ckpt": str(ckpt_path),
        "dataset": "gns_water3d", "split": args.split,
        "radius": radius, "dt": DT,
        "n_walls": 0 if walls is None else len(walls),
        "steps": args.steps, "per_trajectory": per_traj, "summary": summary,
        "domain_curve": dom_curve.tolist(),
        "vel_curve": vel_curve.tolist(),
        "ke_curve": ke_curve.tolist(),
    }, indent=2))
    print("saved -> %s" % out)


if __name__ == "__main__":
    main()
