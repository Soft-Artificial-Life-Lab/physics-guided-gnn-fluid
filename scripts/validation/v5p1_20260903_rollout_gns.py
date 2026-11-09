"""
v5p1 rollout evaluation for the GNS baseline on Water-3D.

Mirrors scripts/validation/v5_20260820_022909_rollout_water3d.py exactly
(same metrics, same domain tolerance semantics, same JSON layout) so the
numbers drop straight into the proposed-vs-GNS comparison table. Only the
model interface differs: GNS consumes boundary-distance features and predicts
acceleration; there are no wall particles.

Usage:
  python scripts/validation/v5p1_20260903_rollout_gns.py \
      --ckpt models/gns_gns_e100_<ts>_v5p1_20260903.pt --steps 400
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

from src.models.gns_baseline import GNSBaseline

V5P1_VERSION_ID = "v5p1_20260903"
DATA_DIR = ROOT / "datasets" / "gns_water3d_v5"
EPS = 1e-12


def build_graph(positions, radius, self_edges=False):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    if self_edges:
        ids = np.arange(len(positions), dtype=np.int64)
        edges = np.concatenate([edges, np.stack([ids, ids])], axis=1)
    rel = (positions[edges[1]] - positions[edges[0]]) / radius
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    return (torch.from_numpy(edges),
            torch.from_numpy(np.concatenate([rel, dist], 1).astype(np.float32)))


@torch.no_grad()
def rollout(model, pos_gt, steps, radius, lo, hi, device, domain_tol=0.01,
            self_edges=False, esc_lo=None, esc_hi=None):
    C = model.history
    steps = min(steps, pos_gt.shape[0] - C) if steps else pos_gt.shape[0] - C
    start = C - 1

    vel_gt = np.zeros_like(pos_gt)
    vel_gt[1:] = pos_gt[1:] - pos_gt[:-1]
    vel_gt[0] = vel_gt[1]

    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.tensor(hi, dtype=torch.float32, device=device)
    # escape is always judged against the dataset's own bounding box, so that
    # a model trained with a corrected reference box is not also given a more
    # lenient escape criterion than the configurations it is compared with
    eb_lo = lo if esc_lo is None else esc_lo
    eb_hi = hi if esc_hi is None else esc_hi
    lo_esc = np.asarray(eb_lo, np.float32) - domain_tol
    hi_esc = np.asarray(eb_hi, np.float32) + domain_tol

    x = torch.from_numpy(pos_gt[start].copy()).to(device)
    hist = torch.from_numpy(vel_gt[start - C + 1: start + 1].copy()).to(device)

    metrics = {k: [] for k in ("vel_mse", "pos_mse", "ke_ratio", "mom_drift",
                               "domain_frac")}
    for k in range(steps):
        t_next = start + k + 1
        ei, ea = build_graph(x.cpu().numpy(), radius, self_edges)
        ei, ea = ei.to(device), ea.to(device)
        v_hat = model.predict_next_velocity(x, hist, ei, ea, lo_t, hi_t, radius)
        x_new = x + v_hat

        v_gt_next = torch.from_numpy(vel_gt[t_next]).to(device)
        x_gt_next = torch.from_numpy(pos_gt[t_next]).to(device)
        ke_pred = 0.5 * torch.sum(v_hat ** 2)
        ke_true = 0.5 * torch.sum(v_gt_next ** 2)
        metrics["vel_mse"].append(torch.mean((v_hat - v_gt_next) ** 2).item())
        metrics["pos_mse"].append(torch.mean((x_new - x_gt_next) ** 2).item())
        metrics["ke_ratio"].append((ke_pred / (ke_true + EPS)).item())
        metrics["mom_drift"].append(
            (v_hat.mean(0) - v_gt_next.mean(0)).norm().item())
        fp = x_new.cpu().numpy()
        outside = ((fp < lo_esc) | (fp > hi_esc)).any(axis=1)
        metrics["domain_frac"].append(float(outside.mean()))

        hist = torch.cat([hist[1:], v_hat.unsqueeze(0)], dim=0)
        x = x_new
    return metrics, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--domain-tol", type=float, default=0.01)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GNSBaseline(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    radius = ckpt["radius"]
    self_edges = bool(ckpt.get("self_edges", False))
    lo, hi = ckpt["bounds_lo"], ckpt["bounds_hi"]
    _meta = json.loads((DATA_DIR / "metadata.json").read_text())
    meta_lo = [b[0] for b in _meta["bounds"]]
    meta_hi = [b[1] for b in _meta["bounds"]]

    files = sorted((DATA_DIR / args.split).glob("traj_*.npz"))
    bar = "=" * 70
    print("\n" + bar)
    print("V5P1 GNS ROLLOUT  ckpt=%s" % ckpt_path.name)
    print("  split=%s trajs=%d radius=%g  boundary=distance features  self_edges=%s"
          % (args.split, len(files), radius, self_edges))
    print(bar)

    per_traj, curves = [], []
    t0 = time.time()
    for f in files:
        pos_gt = np.load(f)["positions"].astype(np.float32)
        m, n = rollout(model, pos_gt, args.steps, radius, lo, hi, device,
                       esc_lo=meta_lo, esc_hi=meta_hi,
                       domain_tol=args.domain_tol, self_edges=self_edges)
        final = {k: v[-1] for k, v in m.items()}
        per_traj.append({"traj": f.stem, "steps": n, "final": final})
        curves.append(m)
        print("  %s: steps=%d vel=%.3e pos=%.3e |KE-1|=%.3f mom=%.3e escape=%.2f%%"
              % (f.stem, n, final["vel_mse"], final["pos_mse"],
                 abs(final["ke_ratio"] - 1.0), final["mom_drift"],
                 final["domain_frac"] * 100))

    def agg(key, tf=lambda v: v):
        vals = [tf(t["final"][key]) for t in per_traj]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {"vel_mse": agg("vel_mse"), "pos_mse": agg("pos_mse"),
               "ke_dev": agg("ke_ratio", lambda v: abs(v - 1.0)),
               "mom_drift": agg("mom_drift"), "domain_frac": agg("domain_frac")}
    n_min = min(len(c["domain_frac"]) for c in curves)
    print(bar)
    print("MEAN over %d trajectories (final step):" % len(per_traj))
    for k, v in summary.items():
        print("  %-12s %.5g +/- %.3g" % (k, v["mean"], v["std"]))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))

    out = Path(args.out) if args.out else (
        ROOT / "logs" / V5P1_VERSION_ID / ("rollout_%s_%dstep.json"
                                           % (ckpt_path.stem, args.steps)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": V5P1_VERSION_ID, "ckpt": ckpt_path.name,
        "dataset": "gns_water3d", "split": args.split, "radius": radius,
        "n_walls": 0, "steps": args.steps, "domain_tol": args.domain_tol,
        "per_trajectory": per_traj, "summary": summary,
        "domain_curve": np.mean([c["domain_frac"][:n_min] for c in curves], 0).tolist(),
        "vel_curve": np.mean([c["vel_mse"][:n_min] for c in curves], 0).tolist(),
        "ke_curve": np.mean([c["ke_ratio"][:n_min] for c in curves], 0).tolist(),
    }, indent=2))
    print("saved -> %s" % out)


if __name__ == "__main__":
    main()
