"""
Near-boundary early-step error analysis (v4_20260707_102544).

Question: does the Hybrid architecture (attention gate + type weighting,
which specifically target boundary interactions) reduce position/velocity
error for fluid particles NEAR the rigid body — especially in the early
rollout steps — even if global metrics dilute the effect?

Method:
  - Autoregressive rollout (same protocol as the v4 evaluator).
  - At each step k, fluid particles are classified by the distance from
    their GT position to the nearest rigid GT particle at that frame:
        near : d < --near-radius  (default 0.3 = graph connectivity radius)
        far  : d >= near-radius
    Classification uses GT geometry only, so the mask is identical for
    every model (fair comparison).
  - Per (model, step, band): particle-pooled MSE of position and velocity
    against GT, pooled over scenarios and repeats.

Usage:
  python scripts/validation/v4_20260707_102544_boundary_error_analysis.py \
      --tags backbone attn_only typew_only full100 --reps 3 --steps 30
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "validation"))

from v4_20260707_102544_rollout_eval import (
    DATA_DIR, DT, load_model, load_frames, build_graph, V4_VERSION_ID)


@torch.no_grad()
def rollout_banded(model, pos_gt, vel_gt, types, steps, radius, near_radius,
                   device, acc):
    """Rollout; accumulate per-(step, band) squared errors into `acc`."""
    C = model.history
    steps = min(steps, pos_gt.shape[0] - C)
    start = C - 1

    fluid = types < 0.5
    rigid_mask = ~fluid
    fluid_t = torch.from_numpy(fluid).to(device)
    types_t = torch.from_numpy(types).to(device)

    x = torch.from_numpy(pos_gt[start].copy()).to(device)
    hist = torch.from_numpy(vel_gt[:C].copy()).to(device)

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

        v_gt_next = torch.from_numpy(vel_gt[t_next]).to(device)
        x_gt_next = torch.from_numpy(pos_gt[t_next]).to(device)
        v_new = torch.where(fluid_t.unsqueeze(-1), v_hat, v_gt_next)
        x_new = torch.where(fluid_t.unsqueeze(-1), x + v_hat * DT, x_gt_next)

        # --- band classification on GT geometry at t_next ---
        gt_f = pos_gt[t_next][fluid]
        gt_r = pos_gt[t_next][rigid_mask]
        d_near, _ = cKDTree(gt_r).query(gt_f, k=1)
        near = d_near < near_radius                       # (N_fluid,)

        pos_se = ((x_new[fluid_t] - x_gt_next[fluid_t]) ** 2).sum(-1).cpu().numpy()
        vel_se = ((v_new[fluid_t] - v_gt_next[fluid_t]) ** 2).sum(-1).cpu().numpy()

        for band, mask in (("near", near), ("far", ~near)):
            if mask.any():
                a = acc[(k + 1, band)]
                a[0] += float(pos_se[mask].sum())
                a[1] += float(vel_se[mask].sum())
                a[2] += int(mask.sum())

        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True,
                   help="ckpt tags, e.g. backbone attn_only typew_only full100")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--near-radius", type=float, default=0.3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_dirs = sorted(d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    gt = {sd.name: load_frames(sd) for sd in sim_dirs}
    print(f"{len(args.tags)} models x {args.reps} reps x {len(sim_dirs)} scenarios, "
          f"{args.steps} steps, near < {args.near_radius}\n")

    out = {"version": V4_VERSION_ID, "near_radius": args.near_radius,
           "steps": args.steps, "reps": args.reps, "models": {}}

    for tag in args.tags:
        matches = sorted((ROOT / "models").glob(
            f"hybrid_v4_{tag}_*_v4_20260707_102544.pt"))
        if not matches:
            print(f"[skip] no ckpt for tag {tag}")
            continue
        model, meta = load_model(matches[-1], device)
        t0 = time.time()

        acc = defaultdict(lambda: [0.0, 0.0, 0])          # (step, band) -> [pos_sse, vel_sse, n]
        for _ in range(args.reps):
            for sd in sim_dirs:
                pos_gt, vel_gt, types = gt[sd.name]
                rollout_banded(model, pos_gt, vel_gt, types, args.steps,
                               args.radius, args.near_radius, device, acc)

        res = {band: {"pos_mse": [], "vel_mse": [], "n": []}
               for band in ("near", "far")}
        for k in range(1, args.steps + 1):
            for band in ("near", "far"):
                sse_p, sse_v, n = acc[(k, band)]
                res[band]["pos_mse"].append(sse_p / max(n, 1))
                res[band]["vel_mse"].append(sse_v / max(n, 1))
                res[band]["n"].append(n)

        out["models"][tag] = {"ckpt": matches[-1].name, **res}
        e5 = res["near"]["pos_mse"][4]; e20 = res["near"]["pos_mse"][19]
        f5 = res["far"]["pos_mse"][4]; f20 = res["far"]["pos_mse"][19]
        frac = res["near"]["n"][4] / (res["near"]["n"][4] + res["far"]["n"][4])
        print(f"{tag:<11} ({time.time()-t0:.0f}s)  near pos_mse "
              f"@5={e5:.5f} @20={e20:.5f} | far @5={f5:.5f} @20={f20:.5f} "
              f"| near frac={frac:.2f}")

    out_path = ROOT / "logs" / V4_VERSION_ID / "boundary_error_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
