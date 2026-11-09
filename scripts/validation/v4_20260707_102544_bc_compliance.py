"""
Boundary-condition compliance analysis (v4_20260707_102544).

Answers: does the learned simulator actually RESPECT boundary conditions,
beyond having low MSE near the boundary? Three physical violation tests,
evaluated on the model's rolled-out FLUID positions against the GT rigid
geometry (rigid particles are GT-snapped during rollout, so the boundary
itself is identical for all models):

  1. penetration : fluid particle is DEEPER inside a rigid body's boundary
     shell than the local shell radius. For each fluid p: nearest rigid
     particle b (body k, centroid c_k); violation if
         ||p - c_k|| < ||b - c_k|| - margin        (convex bodies: valid
     for this dataset's sphere / box / cylinder / paddle shapes)
  2. contact violation : d(p, nearest rigid) < contact_thresh, i.e. closer
     than anything ever observed in GT (GT global min ~ 0.013).
  3. domain violation : p outside the scenario's GT bounding box (over all
     frames) + tolerance — particles leaking through the tank walls.

GT fluid positions are evaluated with the same tests as the reference
baseline (should be ~0 by construction).

Usage:
  python scripts/validation/v4_20260707_102544_bc_compliance.py \
      --tags backbone attn_only typew_only full100 --reps 3 --steps 100
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
    DATA_DIR, DT, load_model, build_graph, V4_VERSION_ID)


def load_frames_full(sim_dir):
    files = sorted(Path(sim_dir).glob("frame_*.npz"))
    pos, vel = [], []
    types = body_ids = None
    for f in files:
        d = np.load(f)
        pos.append(d["positions"].astype(np.float32))
        vel.append(d["velocities"].astype(np.float32))
        if types is None:
            types = d["particle_types"].astype(np.float32)
            body_ids = d["body_ids"].astype(np.int64)
    return np.stack(pos), np.stack(vel), types, body_ids


def violation_fractions(fluid_pos, rigid_pos, rigid_body_ids, bbox_lo, bbox_hi,
                        pen_margin, contact_thresh, domain_tol):
    """Fractions of fluid particles violating each condition at one frame."""
    n = len(fluid_pos)
    tree = cKDTree(rigid_pos)
    d_near, idx = tree.query(fluid_pos, k=1)

    # 1. penetration (deeper inside the convex shell than its local radius)
    pen = np.zeros(n, dtype=bool)
    for k in np.unique(rigid_body_ids):
        c = rigid_pos[rigid_body_ids == k].mean(axis=0)
        sel = rigid_body_ids[idx] == k
        if sel.any():
            r_fluid = np.linalg.norm(fluid_pos[sel] - c, axis=1)
            r_shell = np.linalg.norm(rigid_pos[idx[sel]] - c, axis=1)
            pen[sel] = r_fluid < (r_shell - pen_margin)

    contact = d_near < contact_thresh
    domain = ((fluid_pos < bbox_lo - domain_tol) |
              (fluid_pos > bbox_hi + domain_tol)).any(axis=1)
    return pen.mean(), contact.mean(), domain.mean()


@torch.no_grad()
def rollout_compliance(model, pos_gt, vel_gt, types, body_ids, steps, radius,
                       device, params, acc, acc_gt=None):
    C = model.history
    steps = min(steps, pos_gt.shape[0] - C)
    start = C - 1

    fluid = types < 0.5
    fluid_t = torch.from_numpy(fluid).to(device)
    types_t = torch.from_numpy(types).to(device)
    rigid_ids = body_ids[~fluid]
    bbox_lo = pos_gt.reshape(-1, 3).min(axis=0)
    bbox_hi = pos_gt.reshape(-1, 3).max(axis=0)

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

        rigid_now = pos_gt[t_next][~fluid]
        fr = violation_fractions(
            x_new[fluid_t].cpu().numpy(), rigid_now, rigid_ids,
            bbox_lo, bbox_hi, **params)
        for name, v in zip(("penetration", "contact", "domain"), fr):
            acc[(k + 1, name)].append(float(v))

        if acc_gt is not None:      # GT baseline with identical tests
            fr_gt = violation_fractions(
                pos_gt[t_next][fluid], rigid_now, rigid_ids,
                bbox_lo, bbox_hi, **params)
            for name, v in zip(("penetration", "contact", "domain"), fr_gt):
                acc_gt[(k + 1, name)].append(float(v))

        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--pen-margin", type=float, default=0.01)
    p.add_argument("--contact-thresh", type=float, default=0.012,
                   help="below GT global min fluid-rigid distance (~0.013)")
    p.add_argument("--domain-tol", type=float, default=0.02)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_dirs = sorted(d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    gt = {sd.name: load_frames_full(sd) for sd in sim_dirs}
    params = dict(pen_margin=args.pen_margin,
                  contact_thresh=args.contact_thresh,
                  domain_tol=args.domain_tol)
    print(f"{len(args.tags)} models x {args.reps} reps x {len(sim_dirs)} scenarios | "
          f"pen_margin={args.pen_margin} contact<{args.contact_thresh} "
          f"domain_tol={args.domain_tol}\n")

    out = {"version": V4_VERSION_ID, "params": params, "reps": args.reps,
           "models": {}}
    acc_gt = defaultdict(list)

    for mi, tag in enumerate(args.tags):
        matches = sorted((ROOT / "models").glob(
            f"hybrid_v4_{tag}_*_v4_20260707_102544.pt"))
        if not matches:
            print(f"[skip] {tag}")
            continue
        model, _ = load_model(matches[-1], device)
        t0 = time.time()
        acc = defaultdict(list)
        n_steps = 0
        for rep in range(args.reps):
            for sd in sim_dirs:
                pos_gt, vel_gt, types, body_ids = gt[sd.name]
                n_steps = rollout_compliance(
                    model, pos_gt, vel_gt, types, body_ids, args.steps,
                    args.radius, device, params, acc,
                    acc_gt if (mi == 0 and rep == 0) else None)

        res = {}
        for name in ("penetration", "contact", "domain"):
            res[name] = [float(np.mean(acc[(k, name)]))
                         for k in range(1, n_steps + 1)]
        out["models"][tag] = {"ckpt": matches[-1].name, **res}
        print(f"{tag:<11} ({time.time()-t0:.0f}s)  "
              f"pen @5={res['penetration'][4]*100:.3f}% @20={res['penetration'][19]*100:.3f}% "
              f"@end={res['penetration'][-1]*100:.3f}%  | "
              f"contact @end={res['contact'][-1]*100:.3f}%  | "
              f"domain @end={res['domain'][-1]*100:.3f}%")

    gt_res = {}
    n_steps = max(k for k, _ in acc_gt) if acc_gt else 0
    for name in ("penetration", "contact", "domain"):
        gt_res[name] = [float(np.mean(acc_gt[(k, name)]))
                        for k in range(1, n_steps + 1)]
    out["gt_baseline"] = gt_res
    if n_steps:
        print(f"{'GT':<11}          "
              f"pen @end={gt_res['penetration'][-1]*100:.3f}%  | "
              f"contact @end={gt_res['contact'][-1]*100:.3f}%  | "
              f"domain @end={gt_res['domain'][-1]*100:.3f}%")

    out_path = ROOT / "logs" / V4_VERSION_ID / "bc_compliance.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
