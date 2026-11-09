# -*- coding: utf-8 -*-
"""v6 rollout evaluation on the GNS WaterRamps validation split (2D).

Version ID : v6_20261021_145750

Mirrors the v5 Water-3D evaluator so the two datasets are read the same way,
with the differences WaterRamps forces:

  * 2D, so the model is rebuilt through HybridPhysicsGNNv6 from the config
    stored in the checkpoint.

  * Two kinds of boundary particle. The ramps come from the data (static, zero
    drift, indices n_fluid onward); the container walls come from the
    checkpoint and are appended after them. Both are inputs, both are held
    fixed, and every boundary-boundary edge is dropped - ramp-ramp, ramp-wall
    and wall-wall alike. Metrics are computed over fluid rows only, which is
    what makes the ramp labelling matter: counted as fluid, 9 to 140 motionless
    particles would quietly flatter every one of them.

  * The escape tolerance is measured, not assumed. Ground-truth water leaves
    the metadata box by up to about one connectivity radius on the y-lo face,
    so a tolerance of zero would score ground truth itself as escaping. The
    default measures, over the evaluated trajectories, the smallest tolerance
    at which ground truth reads exactly zero, prints it, and uses it. The
    protocol asks for the value that takes GT to zero; here it is derived
    rather than copied from Water-3D, where it was 0.01.

  * Penetration is reported, which Water-3D could not support. It is likewise
    calibrated against ground truth: `contact` is the closest any GT fluid
    particle ever comes to a ramp particle in that trajectory, and a predicted
    fluid particle closer than that is closer to the obstacle than the
    reference simulation ever gets.

Reported per trajectory, then averaged: vel_mse, pos_mse, |KE-1|, momentum
drift, escape fraction, penetration fraction. Four of these together, because
each one alone is satisfied by a different failure.

Usage:
  python scripts/validation/v6_20261021_145750_rollout_waterramps.py \
      --ckpt out/models/hybrid_wr_wall2_<ts>_v6_20261021_145750.pt --steps 400
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

from src.models.hybrid_gnn_v6 import HybridPhysicsGNNv6

V6_ID = "v6_20261021_145750"
DATA_DIR = ROOT / "datasets" / "gns_waterramps_v6"
DT = 1.0
EPS = 1e-12


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in ckpt["model_config"].items() if k != "version"}
    model = HybridPhysicsGNNv6(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_graph(positions, radius, n_fluid):
    """Radius graph with every boundary-boundary edge dropped."""
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) > 0 and n_fluid < len(positions):
        keep = ~((pairs[:, 0] >= n_fluid) & (pairs[:, 1] >= n_fluid))
        pairs = pairs[keep]
    if len(pairs) == 0:
        edges = np.array([[0], [0]], dtype=np.int64)
    else:
        edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T.astype(np.int64)
    return torch.from_numpy(edges)


def gt_escape_tolerance(pos_gt, n_fluid, bounds, candidates=None):
    """Smallest tolerance at which ground-truth fluid never leaves the box.

    Returns (tol, overshoot) where overshoot is the largest distance any GT
    fluid particle goes past any face. The tolerance is that overshoot rounded
    up onto a coarse grid, so the reported number stays a round figure instead
    of a 7-decimal artefact of one particle in one frame.
    """
    dim = bounds.shape[1]
    water = pos_gt[:, :n_fluid].reshape(-1, dim)
    over = max(float(np.max(bounds[0] - water.min(0))),
               float(np.max(water.max(0) - bounds[1])), 0.0)
    if candidates is None:
        candidates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015,
                      0.0175, 0.02, 0.025, 0.03]
    for c in candidates:
        if c >= over:
            return c, over
    return over, over


def gt_ramp_contact(pos_gt, n_fluid, n_ramp):
    """Closest any GT fluid particle comes to a ramp particle."""
    if n_ramp == 0:
        return None
    ramp = pos_gt[0, n_fluid:n_fluid + n_ramp]
    tree = cKDTree(ramp)
    best = np.inf
    for t in range(pos_gt.shape[0]):
        d, _ = tree.query(pos_gt[t, :n_fluid], k=1)
        best = min(best, float(d.min()))
    return best


@torch.no_grad()
def rollout(model, pos_gt, n_fluid, n_ramp, walls, steps, radius, bounds,
            device, domain_tol, contact_tol):
    """Autoregressive rollout. Fluid is predicted; ramps and walls stay put."""
    C = model.history
    dim = pos_gt.shape[-1]
    n_data = pos_gt.shape[1]                      # fluid + ramps
    steps = min(steps, pos_gt.shape[0] - C) if steps else pos_gt.shape[0] - C
    start = C - 1

    # GNS convention: velocity is the per-step position delta.
    vel_gt = np.zeros_like(pos_gt)
    vel_gt[1:] = pos_gt[1:] - pos_gt[:-1]
    vel_gt[0] = vel_gt[1]

    n_w = 0 if walls is None else len(walls)
    types = np.zeros(n_data + n_w, np.float32)
    types[n_fluid:] = 1.0
    fluid_t = torch.from_numpy(types < 0.5).to(device)
    types_t = torch.from_numpy(types).to(device)
    walls_t = torch.from_numpy(walls).to(device) if n_w else None

    def aug_pos(p):
        t = torch.from_numpy(p).to(device)
        return torch.cat([t, walls_t], dim=0) if n_w else t

    def aug_vel(v):
        t = torch.from_numpy(v).to(device)
        return torch.cat([t, torch.zeros(n_w, dim, device=device)], dim=0) \
            if n_w else t

    lo = np.asarray(bounds[0], np.float32) - domain_tol
    hi = np.asarray(bounds[1], np.float32) + domain_tol
    ramp_tree = (cKDTree(pos_gt[0, n_fluid:n_fluid + n_ramp])
                 if n_ramp and contact_tol is not None else None)

    x = aug_pos(pos_gt[start].copy())
    hist = torch.stack([aug_vel(vel_gt[i].copy()) for i in range(C)], dim=0)

    keys = ("vel_mse", "pos_mse", "ke_ratio", "mom_drift", "domain_frac",
            "penetration_frac")
    metrics = {k: [] for k in keys}
    diverged_at = None
    for k in range(steps):
        t_next = start + k + 1
        edge_index = build_graph(x.cpu().numpy(), radius, n_fluid).to(device)
        rel = x[edge_index[1]] - x[edge_index[0]]
        ea = torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)
        N = x.shape[0]
        nf = torch.cat(
            [hist.permute(1, 0, 2).reshape(N, dim * C), types_t.unsqueeze(-1)],
            dim=-1)
        v_hat = model.predict_next_velocity(nf, edge_index, ea)

        v_gt_next = aug_vel(vel_gt[t_next])
        x_gt_next = aug_pos(pos_gt[t_next])
        fl = fluid_t.unsqueeze(-1)
        v_new = torch.where(fl, v_hat, v_gt_next)
        x_new = torch.where(fl, x + v_hat * DT, x_gt_next)

        if not torch.isfinite(x_new).all():
            diverged_at = k
            break

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
        if ramp_tree is not None:
            d, _ = ramp_tree.query(fp, k=1)
            metrics["penetration_frac"].append(float((d < contact_tol).mean()))
        else:
            metrics["penetration_frac"].append(float("nan"))

        hist = torch.cat([hist[1:], v_new.unsqueeze(0)], dim=0)
        x = x_new
    return metrics, len(metrics["vel_mse"]), diverged_at


def main():
    p = argparse.ArgumentParser(description="v6 WaterRamps rollout evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=400,
                   help="rollout length (0 = to the end of the trajectory)")
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--limit-trajs", type=int, default=0)
    p.add_argument("--domain-tol", type=float, default=-1.0,
                   help="escape tolerance beyond the metadata box; -1 measures "
                        "the smallest value at which ground truth reads zero")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    meta = json.loads((data_dir / "metadata.json").read_text())
    bounds = np.asarray(meta["bounds"], dtype=np.float32).T          # (2, dim)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    model, ckpt = load_model(ckpt_path, device)
    walls = ckpt.get("walls")
    if walls is not None:
        walls = np.asarray(walls, dtype=np.float32)
    radius = ckpt.get("radius", meta["default_connectivity_radius"])

    files = sorted((data_dir / args.split).glob("traj_*.npz"))
    if args.limit_trajs:
        files = files[: args.limit_trajs]

    bar = "=" * 78
    print("\n" + bar)
    print("V6 WATERRAMPS ROLLOUT  ckpt=%s" % ckpt_path.name)
    print("  split=%s trajs=%d  radius=%g  DT=%g  container walls=%s"
          % (args.split, len(files), radius, DT,
             "none" if walls is None else str(len(walls))))
    print("  train val_loss=%.5f @ epoch %s   type_weighting=%s"
          % (ckpt.get("val_loss", float("nan")), ckpt.get("epoch", "?"),
             ckpt["model_config"].get("use_type_weighting")))
    wi = ckpt.get("wall_info") or {}
    if wi.get("margin_source"):
        print("  wall margin: %s  lo=%s hi=%s"
              % (wi["margin_source"],
                 np.round(wi.get("margin_lo", []), 5).tolist(),
                 np.round(wi.get("margin_hi", []), 5).tolist()))
        if wi.get("encroaches"):
            print("  NOTE: these walls sit inside the GT fluid region on %s"
                  % (wi["encroaches"],))
    print(bar, flush=True)

    # --- escape tolerance, measured over the trajectories being evaluated ---
    loaded = []
    for f in files:
        d = np.load(f)
        loaded.append((f, d["positions"].astype(np.float32), int(d["n_fluid"])))
    tols, overs = zip(*[gt_escape_tolerance(pos, nf, bounds)
                        for _, pos, nf in loaded])
    if args.domain_tol >= 0.0:
        domain_tol = args.domain_tol
        tol_src = "given"
    else:
        domain_tol = max(tols)
        tol_src = "measured"
    print("ground-truth overshoot past the metadata box: max %.5f over %d "
          "trajectories (= %.2f radius)"
          % (max(overs), len(loaded), max(overs) / radius))
    print("escape tolerance: %.5f (%s) -- ground truth reads 0 percent at this "
          "value" % (domain_tol, tol_src))

    per_traj, curves = [], []
    t0 = time.time()
    for f, pos_gt, n_fluid in loaded:
        n_ramp = pos_gt.shape[1] - n_fluid
        contact = gt_ramp_contact(pos_gt, n_fluid, n_ramp)
        m, n, div = rollout(model, pos_gt, n_fluid, n_ramp, walls, args.steps,
                            radius, bounds, device, domain_tol, contact)
        if n == 0:
            print("  %s: diverged immediately" % f.stem)
            continue
        final = {k: v[-1] for k, v in m.items()}
        per_traj.append({"traj": f.stem, "steps": n, "n_fluid": n_fluid,
                         "n_ramp": n_ramp,
                         "gt_ramp_contact": contact, "final": final,
                         "diverged_at": div})
        curves.append(m)
        print("  %s: steps=%4d fluid=%4d ramp=%3d  vel=%.3e pos=%.3e "
              "|KE-1|=%7.3f mom=%.2e escape=%6.2f%% pen=%s%s"
              % (f.stem, n, n_fluid, n_ramp, final["vel_mse"],
                 final["pos_mse"], abs(final["ke_ratio"] - 1.0),
                 final["mom_drift"], final["domain_frac"] * 100,
                 "n/a" if n_ramp == 0 else "%.2f%%" % (final["penetration_frac"] * 100),
                 "" if div is None else "  DIVERGED at %d" % div), flush=True)

    if not per_traj:
        raise SystemExit("every trajectory diverged; nothing to summarise")

    def agg(key, transform=lambda v: v, skip_nan=False):
        vals = [transform(t["final"][key]) for t in per_traj]
        if skip_nan:
            vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals)}

    summary = {
        "vel_mse": agg("vel_mse"),
        "pos_mse": agg("pos_mse"),
        "ke_dev": agg("ke_ratio", lambda v: abs(v - 1.0)),
        "mom_drift": agg("mom_drift"),
        "domain_frac": agg("domain_frac"),
        "penetration_frac": agg("penetration_frac", skip_nan=True),
    }
    n_min = min(len(c["domain_frac"]) for c in curves)
    curve = {k: np.mean([c[k][:n_min] for c in curves], axis=0).tolist()
             for k in ("domain_frac", "vel_mse", "ke_ratio")}

    print(bar)
    print("MEAN over %d trajectories, final step:" % len(per_traj))
    for k, v in summary.items():
        print("  %-16s %.5g +/- %.3g  (n=%d)" % (k, v["mean"], v["std"], v["n"]))
    print("  escape      %.2f%%" % (summary["domain_frac"]["mean"] * 100))
    print("  penetration %.2f%%  (over the %d trajectories that have ramps)"
          % (summary["penetration_frac"]["mean"] * 100,
             summary["penetration_frac"]["n"]))
    n_div = sum(1 for t in per_traj if t["diverged_at"] is not None)
    if n_div:
        print("  %d of %d trajectories produced non-finite positions"
              % (n_div, len(per_traj)))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))

    out = Path(args.out) if args.out else (
        ROOT / "logs" / V6_ID / ("rollout_%s.json" % ckpt_path.stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": V6_ID, "ckpt": str(ckpt_path),
        "dataset": "gns_waterramps", "split": args.split,
        "radius": radius, "dt": DT, "dim": int(bounds.shape[1]),
        "n_container_walls": 0 if walls is None else len(walls),
        "wall_info": wi,
        "use_type_weighting": ckpt["model_config"].get("use_type_weighting"),
        "train_val_loss": ckpt.get("val_loss"),
        "train_epoch": ckpt.get("epoch"),
        "escape_tolerance": domain_tol,
        "escape_tolerance_source": tol_src,
        "gt_overshoot_max": max(overs),
        "steps": args.steps, "per_trajectory": per_traj,
        "summary": summary, "curves": curve,
    }, indent=2))
    print("saved -> %s" % out)


if __name__ == "__main__":
    main()
