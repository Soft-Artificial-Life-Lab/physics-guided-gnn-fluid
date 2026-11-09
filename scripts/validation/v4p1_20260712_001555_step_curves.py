"""
Per-step rollout curves: wall_hybrid vs wall_backbone (v4p1_20260712_001555).

Does the Hybrid's 1-step advantage (-4.8% val) survive into the early/mid
rollout steps, or is it erased once autoregressive error dominates?

Records scenario-mean per-step curves (vel_mse, pos_mse, ke_ratio, mom_drift)
averaged over reps, plus the hybrid/backbone ratio per step. Reuses the wall
rollout implementation unchanged.

Usage:
  python scripts/validation/v4p1_20260712_001555_step_curves.py --reps 3
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))
sys.path.append(str(ROOT / "scripts" / "validation"))

ev = importlib.import_module("v4p1_20260712_001555_rollout_wall_eval")
v4p1_train = importlib.import_module("v4p1_20260712_001555_train_wall")

METRICS = ("vel_mse", "pos_mse", "ke_ratio", "mom_drift")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["wall_hybrid", "wall_backbone"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--radius", type=float, default=0.3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_dirs = sorted(d for d in ev.DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    gt = {sd.name: ev.load_frames(sd) for sd in sim_dirs}
    walls = {sd.name: v4p1_train.make_wall_particles(sd) for sd in sim_dirs}

    out = {"version": ev.V4P1_VERSION_ID, "reps": args.reps, "models": {}}
    for tag in args.tags:
        matches = sorted((ROOT / "models").glob(
            f"hybrid_{tag}_*_{ev.V4P1_VERSION_ID}.pt"))
        model, _ = ev.load_model(matches[-1], device)
        t0 = time.time()

        acc = {m: [] for m in METRICS}          # list of (rep x scen) arrays
        for _ in range(args.reps):
            for sd in sim_dirs:
                pos_gt, vel_gt, types = gt[sd.name]
                m, _steps = ev.rollout(model, pos_gt, vel_gt, types,
                                       walls[sd.name], args.steps,
                                       args.radius, device)
                for k in METRICS:
                    acc[k].append(np.asarray(m[k]))

        n = min(len(a) for k in METRICS for a in acc[k])
        curves = {k: np.mean([a[:n] for a in acc[k]], axis=0).tolist()
                  for k in METRICS}
        out["models"][tag] = curves
        print(f"{tag:<14} done ({time.time()-t0:.0f}s, {n} steps)")

    # step table + ratios
    h, b = out["models"][args.tags[0]], out["models"][args.tags[1]]
    key_steps = [1, 3, 5, 10, 20, 40, len(h["vel_mse"])]
    print(f"\n{'step':>5} | {'vel h':>8} {'vel b':>8} {'h/b':>6} | "
          f"{'pos h':>9} {'pos b':>9} {'h/b':>6}")
    for s in key_steps:
        i = s - 1
        rv = h["vel_mse"][i] / b["vel_mse"][i]
        rp = h["pos_mse"][i] / b["pos_mse"][i]
        print(f"{s:>5} | {h['vel_mse'][i]:>8.4f} {b['vel_mse'][i]:>8.4f} {rv:>6.3f} | "
              f"{h['pos_mse'][i]:>9.6f} {b['pos_mse'][i]:>9.6f} {rp:>6.3f}")

    out_path = ROOT / "logs" / ev.V4P1_VERSION_ID / "step_curves.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
