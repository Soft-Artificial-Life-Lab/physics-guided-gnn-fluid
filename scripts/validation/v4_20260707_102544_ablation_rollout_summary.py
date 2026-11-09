"""
v4 ablation rollout summary — N repeated rollout evaluations per checkpoint.

Version ID : v4_20260707_102544

GPU index_add_ is non-deterministic; over a chaotic 77-step rollout this
yields a few-% run-to-run spread. This driver therefore repeats the full
rollout evaluation N times per checkpoint and reports mean +/- std of the
final-step metrics (aggregated over scenarios), for all v4 ablation ckpts.

Usage:
  python scripts/validation/v4_20260707_102544_ablation_rollout_summary.py --reps 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "validation"))

# Reuse the single-run evaluator's building blocks
from v4_20260707_102544_rollout_eval import (
    DATA_DIR, load_model, load_frames, rollout, V4_VERSION_ID)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--models-glob", type=str, default="hybrid_v4_*_v4_20260707_102544.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted((ROOT / "models").glob(args.models_glob))
    sim_dirs = sorted(d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    print(f"{len(ckpts)} checkpoints x {args.reps} reps x {len(sim_dirs)} scenarios\n")

    # Pre-load GT frames once (shared across all models/reps)
    gt = {sd.name: load_frames(sd) for sd in sim_dirs}

    summary = {"version": V4_VERSION_ID, "reps": args.reps, "models": {}}
    for ck in ckpts:
        # tag = text between 'hybrid_v4_' and the training timestamp
        tag = ck.stem.replace("hybrid_v4_", "").rsplit("_", 4)[0]
        model, meta = load_model(ck, device)
        t0 = time.time()

        rep_aggs = []
        for rep in range(args.reps):
            finals = []
            for sd in sim_dirs:
                pos_gt, vel_gt, types = gt[sd.name]
                metrics, _, _, _ = rollout(
                    model, pos_gt, vel_gt, types, args.steps, args.radius, device)
                finals.append({k: v[-1] for k, v in metrics.items()})
            rep_aggs.append({
                "vel_mse": float(np.mean([f["vel_mse"] for f in finals])),
                "pos_mse": float(np.mean([f["pos_mse"] for f in finals])),
                "ke_dev": float(np.mean([abs(f["ke_ratio"] - 1) for f in finals])),
                "mom_drift": float(np.mean([f["mom_drift"] for f in finals])),
            })

        stats = {k: {"mean": float(np.mean([r[k] for r in rep_aggs])),
                     "std": float(np.std([r[k] for r in rep_aggs]))}
                 for k in rep_aggs[0]}
        summary["models"][tag] = {
            "ckpt": ck.name,
            "epoch": meta.get("epoch"),
            "val_loss": meta.get("val_loss"),
            "reps": rep_aggs,
            "stats": stats,
        }
        s = stats
        print(f"{tag:<10} ({time.time()-t0:.0f}s)  "
              f"vel={s['vel_mse']['mean']:.3f}±{s['vel_mse']['std']:.3f}  "
              f"pos={s['pos_mse']['mean']:.4f}±{s['pos_mse']['std']:.4f}  "
              f"|KE-1|={s['ke_dev']['mean']:.4f}±{s['ke_dev']['std']:.4f}  "
              f"mom={s['mom_drift']['mean']:.4f}±{s['mom_drift']['std']:.4f}")

    out = ROOT / "logs" / V4_VERSION_ID / "ablation_rollout_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
