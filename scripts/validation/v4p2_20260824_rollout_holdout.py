"""
v4p2 rollout evaluation restricted to HELD-OUT simulations.

Reuses the v4.1 evaluator's rollout()/load_model()/load_frames() unchanged so
numbers are directly comparable with the published wall_hybrid results; only
checkpoint selection (explicit path) and the simulation filter are new. The
held-out sim list is read from the checkpoint (saved by the v4p2 trainer)
unless --sims overrides it.

Usage:
  python scripts/validation/v4p2_20260824_rollout_holdout.py \
      --ckpt models/hybrid_simsplit_hybrid_<ts>_v4p2_20260824.pt --steps 77
  # cross-check: published frame-split model on the same held-out sims
  python scripts/validation/v4p2_20260824_rollout_holdout.py \
      --ckpt models/hybrid_wall_hybrid_<ts>_v4p1_20260712_001555.pt \
      --sims _004 --steps 77
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

V4P2_VERSION_ID = "v4p2_20260824"
DATA_DIR = v4p1_train.DATA_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--sims", nargs="*", default=None,
                   help="substring filters; default = val_sims from the ckpt")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=77)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--wall-spacing", type=float, default=0.1)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    model, meta = ev.load_model(ckpt_path, device)

    if args.sims:
        keep = lambda name: any(s in name for s in args.sims)
    elif meta.get("val_sims"):
        val_set = set(meta["val_sims"])
        keep = lambda name: name in val_set
    else:
        raise SystemExit("checkpoint has no val_sims - pass --sims")

    sim_dirs = [d for d in sorted(Path(DATA_DIR).iterdir())
                if d.is_dir() and d.name.startswith("sph_") and keep(d.name)]
    if not sim_dirs:
        raise SystemExit("no simulations match the filter")

    bar = "=" * 70
    print("\n" + bar)
    print("V4P2 HELD-OUT ROLLOUT  ckpt=%s" % ckpt_path.name)
    print("  sims (%d): %s" % (len(sim_dirs), ", ".join(d.name for d in sim_dirs)))
    print("  steps=%d reps=%d" % (args.steps, args.reps))
    print(bar)

    gt = {sd.name: ev.load_frames(sd) for sd in sim_dirs}
    walls = {sd.name: v4p1_train.make_wall_particles(sd, spacing=args.wall_spacing)
             for sd in sim_dirs}

    rep_aggs, per_sim_all = [], {}
    t0 = time.time()
    for rep in range(args.reps):
        finals = []
        for sd in sim_dirs:
            pos_gt, vel_gt, types = gt[sd.name]
            m, _ = ev.rollout(model, pos_gt, vel_gt, types, walls[sd.name],
                              args.steps, args.radius, device)
            f = {k: v[-1] for k, v in m.items()}
            finals.append(f)
            per_sim_all.setdefault(sd.name, []).append(f)
        rep_aggs.append({
            "vel_mse": float(np.mean([f["vel_mse"] for f in finals])),
            "pos_mse": float(np.mean([f["pos_mse"] for f in finals])),
            "ke_dev": float(np.mean([abs(f["ke_ratio"] - 1) for f in finals])),
            "mom_drift": float(np.mean([f["mom_drift"] for f in finals])),
            "domain_frac": float(np.mean([f["domain_frac"] for f in finals])),
        })

    stats = {k: {"mean": float(np.mean([r[k] for r in rep_aggs])),
                 "std": float(np.std([r[k] for r in rep_aggs]))}
             for k in rep_aggs[0]}
    print("HELD-OUT MEAN over %d sims x %d reps (%.0fs):"
          % (len(sim_dirs), args.reps, time.time() - t0))
    print("  vel=%.3f±%.3f  pos=%.4f  |KE-1|=%.4f  mom=%.4f  domain=%.2f%%"
          % (stats["vel_mse"]["mean"], stats["vel_mse"]["std"],
             stats["pos_mse"]["mean"], stats["ke_dev"]["mean"],
             stats["mom_drift"]["mean"], stats["domain_frac"]["mean"] * 100))

    out = Path(args.out) if args.out else (
        ROOT / "logs" / V4P2_VERSION_ID / ("holdout_%s.json" % ckpt_path.stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": V4P2_VERSION_ID, "ckpt": ckpt_path.name,
        "sims": [d.name for d in sim_dirs], "steps": args.steps,
        "reps": rep_aggs, "stats": stats,
        "per_sim": {k: v for k, v in per_sim_all.items()},
    }, indent=2))
    print("saved -> %s" % out)


if __name__ == "__main__":
    main()
