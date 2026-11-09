# -*- coding: utf-8 -*-
"""Convert DeepMind GNS **WaterRamps** TFRecords into per-trajectory .npz.

Version ID : v6_20261021_145750

Why this is not v5_convert_water3d.py
-------------------------------------
Two differences, and the second one is the reason this file exists rather
than a flag on the v5 converter.

1. WaterRamps is two-dimensional (dim=2), 601 frames per trajectory, radius
   0.015, dt 0.0025.

2. **WaterRamps contains static ramp obstacles**, shipped in the data as GNS
   kinematic particles (type 3) -- 9 to 135 of them, in 26 of the 30
   validation trajectories. The v5 converter ends with

       fluid = np.zeros(pos.shape[1], dtype=np.int8)   # all water -> 0

   which is correct for Water-3D, where every particle really is water, but
   here it would relabel the ramps as fluid. The obstacles would then vanish
   as obstacles and would also be counted as fluid by every compliance
   metric -- escape, kinetic energy, momentum. The error is silent: the v5
   converter prints a notice and carries on.

   Here the ramps are kept and mapped onto the project's boundary type id 1,
   which is the same id the synthesised container walls use. That is not a
   convenience: a ramp particle and a wall particle are the same kind of
   object for this method -- a static node whose state is an input.

Particle ordering
-----------------
Fluid particles are written first, then boundary particles. Downstream code
identifies boundaries by index (``i >= n_fluid``) when it drops
boundary-boundary edges and when it decides what to integrate, so the two
groups must not be interleaved. The raw files do interleave them.

Output per trajectory, ``traj_XXX.npz``:

    positions       float32 [T, N, 2]   fluid rows first, then boundary
    particle_type   int8    [N]         0 = fluid, 1 = boundary (ramp)
    n_fluid         int64   scalar
    bounds          float32 [2, 2]      metadata box, row 0 = lo, row 1 = hi
    ramp_drift      float32 scalar      max |position change| of a ramp
                                        particle across the trajectory

``ramp_drift`` is recorded rather than assumed: the whole one-way coupling
rests on these particles being static, so the claim is stored next to the
data it is made about.

Usage:
  python scripts/data_generation/v6_20261021_145750_convert_waterramps.py \
      --train 30 --valid 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_generation.gns_tfrecord_reader import read_trajectories

SRC = ROOT / "datasets" / "gns_datasets" / "WaterRamps"
OUT = ROOT / "datasets" / "gns_waterramps_v6"

GNS_WATER_TYPE = 5        # water
GNS_KINEMATIC_TYPE = 3    # static obstacle ("ramp")
FLUID, BOUNDARY = 0, 1    # this project's ids


def convert(split_file: str, split_name: str, n_traj: int, meta: dict,
            out_root: Path) -> list:
    dim = meta["dim"]
    if dim != 2:
        raise SystemExit("expected a 2D dataset, metadata says dim=%d" % dim)
    bounds = np.asarray(meta["bounds"], dtype=np.float32).T      # (2, dim)
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for idx, pos, ptype in read_trajectories(SRC / split_file, dim, n_traj):
        present = set(np.unique(ptype).tolist())
        unknown = present - {GNS_WATER_TYPE, GNS_KINEMATIC_TYPE}
        if unknown:
            raise SystemExit(
                "traj%d carries unhandled GNS particle types %s -- refusing to "
                "guess how they should be labelled" % (idx, sorted(unknown)))

        is_fluid = ptype == GNS_WATER_TYPE
        is_ramp = ptype == GNS_KINEMATIC_TYPE
        n_fluid = int(is_fluid.sum())
        n_ramp = int(is_ramp.sum())

        # Fluid first, then boundary -- downstream code splits on the index.
        order = np.concatenate([np.flatnonzero(is_fluid),
                                np.flatnonzero(is_ramp)])
        pos = np.ascontiguousarray(pos[:, order]).astype(np.float32)
        types = np.full(pos.shape[1], FLUID, dtype=np.int8)
        types[n_fluid:] = BOUNDARY

        if n_ramp:
            ramp = pos[:, n_fluid:]
            drift = float(np.abs(ramp[1:] - ramp[:-1]).max())
        else:
            drift = 0.0

        path = out_dir / ("traj_%03d.npz" % idx)
        np.savez(path, positions=pos, particle_type=types,
                 n_fluid=np.int64(n_fluid), bounds=bounds,
                 ramp_drift=np.float32(drift))
        rows.append((idx, pos.shape[0], n_fluid, n_ramp, drift))
        print("  traj%-3d frames=%d fluid=%4d ramp=%3d drift=%.1e -> %s (%.0f MB)"
              % (idx, pos.shape[0], n_fluid, n_ramp, drift, path.name,
                 path.stat().st_size / 1e6), flush=True)

    print("%s: %d trajectories in %.0fs" % (split_name, len(rows),
                                            time.time() - t0))
    return rows


def main():
    ap = argparse.ArgumentParser(description="convert WaterRamps to npz")
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--valid", type=int, default=10)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--src", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    global SRC
    if args.src:
        SRC = Path(args.src)
    out_root = Path(args.out) if args.out else OUT

    meta = json.loads((SRC / "metadata.json").read_text())
    print("WaterRamps: dim=%d seq_len=%d radius=%g dt=%g bounds=%s"
          % (meta["dim"], meta["sequence_length"],
             meta["default_connectivity_radius"], meta["dt"], meta["bounds"]))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2))

    summary = {}
    for split, n in (("train", args.train), ("valid", args.valid),
                     ("test", args.test)):
        if n <= 0:
            continue
        print("\n[%s]" % split)
        rows = convert("%s.tfrecord" % split, split, n, meta, out_root)
        summary[split] = [
            {"traj": r[0], "frames": r[1], "n_fluid": r[2], "n_ramp": r[3],
             "ramp_drift": r[4]} for r in rows]

    drifts = [r["ramp_drift"] for rs in summary.values() for r in rs]
    n_with = sum(1 for rs in summary.values() for r in rs if r["n_ramp"] > 0)
    total = sum(len(rs) for rs in summary.values())
    print("\n%d/%d trajectories carry ramp particles" % (n_with, total))
    print("max ramp drift over everything converted: %.3e" % max(drifts or [0.0]))
    if max(drifts or [0.0]) > 0.0:
        print("WARNING: ramp particles are NOT exactly static in this data; the "
              "one-way coupling assumes they are")
    (out_root / "convert_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote %s" % (out_root / "convert_summary.json"))


if __name__ == "__main__":
    main()
