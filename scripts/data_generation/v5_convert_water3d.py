# -*- coding: utf-8 -*-
"""Convert a subset of DeepMind GNS Water-3D into per-trajectory .npz files.

Water-3D ships as SequenceExample TFRecords holding only positions; velocities
are the frame-to-frame position deltas (GNS convention, so dt = 1 in model
units) and every particle is water (type 5) — the container is implicit in the
metadata bounds rather than represented by boundary particles.

Output (one file per trajectory):
    positions       float32 [T, N, 3]
    particle_type   int8    [N]      0 = fluid  (GNS type 5 remapped)
    bounds          float32 [2, 3]   domain box from metadata

Usage:
    python scripts/data_generation/v5_convert_water3d.py --train 10 --valid 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

from gns_tfrecord_reader import read_trajectories  # noqa: E402

SRC = ROOT / "datasets" / "gns_datasets" / "Water-3D"
OUT = ROOT / "datasets" / "gns_water3d_v5"
GNS_WATER_TYPE = 5


def convert(split_file: str, split_name: str, n_traj: int, meta: dict) -> int:
    dim = meta["dim"]
    bounds = np.asarray(meta["bounds"], dtype=np.float32).T   # [2, 3] = (lo, hi)
    out_dir = OUT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    t0 = time.time()
    for idx, pos, ptype in read_trajectories(SRC / split_file, dim, n_traj):
        types = np.unique(ptype)
        if not np.array_equal(types, [GNS_WATER_TYPE]):
            print(f"  traj{idx}: unexpected particle types {types.tolist()} — kept as fluid")
        fluid = np.zeros(pos.shape[1], dtype=np.int8)          # all water -> 0

        path = out_dir / f"traj_{idx:03d}.npz"
        np.savez(path, positions=pos.astype(np.float32),
                 particle_type=fluid, bounds=bounds)
        kept += 1
        print(f"  traj{idx}: frames={pos.shape[0]} particles={pos.shape[1]} "
              f"-> {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    print(f"{split_name}: {kept} trajectories in {time.time() - t0:.0f}s")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--valid", type=int, default=3)
    args = ap.parse_args()

    meta = json.loads((SRC / "metadata.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Water-3D: dim={meta['dim']} seq_len={meta['sequence_length']} "
          f"radius={meta['default_connectivity_radius']} bounds={meta['bounds']}")

    convert("train.tfrecord", "train", args.train, meta)
    convert("valid.tfrecord", "valid", args.valid, meta)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
