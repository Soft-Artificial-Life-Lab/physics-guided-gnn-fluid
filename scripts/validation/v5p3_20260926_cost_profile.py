# -*- coding: utf-8 -*-
"""Measure the computational cost that the boundary particles add.

Reports, per Water-3D validation frame and per boundary configuration, the node
count, the edge count after wall-wall edges are dropped, and the wall-clock time
of one forward pass on the GPU. The numbers go into the cost table of the paper
appendix, so nothing here is estimated: every value is timed on the same device
that produced the training runs.

Usage:
    python scripts/validation/v5p3_20260926_cost_profile.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))

import importlib

v5 = importlib.import_module("v5_20260820_022909_train_water3d")

DATA = ROOT / "datasets" / "gns_water3d_v5"
RADIUS = 0.035
C_HIST = 3
WARMUP, REPEAT = 5, 20


def radius_graph(pos, r, types, drop_boundary_pairs):
    """Bidirectional edges within r; optionally drop boundary-boundary pairs."""
    n = pos.shape[0]
    idx = torch.arange(n, device=pos.device)
    # chunked to keep the pairwise distance matrix off the 24 GB budget
    src, dst = [], []
    chunk = 4096
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = torch.cdist(pos[a:b], pos)
        m = (d < r) & (d > 0)
        if drop_boundary_pairs:
            bb = (types[a:b, None] > 0.5) & (types[None, :] > 0.5)
            m &= ~bb
        ii, jj = m.nonzero(as_tuple=True)
        src.append(idx[a:b][ii])
        dst.append(jj)
    return torch.stack([torch.cat(src), torch.cat(dst)])


def profile(tag, pos_fluid, walls, device):
    if walls is None:
        pos = pos_fluid
        types = torch.zeros(pos.shape[0], device=device)
    else:
        pos = torch.cat([pos_fluid, walls], 0)
        types = torch.cat([torch.zeros(pos_fluid.shape[0], device=device),
                           torch.ones(walls.shape[0], device=device)])

    ei = radius_graph(pos, RADIUS, types, drop_boundary_pairs=walls is not None)
    n, e = pos.shape[0], ei.shape[1]

    model = v5.v4.HybridPhysicsGNNv4(
        history=C_HIST, input_edge_dim=4, hidden_dim=64, num_layers=10,
        boundary_importance=2.0).to(device).eval()

    rel = pos[ei[1]] - pos[ei[0]]
    ea = torch.cat([rel, rel.norm(dim=1, keepdim=True)], 1)
    nf = torch.cat([torch.zeros(n, 3 * C_HIST, device=device),
                    types.unsqueeze(-1)], -1)

    with torch.no_grad():
        for _ in range(WARMUP):
            model.predict_next_velocity(nf, ei, ea)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            model.predict_next_velocity(nf, ei, ea)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / REPEAT * 1e3

    print("%-26s nodes=%7d  edges=%9d  forward=%7.2f ms" % (tag, n, e, ms))
    return {"tag": tag, "nodes": n, "edges": e, "forward_ms": ms}


def main():
    device = torch.device("cuda")
    meta = json.loads((DATA / "metadata.json").read_text())
    bounds = np.array(meta["bounds"], dtype=np.float64).T \
        if len(meta["bounds"]) == 3 else np.array(meta["bounds"])
    f = sorted((DATA / "valid").glob("*.npz"))[0]
    d = np.load(f)
    w = v5.make_box_walls(bounds, spacing=RADIUS / 2, margin=0.010, layers=2)
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    rows = []
    for frame in (0, 200, 400):
        pos = torch.tensor(d["positions"][frame], dtype=torch.float32,
                           device=device)
        rows.append(profile("frame %3d, no walls" % frame, pos, None, device))
        rows.append(profile("frame %3d, walls" % frame, pos, wt, device))

    out = ROOT / "logs" / "v5p2_20260922" / "cost_profile.json"
    out.write_text(json.dumps({"radius": RADIUS, "trajectory": f.name,
                               "rows": rows}, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
