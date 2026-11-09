"""
v4 rollout visualisation — GT vs prediction side-by-side 3D GIF.

Version ID : v4_20260707_102544

Reads the trajectory .npz files saved by the v4 rollout evaluator
(--save-traj) and renders a side-by-side comparison animation:
  left  = ground-truth SPH simulation
  right = v4 autoregressive rollout (same seed frames)
Fluid particles blue, rigid/boundary particles red.

Usage:
  python scripts/visualization/v4_20260707_102544_render_rollout.py            # all saved trajs
  python scripts/visualization/v4_20260707_102544_render_rollout.py --match falling_ball
"""
import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
V4_VERSION_ID = "v4_20260707_102544"

DATASET_ID = "v3p2_20260506_043045"
DATA_DIR = ROOT / "datasets" / f"flip_simulations_{DATASET_ID}"
TRAJ_DIR = ROOT / "logs" / V4_VERSION_ID
OUT_DIR = ROOT / "visualizations" / V4_VERSION_ID
HISTORY = 3          # rollout was seeded with the first `history` GT frames


def load_gt(sim_dir, start, n):
    files = sorted(Path(sim_dir).glob("frame_*.npz"))[start: start + n]
    return np.stack([np.load(f)["positions"].astype(np.float32) for f in files])


def render(traj_file, fps=12, stride=1):
    m = re.search(r"rollout_.+_(sph_\w+_\d{3})\.npz$", traj_file.name)
    if not m:
        print(f"  skip (name not parseable): {traj_file.name}")
        return
    scenario = m.group(1)
    sim_dir = DATA_DIR / scenario
    if not sim_dir.exists():
        print(f"  skip (no GT dir): {scenario}")
        return

    d = np.load(traj_file)
    pred = d["positions"]                    # (T+1, N, 3), frame 0 == GT frame HISTORY-1
    types = d["particle_types"]
    gt = load_gt(sim_dir, HISTORY - 1, pred.shape[0])
    T = min(len(gt), len(pred))
    fluid, rigid = types < 0.5, types >= 0.5

    lo, hi = gt.reshape(-1, 3).min(0) - 0.05, gt.reshape(-1, 3).max(0) + 0.05
    frames = range(0, T, stride)

    # Dataset convention: Y is height (gravity along -Y). Map data (x, y, z)
    # -> plot (x, z, y) so that "up" is vertical in the 3D view.
    fig = plt.figure(figsize=(11, 5.5))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    titles = ["Ground Truth (SPH)", "v4 Hybrid GNN rollout"]
    scats = []
    for ax, title in zip(axes, titles):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.set_title(title, fontsize=11)
        ax.set_box_aspect(tuple((hi - lo)[[0, 2, 1]]))
        s_f = ax.scatter([], [], [], s=2.5, c="#1f77b4", alpha=0.6, depthshade=False)
        s_r = ax.scatter([], [], [], s=6.0, c="#d62728", alpha=0.9, depthshade=False)
        scats.append((s_f, s_r))
    sup = fig.suptitle("", fontsize=12)
    fig.tight_layout()

    def update(i):
        for (s_f, s_r), pos in zip(scats, (gt[i], pred[i])):
            s_f._offsets3d = (pos[fluid, 0], pos[fluid, 2], pos[fluid, 1])
            s_r._offsets3d = (pos[rigid, 0], pos[rigid, 2], pos[rigid, 1])
        sup.set_text(f"{scenario}   step {i}/{T - 1}")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rollout_{scenario}.gif"
    t0 = time.time()
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  {out.name}  ({len(list(frames))} frames, {time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", type=str, default="",
                   help="only render trajectories whose filename contains this")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--stride", type=int, default=1, help="render every k-th step")
    args = p.parse_args()

    trajs = sorted(TRAJ_DIR.glob("rollout_*.npz"))
    if args.match:
        trajs = [t for t in trajs if args.match in t.name]
    print(f"Rendering {len(trajs)} trajectory file(s) -> {OUT_DIR}")
    for t in trajs:
        render(t, fps=args.fps, stride=args.stride)


if __name__ == "__main__":
    main()
