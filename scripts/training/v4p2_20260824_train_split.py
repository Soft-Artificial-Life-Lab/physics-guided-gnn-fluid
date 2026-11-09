"""
v4p2 training - reviewer-requested generalization experiments on the OWN
dataset (flip_simulations_v3p2), addressing review point (1):

  "인접 프레임 단위 분할은 동일 궤적 정보가 학습·검증에 중복될 수 있다.
   시뮬레이션 단위 분할과 미관측 조건 일반화 결과를 추가 권고."

Version ID : v4p2_20260824
Base       : v4.1 wall driver (v4p1_20260712_001555_train_wall.py). Model,
             recipe, wall particles, hyperparameters are all identical to the
             published wall_hybrid configuration - ONLY the train/val split
             changes. Importing the v4p1 module also applies its pushforward
             monkeypatch to v4.

Split modes
-----------
--split-mode sim   : hold out ENTIRE simulations. Every scenario type keeps
                     its --val-seed replicate (default 004) for validation;
                     the other four replicates train. -> 24 train / 6 val sims.
                     No trajectory appears on both sides.
--split-mode type  : hold out an entire scenario TYPE (--holdout-type, e.g.
                     sph_sphere_bounce). The model never sees that dynamics
                     regime; rollout on those 5 sims measures unseen-condition
                     generalization.

Usage:
  python scripts/training/v4p2_20260824_train_split.py --tag simsplit_hybrid \
      --split-mode sim
  python scripts/training/v4p2_20260824_train_split.py --tag typeout_bounce \
      --split-mode type --holdout-type sph_sphere_bounce
"""
import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "training"))

v4 = importlib.import_module("v4_20260707_102544_train_v4")
v4p1 = importlib.import_module("v4p1_20260712_001555_train_wall")  # patches v4
from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

V4P2_VERSION_ID = "v4p2_20260824"
DATA_DIR = v4.DATA_DIR


def scenario_type(sim_name):
    """sph_falling_ball_003 -> sph_falling_ball"""
    return sim_name.rsplit("_", 1)[0]


def main():
    p = argparse.ArgumentParser(description="v4p2 split-generalization training")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--split-mode", choices=["sim", "type"], required=True)
    p.add_argument("--val-seed", type=str, default="004",
                   help="sim mode: replicate suffix held out per type")
    p.add_argument("--holdout-type", type=str, default="sph_sphere_bounce",
                   help="type mode: scenario type excluded from training")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--pushforward-prob", type=float, default=0.3)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--wall-spacing", type=float, default=0.1)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--no-type-weighting", action="store_true")
    p.add_argument("--boundary-importance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    train_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = ROOT / "models" / ("hybrid_" + args.tag + "_" + train_ts
                                   + "_" + V4P2_VERSION_ID + ".pt")
    log_dir = ROOT / "logs" / V4P2_VERSION_ID
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("training_" + args.tag + "_" + train_ts + ".json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_dirs = sorted(d for d in Path(DATA_DIR).iterdir()
                      if d.is_dir() and d.name.startswith("sph_"))
    if args.split_mode == "sim":
        val_dirs = [d for d in sim_dirs if d.name.endswith("_" + args.val_seed)]
        train_dirs = [d for d in sim_dirs if d not in val_dirs]
        split_desc = "sim-level (val = *_%s, one replicate per type)" % args.val_seed
    else:
        val_dirs = [d for d in sim_dirs
                    if scenario_type(d.name) == args.holdout_type]
        train_dirs = [d for d in sim_dirs if d not in val_dirs]
        split_desc = "type-holdout (%s never trained)" % args.holdout_type
    assert val_dirs and train_dirs, "split produced an empty side"

    bar = "=" * 70
    print("\n" + bar)
    print("V4P2 SPLIT TRAINING  tag=%s  version=%s" % (args.tag, V4P2_VERSION_ID))
    print("  split: %s" % split_desc)
    print("  train sims=%d  val sims=%d" % (len(train_dirs), len(val_dirs)))
    print("  val: %s" % ", ".join(d.name for d in val_dirs))
    print("  device=%s  epochs=%d  ckpt -> %s" % (device, args.epochs, ckpt_path.name))
    print(bar)

    def build_sets(dirs):
        sets = []
        for sd in dirs:
            walls = v4p1.make_wall_particles(sd, spacing=args.wall_spacing)
            sets.append(v4p1.WallDataset(sd, walls, radius=args.radius,
                                         history=args.history))
        return sets

    train_sets, val_sets = build_sets(train_dirs), build_sets(val_dirs)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=1, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=1, shuffle=False)
    print("Train samples: %d  Val samples: %d (held-out simulations)"
          % (len(train_loader.dataset), len(val_loader.dataset)))

    model = HybridPhysicsGNNv4(
        history=args.history,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_learned_attention=not args.no_attention,
        use_type_weighting=not args.no_type_weighting,
        boundary_importance=args.boundary_importance,
    ).to(device)
    print("Model params: %d" % model.count_parameters())

    stats = v4.compute_normalizer_stats(train_sets, args.history)
    model.set_normalizer_stats(
        stats["node_mean"], stats["node_std"],
        stats["edge_mean"], stats["edge_std"],
        stats["target_mean"], stats["target_std"])
    (log_dir / ("normalizer_" + args.tag + "_" + train_ts + ".json")).write_text(
        json.dumps({k: v_.tolist() for k, v_ in stats.items()}, indent=2))

    criterion = v4.V4Loss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    history = {"train": [], "val": [], "config": vars(args),
               "version": V4P2_VERSION_ID,
               "split_desc": split_desc,
               "train_sims": [d.name for d in train_dirs],
               "val_sims": [d.name for d in val_dirs],
               "n_params": model.count_parameters()}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        tr, trc = v4.run_epoch(model, train_loader, criterion, optimizer, device,
                               train=True, noise_std=args.noise_std,
                               pushforward_prob=args.pushforward_prob,
                               radius=args.radius, accum_steps=args.accum_steps,
                               rng=rng)
        va, vac = v4.run_epoch(model, val_loader, criterion, optimizer, device,
                               train=False)
        scheduler.step()
        history["train"].append({"epoch": epoch, "loss": tr, **trc})
        history["val"].append({"epoch": epoch, "loss": va, **vac})

        improved = va < best_val
        if improved:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "model_config": model.get_config(),
                        "train_config": vars(args),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "val_loss": va,
                        "val_sims": [d.name for d in val_dirs],
                        "version": V4P2_VERSION_ID}, ckpt_path)
        print("[v4p2/%s] ep %3d/%d train=%.5f val=%.5f  "
              "d=%.4f p=%.4f m=%.4f e=%.4f  (%.1fs) %s"
              % (args.tag, epoch, args.epochs, tr, va,
                 vac["delta_loss"], vac["pos_loss"],
                 vac["momentum_loss"], vac["energy_loss"],
                 time.time() - ep_t, "(best)" if improved else ""), flush=True)
        log_path.write_text(json.dumps(history, indent=2))

    print("\nDone in %.1f min   best val = %.5f" % ((time.time() - t0) / 60, best_val))
    print("Best ckpt: %s" % ckpt_path)


if __name__ == "__main__":
    main()
