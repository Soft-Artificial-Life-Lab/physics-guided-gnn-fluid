"""
Validate saved Hybrid models using the EXACT validation method from the
training scripts (scripts/training/train_hybrid.py & train_dynamic_hybrid.py):
RigidBodySPHDataset + RigidBodyPhysicsLoss + validate() function.

Two hybrid models exist:
  A) models/hybrid_rigid_body_sph_best.pt
        trained on datasets/flip_simulations/sph_*   (data no longer present)
  B) scripts/training/models/hybrid_dynamic_two_way_coupling_best.pt
        trained on datasets/flip_simulations/flip_dynamic_*   (still present)

For (A) we can only report the in-checkpoint val-loss history.
For (B) we can re-run the same validate() loop on the live data.
"""
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from src.models.hybrid_gnn import HybridPhysicsGNN
from train_hybrid import RigidBodySPHDataset, RigidBodyPhysicsLoss, validate


def report_recorded(ckpt_path: Path, title: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"\n[{title}] recorded in {ckpt_path.name}")
    print(f"  best epoch         : {ckpt['epoch']+1}")
    print(f"  best val_loss      : {ckpt['val_loss']:.6f}")
    print(f"  val_losses history : {[round(v,6) for v in ckpt['val_losses']]}")
    print(f"  train_losses       : {[round(v,6) for v in ckpt['train_losses']]}")
    return ckpt


def rerun_validate(ckpt_path: Path, scenario_filter, data_root: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']['model']
    radius = ckpt['config']['training']['radius']
    dt = ckpt['config']['training']['dt']

    model = HybridPhysicsGNN(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scenario_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and any(f in d.name for f in scenario_filter)
    ])
    print(f"\n  Matched {len(scenario_dirs)} scenarios: "
          f"{[d.name for d in scenario_dirs]}")

    val_datasets = []
    torch.manual_seed(42)
    for d in scenario_dirs:
        ds = RigidBodySPHDataset(d, radius=radius)
        if len(ds) < 2:
            continue
        split_idx = int(0.8 * len(ds))
        _, val_sub = torch.utils.data.random_split(
            ds, [split_idx, len(ds) - split_idx]
        )
        val_datasets.append(val_sub)

    val_loader = DataLoader(ConcatDataset(val_datasets),
                            batch_size=1, shuffle=False)
    total = sum(len(v) for v in val_datasets)
    print(f"  Total val sequences: {total}")

    criterion = RigidBodyPhysicsLoss(
        vel_weight=1.0, pos_weight=0.1, momentum_weight=0.05, dt=dt
    )

    t0 = time.time()
    val_loss, comp = validate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(f"\n  RE-MEASURED val_loss : {val_loss:.6f}")
    print(f"  vel_loss             : {comp['vel_loss']:.6f}")
    print(f"  pos_loss             : {comp['pos_loss']:.9f}")
    print(f"  momentum_loss        : {comp['momentum_loss']:.6f}")
    print(f"  inference            : {elapsed:.1f}s "
          f"({elapsed*1000/total:.2f} ms/sample)")
    print(f"  recorded best        : {ckpt['val_loss']:.6f}")
    return val_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = ROOT / "datasets" / "flip_simulations"
    print("="*70)
    print(f"HYBRID VALIDATION  (device={device})")
    print("="*70)

    # (A) Static-scene hybrid ----------------------------------------
    ckpt_a = ROOT / "models" / "hybrid_rigid_body_sph_best.pt"
    print("\n" + "-"*70)
    print("(A) hybrid_rigid_body_sph_best.pt")
    print("-"*70)
    report_recorded(ckpt_a, "A")
    # training scenarios (sph_*) no longer in datasets/flip_simulations
    static_filter = ['sph_falling_ball', 'sph_paddle_wave',
                     'sph_rotating_stirrer', 'sph_moving_obstacle',
                     'sph_double_paddle', 'sph_sphere_bounce']
    present = [d for d in data_root.iterdir()
               if d.is_dir() and any(f in d.name for f in static_filter)]
    print(f"\n  Original training scenes present on disk: {len(present)}")
    print("  -> cannot re-run validate() faithfully; authoritative number is the")
    print(f"     checkpoint's best val_loss above.")

    # (B) Dynamic two-way-coupling hybrid ---------------------------
    ckpt_b = ROOT / "scripts" / "training" / "models" / "hybrid_dynamic_two_way_coupling_best.pt"
    print("\n" + "-"*70)
    print("(B) hybrid_dynamic_two_way_coupling_best.pt")
    print("-"*70)
    report_recorded(ckpt_b, "B")
    dynamic_filter = ['dynamic_falling_ball', 'dynamic_floating_ball',
                      'dynamic_heavy_box', 'dynamic_multi_ball',
                      'dynamic_bouncing_sphere']
    rerun_validate(ckpt_b, dynamic_filter, data_root, device)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
