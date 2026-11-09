#!/usr/bin/env bash
# Evaluate the corrected-geometry GNS baseline at 77 and 400 steps, then print
# it beside the configurations it has to be compared with. Escape is judged
# against the dataset metadata box in every row (see the evaluator patch), so
# the corrected reference box does not also buy a looser criterion.
set -u
cd "$(dirname "$0")/../.."
CKPT=$(ls -t models/gns_gns_corr_e100_*.pt | head -1)
OUT=logs/v5p3_20260926

python scripts/validation/v5p1_20260903_rollout_gns.py --ckpt "$CKPT" --steps 77 \
  --out "$OUT/rollout_gns_corr_77step.json"
python scripts/validation/v5p1_20260903_rollout_gns.py --ckpt "$CKPT" --steps 400 \
  --out "$OUT/rollout_gns_corr_400step.json"

python - <<'PY'
import json
rows = [
 ("GNS, metadata box (published)",
  "logs/v5p1_20260903/rollout_gns_gns_e100_20260904_001444_v5p1_20260903_400step.json"),
 ("GNS, corrected box (new)",
  "logs/v5p3_20260926/rollout_gns_corr_400step.json"),
 ("Boundary particles, metadata box (m000)",
  "logs/v5p2_20260922/rollout_w3d_sweep_m000_e100_400step.json"),
 ("Boundary particles, corrected (proposed)",
  "logs/v5_20260820_022909/rollout_wall2_e100_400step_tol.json"),
 ("No walls",
  "logs/v5_20260820_022909/rollout_nowall_e100_400step_tol.json"),
]
print()
print("%-42s %9s %10s %9s" % ("config (seed 42, step 400)", "escape%", "|KE-1|", "posMSE"))
print("-" * 74)
for name, path in rows:
    try:
        s = json.load(open(path, encoding="utf-8"))["summary"]
    except FileNotFoundError:
        print("%-42s %9s" % (name, "missing")); continue
    print("%-42s %9.2f %10.1f %9.3f" % (name, s["domain_frac"]["mean"] * 100,
                                        s["ke_dev"]["mean"], s["pos_mse"]["mean"]))
PY
