#!/bin/bash
# Resume v3.2 training for the 5 remaining configs.
# gnn_*, gat_vanilla already done. Skip gat_physics partial JSON.
# Each config runs in its own python invocation, so GPU memory is fully
# released between models.

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATASET_ID="v3p2_20260506_043045"
LOG_DIR="logs/$DATASET_ID"
mkdir -p "$LOG_DIR"

# Remaining configs
CONFIGS=(
  "gat physics"
  "gat improved"
  "hybrid vanilla"
  "hybrid physics"
  "hybrid improved"
)
EPOCHS=100

echo "=== RESUME $(date) ==="

# Remove the stale partial gat_physics log
rm -f "$LOG_DIR"/training_gat_physics_*.json
rm -f models/gat_physics_*_${DATASET_ID}.pt

i=0
for cfg in "${CONFIGS[@]}"; do
  arch=$(echo "$cfg" | cut -d' ' -f1)
  loss=$(echo "$cfg" | cut -d' ' -f2)
  i=$((i + 1))
  LOG="$LOG_DIR/run_${arch}_${loss}.log"
  echo
  echo "=== [$i/${#CONFIGS[@]}] arch=$arch loss=$loss  $(date) ==="
  echo "    log -> $LOG"
  python scripts/training/v3p2_20260506_043045_train_unified.py \
      --arch "$arch" --loss "$loss" --epochs "$EPOCHS" \
      > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAILED (exit $rc) — see $LOG" >&2
  else
    echo "    OK   $(date)"
  fi
done

echo
echo "=== ALL 5 RESUMED DONE $(date) ==="
ls -la models/*${DATASET_ID}.pt
