#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DATASET_ID="v3p2_20260506_043045"
LOG_DIR="logs/$DATASET_ID"
mkdir -p "$LOG_DIR"
LOSSES=(vanilla physics improved)
EPOCHS=100
echo "=== START $(date) ==="
i=0
for loss in "${LOSSES[@]}"; do
  i=$((i + 1))
  LOG="$LOG_DIR/run_hybrid_tuned_${loss}.log"
  echo
  echo "=== [$i/${#LOSSES[@]}] arch=hybrid_tuned loss=$loss  $(date) ==="
  python scripts/training/v3p2_20260506_043045_train_unified.py \
      --arch hybrid_tuned --loss "$loss" --epochs "$EPOCHS" > "$LOG" 2>&1
  rc=$?
  echo "    exit=$rc  $(date)"
done
echo
echo "=== ALL TUNED HYBRID DONE $(date) ==="
ls -la models/hybrid_tuned_*_${DATASET_ID}.pt
