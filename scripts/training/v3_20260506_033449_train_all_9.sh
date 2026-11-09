#!/bin/bash
# Master script: train all 9 configurations on v2 dataset, sequentially.
# Estimated total wall time: ~20 hours (100 epochs × ~80s/ep × 9 models).
# Run with:
#   bash scripts/training/v3_20260506_033449_train_all_9.sh > logs/v3_20260506_033449/MASTER.log 2>&1 &

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATASET_ID="v3_20260506_033449"
LOG_DIR="logs/$DATASET_ID"
mkdir -p "$LOG_DIR"

ARCHS=(gnn gat hybrid)
LOSSES=(vanilla physics improved)
EPOCHS=100

echo "=== START $(date) ==="
echo "Total configs: ${#ARCHS[@]} x ${#LOSSES[@]} = $(( ${#ARCHS[@]} * ${#LOSSES[@]} ))"
echo "Output dir: $LOG_DIR"

i=0
for arch in "${ARCHS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    i=$((i + 1))
    LOG="$LOG_DIR/run_${arch}_${loss}.log"
    echo
    echo "=== [$i/9] arch=$arch loss=$loss  $(date) ==="
    echo "    log -> $LOG"
    python scripts/training/v3_20260506_033449_train_unified.py \
        --arch "$arch" --loss "$loss" --epochs "$EPOCHS" \
        > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "    FAILED (exit $rc) — see $LOG" >&2
    else
      echo "    OK   $(date)"
    fi
  done
done

echo
echo "=== ALL 9 DONE $(date) ==="
ls -la models/*v3_20260506_033449.pt
