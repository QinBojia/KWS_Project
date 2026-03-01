#!/bin/bash
# Stage 2c: Retrain top unique architectures from 2b with 1000 epochs + early stop
# No new candidate generation - just longer training on the winners
set -e
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
LOG="experiments/grid_search/round2c.log"

echo "========== Stage 2c: Finals, 1000ep + optimizations ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

mkdir -p experiments/grid_search/round2c

$PY -u scripts/grid_search_2c_final.py --seed 42 2>&1 | tee -a "$LOG"

echo "Done: $(date)" | tee -a "$LOG"
