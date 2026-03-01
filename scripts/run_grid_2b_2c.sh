#!/bin/bash
# Chain: 2b → 2c, runs unattended
# Usage: bash scripts/run_grid_2b_2c.sh

set -e
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
LOG_DIR="experiments/grid_search"

echo "========== Stage 2b: top-10 refinement, 200ep + optimizations =========="
echo "Start: $(date)"
mkdir -p "$LOG_DIR/round2b"
$PY -u scripts/grid_search_tenet.py --round 2b --top-n 10 --seed 42 \
    2>&1 | tee "$LOG_DIR/round2b.log"
echo "2b done: $(date)"

echo ""
echo "========== Stage 2c: top-10 finals, 1000ep + optimizations =========="
echo "Start: $(date)"
mkdir -p "$LOG_DIR/round2c"
$PY -u scripts/grid_search_tenet.py --round 2c --top-n 10 --seed 42 \
    2>&1 | tee "$LOG_DIR/round2c.log"
echo "2c done: $(date)"

echo ""
echo "========== ALL DONE =========="
echo "Results:"
echo "  2b: $LOG_DIR/round2b/summary.json"
echo "  2c: $LOG_DIR/round2c/summary.json"
