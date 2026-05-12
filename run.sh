#!/bin/bash
set -euo pipefail
# ============================================================
# Unified experiment launcher for Tiger-CL
# Usage: bash run.sh <dataset> <method> [gpu_id] [extra_args...]
#
# Datasets: Toys_and_Games, Video_Games, CDs_and_Vinyl, Books
#
# Methods:
#   baseline_h2       — T5 baseline, history=2
#   baseline_h10      — T5 baseline, history=10
#   baseline_h20      — T5 baseline, history=20
#   routing           — Cross-attention routing, h=10
#   routing_aux       — Routing + auxiliary loss, h=10
#   pkm               — T5 + Product-Key Memory, h=10
#   pkm_routing_aux   — PKM + Routing + Aux (full combo), h=10
#   all               — Run all experiments sequentially
#
# Examples:
#   bash run.sh Toys_and_Games baseline_h10 0
#   bash run.sh CDs_and_Vinyl routing_aux 1
#   bash run.sh Video_Games all 0
# ============================================================

DATASET=${1:?Usage: bash run.sh <dataset> <method> [gpu_id]}
METHOD=${2:?Usage: bash run.sh <dataset> <method> [gpu_id]}
GPU_ID=${3:-0}
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Dataset → time range
case $DATASET in
    Toys_and_Games)   TIME_RANGE="2016-10-2018-11" ;;
    Video_Games)      TIME_RANGE="2012-10-2018-11" ;;
    CDs_and_Vinyl)    TIME_RANGE="2014-10-2018-11" ;;
    Books)            TIME_RANGE="2016-10-2018-11" ;;
    *)  echo "Error: Unknown dataset "
        echo "Supported: Toys_and_Games, Video_Games, CDs_and_Vinyl, Books"
        exit 1 ;;
esac

# Method → sweep labels
case $METHOD in
    baseline_h2)      LABELS="h2_t5" ;;
    baseline_h10)     LABELS="h10_t5" ;;
    baseline_h20)     LABELS="h20_t5" ;;
    routing)          LABELS="h10_route_ffn" ;;
    routing_aux)      LABELS="h10_route_ffn_aux" ;;
    pkm)             LABELS="h10_pkm" ;;
    pkm_routing_aux) LABELS="h10_route_pkm_gate_aux" ;;
    all)             LABELS="" ;;
    *)  echo "Error: Unknown method "
        echo "Supported: baseline_h2, baseline_h10, baseline_h20, routing, routing_aux, pkm, pkm_routing_aux, all"
        exit 1 ;;
esac

echo "=========================================="
echo " Dataset: $DATASET"
echo " Method:  $METHOD"
echo " GPU:     $GPU_ID"
echo " Time:    $TIME_RANGE"
echo "=========================================="

# Check that data exists
DATA_DIR="$REPO_ROOT/data"
if [ ! -d "$DATA_DIR/D0" ]; then
    echo "Error: Data not found at $DATA_DIR/D0"
    echo "Run: bash scripts/setup_data.sh $DATASET"
    exit 1
fi

INDEX_FILE="$DATA_DIR/info/${DATASET}.TIGER-index.json"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Error: TIGER index not found at $INDEX_FILE"
    echo "Run: bash scripts/setup_data.sh $DATASET"
    exit 1
fi

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python $REPO_ROOT/docs/run_delta_set_sweep.py \
    --dataset $DATASET \
    --data_root $DATA_DIR \
    --amazon_root $DATA_DIR \
    --time_range $TIME_RANGE \
    --num_workers 1 --worker_id 0"

if [ -n "$LABELS" ]; then
    CMD="$CMD --labels $LABELS"
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
echo ""
eval $CMD
