#!/bin/bash
set -euo pipefail
# ============================================================
# Setup data pipeline: download → process → RQ-VAE → TIGER index
# Usage: bash scripts/setup_data.sh <dataset> [device]
# Example: bash scripts/setup_data.sh Toys_and_Games cuda:0
# ============================================================

DATASET=${1:?Usage: bash scripts/setup_data.sh <dataset> [device]}
DEVICE=${2:-cuda:0}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$REPO_ROOT/raw_data"
DATA_DIR="$REPO_ROOT/data"
INFO_DIR="$DATA_DIR/info"

mkdir -p "$RAW_DIR" "$INFO_DIR"

# Dataset → time range mapping
case $DATASET in
    Toys_and_Games)   ST_YEAR=2016; ST_MONTH=10; ED_YEAR=2018; ED_MONTH=11 ;;
    Video_Games)      ST_YEAR=2012; ST_MONTH=10; ED_YEAR=2018; ED_MONTH=11 ;;
    CDs_and_Vinyl)    ST_YEAR=2014; ST_MONTH=10; ED_YEAR=2018; ED_MONTH=11 ;;
    Books)            ST_YEAR=2016; ST_MONTH=10; ED_YEAR=2018; ED_MONTH=11 ;;
    *)  echo "Unknown dataset: $DATASET"; echo "Supported: Toys_and_Games, Video_Games, CDs_and_Vinyl, Books"; exit 1 ;;
esac

TIME_RANGE="${ST_YEAR}-${ST_MONTH}-${ED_YEAR}-${ED_MONTH}"
FULL_NAME="${DATASET}_5_${TIME_RANGE}"

echo "=========================================="
echo " Dataset: $DATASET"
echo " Time range: $TIME_RANGE"
echo " Device: $DEVICE"
echo "=========================================="

# ── Step 1: Download raw data ──
echo "[Step 1/6] Downloading raw data..."
cd "$RAW_DIR"
BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"

if [ ! -f "${DATASET}.json" ]; then
    wget -q "${BASE_URL}/categoryFiles/${DATASET}.json.gz"
    gunzip "${DATASET}.json.gz"
    echo "  Downloaded ${DATASET}.json"
else
    echo "  ${DATASET}.json already exists, skipping"
fi

if [ ! -f "meta_${DATASET}.json" ]; then
    wget -q "${BASE_URL}/metaFiles2/meta_${DATASET}.json.gz"
    gunzip "meta_${DATASET}.json.gz"
    echo "  Downloaded meta_${DATASET}.json"
else
    echo "  meta_${DATASET}.json already exists, skipping"
fi

# ── Step 2: Process data (temporal split + K-core) ──
echo "[Step 2/6] Processing data (temporal split + K-core filtering)..."
cd "$REPO_ROOT/data"
python 0_process.py --category "$DATASET" --K 5 \
    --st_year $ST_YEAR --st_month $ST_MONTH \
    --ed_year $ED_YEAR --ed_month $ED_MONTH

# ── Step 3: Generate test groups ──
echo "[Step 3/6] Generating test groups..."
python 1_generate_group.py --data_root "$DATA_DIR" --dataset "${FULL_NAME}"

# ── Step 4: Generate user-group map ──
echo "[Step 4/6] Generating user-group map..."
python 2_generate_user_group_map.py --data_root "$DATA_DIR" --dataset_name "${FULL_NAME}" --n_groups 5

# ── Step 5: Generate T5 embeddings ──
echo "[Step 5/6] Generating item embeddings with T5..."
TDCB_PATH="${INFO_DIR}/${FULL_NAME}_combine_tdcb_maps.npy"
EMB_PATH="${INFO_DIR}/${DATASET}.emb-t5-tdcb.npy"

if [ ! -f "$EMB_PATH" ]; then
    cd "$REPO_ROOT/RQ-VAE"
    python generate_embeddings.py \
        --tdcb_path "$TDCB_PATH" \
        --output_path "$EMB_PATH" \
        --model_name google-t5/t5-small \
        --batch_size 128 --device "$DEVICE"
else
    echo "  Embeddings already exist, skipping"
fi

# ── Step 6: Train RQ-VAE + generate TIGER index ──
echo "[Step 6/6] Training RQ-VAE and generating TIGER index..."
INDEX_PATH="${INFO_DIR}/${DATASET}.TIGER-index.json"

if [ ! -f "$INDEX_PATH" ]; then
    cd "$REPO_ROOT/RQ-VAE"
    python main.py \
        --num_emb_list 256 256 256 256 \
        --sk_epsilons 0.0 0.0 0.0 0.003 \
        --device "$DEVICE" \
        --data_path "$EMB_PATH" \
        --dataset "$DATASET" \
        --batch_size 480
else
    echo "  TIGER index already exists, skipping"
fi

echo ""
echo "=========================================="
echo " Setup complete for $DATASET!"
echo " Data: $DATA_DIR/D0/ ~ D4/"
echo " Index: $INDEX_PATH"
echo "=========================================="
