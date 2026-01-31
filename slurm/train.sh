#!/bin/bash
set -euo pipefail
# proxy
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export NO_PROXY=localhost,127.0.0.1,.byted.org,byted.org,.bytedance.net,bytedance.net
export no_proxy=$NO_PROXY

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

CODE_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev
cd "${CODE_ROOT}"

DATASET=Toys_and_Games

# Set this to the directory produced by 0_process.py (the one containing D0/D1/D2/D3/D4).
DATA_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data

# Set this to the directory that contains data/info/${DATASET}.TIGER-index.json (i.e., the one containing the info/ subdir).
AMAZON_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data

# Tokenizer source for training/evaluation (decoder-only currently still uses T5Tokenizer).
BASE_MODEL=google-t5/t5-small

# Strategy for train.py: t5_seq2seq / decoder_only
STRATEGY=decoder_only
TEST_MODEL_TYPE=decoder_only

LR=3e-4
WD=0.001
SUFFIX=debug-4gpu-decoder-only
BATCH_SIZE=256
TEST_BATCH_SIZE=100
NUM_BEAMS=20

EPOCHS=3
MODEL_MAX_LENGTH=512
MAX_NEW_TOKENS=10

TIME_RANGE=2016-10-2018-11
INDEX_FILE=.TIGER-index.json
CONFIG_FILE=${CODE_ROOT}/configs/train_decoder_only_pkm.yaml

# -------------------------
# Decoder-only model hyperparameters (only effective when STRATEGY=decoder_only)
# -------------------------
D_MODEL=512
N_LAYERS=12
N_HEADS=8
FFN_DIM=2048
DROPOUT=0.0
ROPE_THETA=10000.0
MAX_SEQ_LEN=2048

# -------------------------
# PKM/HashingMemory hyperparameters (enable as needed)
# Note: train.py/test.py no longer support legacy --pk_* / store_true flags.
# -------------------------
PK_IS_ENABLED=1
PK_LAYERS="5"          # e.g., "1,3,5" (0-based layer indices)
PK_MEM_N_KEYS=128
PK_MEM_HEADS=4
PK_MEM_K_DIM=512
PK_MEM_V_DIM=-1
PK_TOPK=8
PK_MEM_GATED=0

mkdir -p ./log/${DATASET}/train
mkdir -p ./log/${DATASET}/test

DECODER_ONLY_OVERRIDES=()
if [[ "${STRATEGY}" == "decoder_only" ]]; then
  DECODER_ONLY_OVERRIDES+=(
    "model.decoder_only.d_model=${D_MODEL}"
    "model.decoder_only.n_layers=${N_LAYERS}"
    "model.decoder_only.n_heads=${N_HEADS}"
    "model.decoder_only.ffn_dim=${FFN_DIM}"
    "model.decoder_only.dropout=${DROPOUT}"
    "model.decoder_only.rope_theta=${ROPE_THETA}"
    "model.decoder_only.max_seq_len=${MAX_SEQ_LEN}"
  )

  # Write explicit false to avoid any legacy "store_true" behavior.
  DECODER_ONLY_OVERRIDES+=(
    "pkm.decoder_only.pk_is_enabled=false"
    "pkm.decoder_only.pk_mem_gated=false"
  )

  if [[ "${PK_IS_ENABLED}" == "1" ]]; then
    DECODER_ONLY_OVERRIDES+=(
      "pkm.decoder_only.pk_is_enabled=true"
      "pkm.decoder_only.pk_layers=${PK_LAYERS}"
      "pkm.decoder_only.pk_mem_n_keys=${PK_MEM_N_KEYS}"
      "pkm.decoder_only.pk_mem_heads=${PK_MEM_HEADS}"
      "pkm.decoder_only.pk_mem_k_dim=${PK_MEM_K_DIM}"
      "pkm.decoder_only.pk_mem_v_dim=${PK_MEM_V_DIM}"
      "pkm.decoder_only.pk_topk=${PK_TOPK}"
    )
    if [[ "${PK_MEM_GATED}" == "1" ]]; then
      DECODER_ONLY_OVERRIDES+=( "pkm.decoder_only.pk_mem_gated=true" )
    fi
  fi
fi

################################
# Stage 0: Train on D0 (1 GPU)
################################
PREV_CKPT=${CODE_ROOT}/ckpt/${DATASET}/${STRATEGY}-D0-${LR}lr-${WD}wd-${SUFFIX}
mkdir -p "${PREV_CKPT}"

torchrun --nproc_per_node=2 --master_port=2309 train.py \
  config="${CONFIG_FILE}" \
  "strategy=${STRATEGY}" \
  "model.${STRATEGY}.base_model=${BASE_MODEL}" \
  "train.output_dir=${PREV_CKPT}" \
  "dataset.data_path=${AMAZON_ROOT}" \
  "dataset.name=${DATASET}" \
  "dataset.train_file=${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.valid_file=${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.test_file=${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.index_file=${INDEX_FILE}" \
  "train.batch_size=${BATCH_SIZE}" \
  "train.learning_rate=${LR}" \
  "train.epochs=${EPOCHS}" \
  "train.weight_decay=${WD}" \
  "train.save_and_eval_strategy=epoch" \
  "train.model_max_length=${MODEL_MAX_LENGTH}" \
  "${DECODER_ONLY_OVERRIDES[@]}" \
  > "./log/${DATASET}/train/${STRATEGY}-D0-train-4gpu.log"

################################
# Forward tests (per-D grouped, single GPU sequential)
################################
train_d=0
CUR_CKPT="${PREV_CKPT}"

for test_d in $(seq $((train_d + 1)) 4); do
  echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users, 1 GPU) ====="

  GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
  GROUP_FILES=(${GROUP_DIR}/*.csv)

  for group_file in "${GROUP_FILES[@]}"; do
    group_name=$(basename "${group_file}" .csv)
    logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}-1gpu.log

    echo "  → D${test_d}, ${group_name}"

    python test.py \
      config="${CONFIG_FILE}" \
      "model.type=${TEST_MODEL_TYPE}" \
      "global.gpu_id=0" \
      "model.ckpt_path=${CUR_CKPT}" \
      "model.tokenizer_path=${CUR_CKPT}" \
      "model.base_model=${BASE_MODEL}" \
      "dataset.name=${DATASET}" \
      "dataset.data_path=${AMAZON_ROOT}" \
      "dataset.train_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
      "dataset.valid_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
      "dataset.test_file=${group_file}" \
      "dataset.index_file=${INDEX_FILE}" \
      "test.batch_size=${TEST_BATCH_SIZE}" \
      "test.num_beams=${NUM_BEAMS}" \
      "test.max_new_tokens=${MAX_NEW_TOKENS}" \
      "test.filter_items=true" \
      > "${logfile}"
  done

  echo "===== Finished D${test_d} grouped testing ====="
done

################################
# Stage 1 & 2: Continual finetune (1 GPU)
################################
for train_d in 1 2 3; do
  CUR_CKPT=${CODE_ROOT}/ckpt/${DATASET}/${STRATEGY}-D${train_d}-${LR}lr-${WD}wd-${SUFFIX}
  mkdir -p "${CUR_CKPT}"

  torchrun --nproc_per_node=2 --master_port=$((2310 + train_d)) train.py \
    config="${CONFIG_FILE}" \
    "strategy=${STRATEGY}" \
    "model.${STRATEGY}.base_model=${BASE_MODEL}" \
    "train.output_dir=${CUR_CKPT}" \
    "dataset.data_path=${AMAZON_ROOT}" \
    "dataset.name=${DATASET}" \
    "dataset.train_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
    "dataset.valid_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
    "dataset.test_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
    "dataset.index_file=${INDEX_FILE}" \
    "train.batch_size=${BATCH_SIZE}" \
    "train.learning_rate=${LR}" \
    "train.epochs=${EPOCHS}" \
    "train.weight_decay=${WD}" \
    "train.save_and_eval_strategy=epoch" \
    "train.model_max_length=${MODEL_MAX_LENGTH}" \
    "${DECODER_ONLY_OVERRIDES[@]}" \
    > "./log/${DATASET}/train/${STRATEGY}-D${train_d}-finetune-4gpu.log"

  for test_d in $(seq $((train_d + 1)) 4); do
    echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users, 1 GPU) ====="

    GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
    GROUP_FILES=(${GROUP_DIR}/*.csv)

    for group_file in "${GROUP_FILES[@]}"; do
      group_name=$(basename "${group_file}" .csv)
      logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}-1gpu.log

      echo "  → D${test_d}, ${group_name}"

      python test.py \
        config="${CONFIG_FILE}" \
        "model.type=${TEST_MODEL_TYPE}" \
        "global.gpu_id=0" \
        "model.ckpt_path=${CUR_CKPT}" \
        "model.tokenizer_path=${CUR_CKPT}" \
        "model.base_model=${BASE_MODEL}" \
        "dataset.name=${DATASET}" \
        "dataset.data_path=${AMAZON_ROOT}" \
        "dataset.train_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
        "dataset.valid_file=${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv" \
        "dataset.test_file=${group_file}" \
        "dataset.index_file=${INDEX_FILE}" \
        "test.batch_size=${TEST_BATCH_SIZE}" \
        "test.num_beams=${NUM_BEAMS}" \
        "test.max_new_tokens=${MAX_NEW_TOKENS}" \
        "test.filter_items=true" \
        > "${logfile}"
    done

    echo "===== Finished D${test_d} grouped testing ====="
  done

  PREV_CKPT="${CUR_CKPT}"
done

PLOT_DIR=./log/${DATASET}/plots
mkdir -p "${PLOT_DIR}"
python "${CODE_ROOT}/plot_test_results.py" \
  --log_dir "./log/${DATASET}/test" \
  --out_dir "${PLOT_DIR}" \
  --dataset "${DATASET}"