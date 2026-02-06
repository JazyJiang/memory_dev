#!/bin/bash
set -euo pipefail
# =================   proxy   ===============================
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export NO_PROXY=localhost,127.0.0.1,.byted.org,byted.org,.bytedance.net,bytedance.net
export no_proxy=$NO_PROXY
# =================   no tf   ===============================
export TRANSFORMERS_NO_TF=1
export USE_TF=0
# ===========================================================

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

CODE_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev
cd "${CODE_ROOT}"

DATASET=Toys_and_Games

# Set this to the directory produced by 0_process.py (the one containing D0/D1/D2/D3/D4).
DATA_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data

# Set this to the directory that contains data/info/${DATASET}.TIGER-index.json (i.e., the one containing the info/ subdir).
AMAZON_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data

BASE_MODEL=google-t5/t5-small
STRATEGY=t5_seq2seq
TEST_MODEL_TYPE=t5_seq2seq

TIME_RANGE=2016-10-2018-11
INDEX_FILE=.TIGER-index.json
CONFIG_FILE=${CODE_ROOT}/configs/train_decoder_only_pkm.yaml

# -------------------------
# Experiment setting
# -------------------------
# 从已训好的 D0（含 PKM）作为初始化，然后只训练 V，在 D1 上训练，再测 D2-D4
INIT_FROM_CKPT=${CODE_ROOT}/ckpt/${DATASET}/t5_seq2seq-D0-3e-4lr-0.001wd-ablation-t5-small-w-pkm
TRAIN_D=1

LR=3e-4
WD=0.0
EPOCHS=50
MODEL_MAX_LENGTH=512

TEST_BATCH_SIZE=256
NUM_BEAMS=20
MAX_NEW_TOKENS=10

PREFIX=v-only-fromD0-trainD${TRAIN_D}
OUT_DIR=${CODE_ROOT}/ckpt/${DATASET}/${STRATEGY}-D${TRAIN_D}-${LR}lr-${WD}wd-${PREFIX}

mkdir -p "${OUT_DIR}"
mkdir -p ./log/${DATASET}/train
mkdir -p ./log/${DATASET}/test

# -------------------------
# PKM config
# -------------------------
T5_PK_IS_ENABLED=1
T5_PK_ENCODER_LAYERS=""
T5_PK_DECODER_LAYERS="2"

PK_MEM_N_KEYS=128
PK_MEM_HEADS=4
PK_MEM_K_DIM=512
PK_MEM_V_DIM=-1
PK_TOPK=8
T5_PK_MEM_KNN=32

# 关键：建议 false，保证每层有自己独立的 V（否则会 shared values）
T5_PK_MEM_SHARE_VALUES=0

T5_PK_VALUE_FIXED_LR=0.001
PK_MEM_GATED=0

# -------------------------
# Train (value-only)
# -------------------------
torchrun --nproc_per_node=1 --master_port=2319 train_t5_pkm_v_only.py \
  config="${CONFIG_FILE}" \
  "strategy=${STRATEGY}" \
  "model.${STRATEGY}.base_model=${BASE_MODEL}" \
  "train.output_dir=${OUT_DIR}" \
  "train.init_from_ckpt=${INIT_FROM_CKPT}" \
  "dataset.data_path=${AMAZON_ROOT}" \
  "dataset.name=${DATASET}" \
  "dataset.train_file=${DATA_ROOT}/D${TRAIN_D}/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.valid_file=${DATA_ROOT}/D${TRAIN_D}/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.test_file=${DATA_ROOT}/D${TRAIN_D}/${DATASET}_5_${TIME_RANGE}.csv" \
  "dataset.index_file=${INDEX_FILE}" \
  "train.batch_size=256" \
  "train.learning_rate=${LR}" \
  "train.weight_decay=${WD}" \
  "train.epochs=${EPOCHS}" \
  "train.save_and_eval_strategy=epoch" \
  "train.model_max_length=${MODEL_MAX_LENGTH}" \
  "pkm.t5_seq2seq.pk_is_enabled=true" \
  "pkm.t5_seq2seq.pk_encoder_layers=${T5_PK_ENCODER_LAYERS}" \
  "pkm.t5_seq2seq.pk_decoder_layers=${T5_PK_DECODER_LAYERS}" \
  "pkm.t5_seq2seq.pk_mem_n_keys=${PK_MEM_N_KEYS}" \
  "pkm.t5_seq2seq.pk_mem_heads=${PK_MEM_HEADS}" \
  "pkm.t5_seq2seq.pk_mem_knn=${T5_PK_MEM_KNN}" \
  "pkm.t5_seq2seq.pk_mem_share_values=false" \
  "pkm.t5_seq2seq.pk_mem_k_dim=${PK_MEM_K_DIM}" \
  "pkm.t5_seq2seq.pk_mem_v_dim=${PK_MEM_V_DIM}" \
  "pkm.t5_seq2seq.pk_topk=${PK_TOPK}" \
  "pkm.t5_seq2seq.pk_value_fixed_lr=${T5_PK_VALUE_FIXED_LR}" \
  > "./log/${DATASET}/train/${PREFIX}.log"

# -------------------------
# Forward tests: test D2-D4 (grouped)
# -------------------------
for test_d in 2 3 4; do
  if [[ "${test_d}" -le "${TRAIN_D}" ]]; then
    continue
  fi

  echo "===== Testing ${OUT_DIR} on D${test_d} (grouped users, 1 GPU) ====="

  GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
  GROUP_FILES=(${GROUP_DIR}/*.csv)

  for group_file in "${GROUP_FILES[@]}"; do
    group_name=$(basename "${group_file}" .csv)
    logfile=./log/${DATASET}/test/${PREFIX}-TIGER-trainD${TRAIN_D}-testD${test_d}-${group_name}.log

    echo "  → D${test_d}, ${group_name}"

    python test.py \
      config="${CONFIG_FILE}" \
      "model.type=${TEST_MODEL_TYPE}" \
      "global.gpu_id=0" \
      "model.ckpt_path=${OUT_DIR}" \
      "model.tokenizer_path=${OUT_DIR}" \
      "model.base_model=${BASE_MODEL}" \
      "dataset.name=${DATASET}" \
      "dataset.data_path=${AMAZON_ROOT}" \
      "dataset.train_file=${DATA_ROOT}/D${TRAIN_D}/${DATASET}_5_${TIME_RANGE}.csv" \
      "dataset.valid_file=${DATA_ROOT}/D${TRAIN_D}/${DATASET}_5_${TIME_RANGE}.csv" \
      "dataset.test_file=${group_file}" \
      "dataset.index_file=${INDEX_FILE}" \
      "test.batch_size=${TEST_BATCH_SIZE}" \
      "test.num_beams=${NUM_BEAMS}" \
      "test.max_new_tokens=${MAX_NEW_TOKENS}" \
      "test.filter_items=true" \
      "pkm.t5_seq2seq.pk_is_enabled=true" \
      "pkm.t5_seq2seq.pk_encoder_layers=${T5_PK_ENCODER_LAYERS}" \
      "pkm.t5_seq2seq.pk_decoder_layers=${T5_PK_DECODER_LAYERS}" \
      "pkm.t5_seq2seq.pk_mem_n_keys=${PK_MEM_N_KEYS}" \
      "pkm.t5_seq2seq.pk_mem_heads=${PK_MEM_HEADS}" \
      "pkm.t5_seq2seq.pk_mem_knn=${T5_PK_MEM_KNN}" \
      "pkm.t5_seq2seq.pk_mem_share_values=false" \
      "pkm.t5_seq2seq.pk_mem_k_dim=${PK_MEM_K_DIM}" \
      "pkm.t5_seq2seq.pk_mem_v_dim=${PK_MEM_V_DIM}" \
      "pkm.t5_seq2seq.pk_topk=${PK_TOPK}" \
      "pkm.t5_seq2seq.pk_value_fixed_lr=${T5_PK_VALUE_FIXED_LR}" \
      > "${logfile}"
  done

  echo "===== Finished D${test_d} grouped testing ====="
done