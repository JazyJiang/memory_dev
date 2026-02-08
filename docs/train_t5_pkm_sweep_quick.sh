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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-2400}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

DATASET=${DATASET:-Toys_and_Games}

DATA_ROOT=${DATA_ROOT:-/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data}
AMAZON_ROOT=${AMAZON_ROOT:-/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data}

BASE_MODEL=${BASE_MODEL:-google-t5/t5-small}
STRATEGY=t5_seq2seq
TEST_MODEL_TYPE=t5_seq2seq

TIME_RANGE=${TIME_RANGE:-2016-10-2018-11}
INDEX_FILE=${INDEX_FILE:-.TIGER-index.json}

CONFIG_FILE=${CONFIG_FILE:-${CODE_ROOT}/configs/train_decoder_only_pkm.yaml}

EPOCHS=${EPOCHS:-50}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10}
NUM_BEAMS=${NUM_BEAMS:-20}

WD=${WD:-0.001}                  # fixed (not swept for now)
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-256}

# -------------------------
# Core sweep variables (set via env, or edit defaults here)
# -------------------------
LR=${LR:-3e-4}
BATCH_SIZE=${BATCH_SIZE:-256}

T5_PK_ENCODER_LAYERS=${T5_PK_ENCODER_LAYERS:-""}   # e.g. "0,2" or "" (empty ok)
T5_PK_DECODER_LAYERS=${T5_PK_DECODER_LAYERS:-"2"}  # e.g. "1,3"

PK_MEM_N_KEYS=${PK_MEM_N_KEYS:-128}
PK_TOPK=${PK_TOPK:-8}

PK_MEM_HEADS=${PK_MEM_HEADS:-4}   # fixed (not swept for now)
PK_MEM_K_DIM=${PK_MEM_K_DIM:-512}
PK_MEM_V_DIM=${PK_MEM_V_DIM:--1}

PK_MEM_GATED=${PK_MEM_GATED:-0}                 # 0/1
T5_PK_MEM_SHARE_VALUES=${T5_PK_MEM_SHARE_VALUES:-0}  # 0/1

T5_PK_VALUE_FIXED_LR=${T5_PK_VALUE_FIXED_LR:-0.001}  # can be "tied" or float
T5_PK_VALUE_WEIGHT_DECAY=${T5_PK_VALUE_WEIGHT_DECAY:-0.0}

CLEANUP_CKPT=${CLEANUP_CKPT:-1}   # 1 => delete ckpt after this hparam run finishes
RESULT_JSONL=${RESULT_JSONL:-./log/${DATASET}/sweep_t5_pkm/result.jsonl}

# -------------------------
# Derived / normalized values
# -------------------------
if [[ "${T5_PK_VALUE_FIXED_LR}" == "tied" ]]; then
  EFFECTIVE_PK_VALUE_LR="${LR}"
else
  EFFECTIVE_PK_VALUE_LR="${T5_PK_VALUE_FIXED_LR}"
fi

if [[ "${PK_TOPK}" -gt "${PK_MEM_N_KEYS}" ]]; then
  echo "ERROR: invalid combo: PK_TOPK(${PK_TOPK}) > PK_MEM_N_KEYS(${PK_MEM_N_KEYS})"
  exit 2
fi

if [[ "${PK_MEM_GATED}" == "1" ]]; then
  PK_MEM_GATED_BOOL=true
else
  PK_MEM_GATED_BOOL=false
fi

if [[ "${T5_PK_MEM_SHARE_VALUES}" == "1" ]]; then
  PK_MEM_SHARE_VALUES_BOOL=true
else
  PK_MEM_SHARE_VALUES_BOOL=false
fi

# Make a stable run tag (safe for paths)
sanitize_layers() {
  local s="$1"
  if [[ -z "${s}" ]]; then
    echo "none"
  else
    echo "${s//,/--}"
  fi
}

ENC_TAG=$(sanitize_layers "${T5_PK_ENCODER_LAYERS}")
DEC_TAG=$(sanitize_layers "${T5_PK_DECODER_LAYERS}")

RUN_TAG="t5pkm_lr${LR}_bs${BATCH_SIZE}_enc${ENC_TAG}_dec${DEC_TAG}_nk${PK_MEM_N_KEYS}_topk${PK_TOPK}_k${PK_MEM_K_DIM}_v${PK_MEM_V_DIM}_pklr${EFFECTIVE_PK_VALUE_LR}_pkwd${T5_PK_VALUE_WEIGHT_DECAY}_g${PK_MEM_GATED}_sv${T5_PK_MEM_SHARE_VALUES}"
LOCAL_CKPT_ROOT=${LOCAL_CKPT_ROOT:-/tmp/${USER}/memory_dev_ckpt}
RUN_CKPT_ROOT="${LOCAL_CKPT_ROOT}/ckpt/${DATASET}/sweep_t5_pkm/${RUN_TAG}"
RUN_LOG_ROOT="${CODE_ROOT}/log/${DATASET}/sweep_t5_pkm/${RUN_TAG}"
TRAIN_LOG_DIR="${RUN_LOG_ROOT}/train"
TEST_LOG_DIR="${RUN_LOG_ROOT}/test"

mkdir -p "${TRAIN_LOG_DIR}" "${TEST_LOG_DIR}"
mkdir -p "$(dirname "${RESULT_JSONL}")"

PARAMS_JSON="${RUN_LOG_ROOT}/params.json"
cat > "${PARAMS_JSON}" << EOF
{
  "dataset": "${DATASET}",
  "strategy": "${STRATEGY}",
  "base_model": "${BASE_MODEL}",
  "train.learning_rate": ${LR},
  "train.batch_size": ${BATCH_SIZE},
  "train.weight_decay": ${WD},
  "train.epochs": ${EPOCHS},
  "train.model_max_length": ${MODEL_MAX_LENGTH},

  "pkm.t5_seq2seq.pk_is_enabled": true,
  "pkm.t5_seq2seq.pk_encoder_layers": "${T5_PK_ENCODER_LAYERS}",
  "pkm.t5_seq2seq.pk_decoder_layers": "${T5_PK_DECODER_LAYERS}",
  "pkm.t5_seq2seq.pk_mem_n_keys": ${PK_MEM_N_KEYS},
  "pkm.t5_seq2seq.pk_mem_heads": ${PK_MEM_HEADS},
  "pkm.t5_seq2seq.pk_topk": ${PK_TOPK},
  "pkm.t5_seq2seq.pk_mem_k_dim": ${PK_MEM_K_DIM},
  "pkm.t5_seq2seq.pk_mem_v_dim": ${PK_MEM_V_DIM},
  "pkm.t5_seq2seq.pk_value_fixed_lr_raw": "${T5_PK_VALUE_FIXED_LR}",
  "pkm.t5_seq2seq.pk_value_fixed_lr_effective": ${EFFECTIVE_PK_VALUE_LR},
  "pkm.t5_seq2seq.pk_value_weight_decay": ${T5_PK_VALUE_WEIGHT_DECAY},
  "pkm.t5_seq2seq.pk_mem_gated": ${PK_MEM_GATED_BOOL},
  "pkm.t5_seq2seq.pk_mem_share_values": ${PK_MEM_SHARE_VALUES_BOOL},

  "run_tag": "${RUN_TAG}"
}
EOF

T5_OVERRIDES=(
  "pkm.t5_seq2seq.pk_is_enabled=true"
  "pkm.t5_seq2seq.pk_encoder_layers=${T5_PK_ENCODER_LAYERS}"
  "pkm.t5_seq2seq.pk_decoder_layers=${T5_PK_DECODER_LAYERS}"
  "pkm.t5_seq2seq.pk_mem_n_keys=${PK_MEM_N_KEYS}"
  "pkm.t5_seq2seq.pk_mem_heads=${PK_MEM_HEADS}"
  "pkm.t5_seq2seq.pk_mem_k_dim=${PK_MEM_K_DIM}"
  "pkm.t5_seq2seq.pk_mem_v_dim=${PK_MEM_V_DIM}"
  "pkm.t5_seq2seq.pk_topk=${PK_TOPK}"
  "pkm.t5_seq2seq.pk_value_fixed_lr=${EFFECTIVE_PK_VALUE_LR}"
  "pkm.t5_seq2seq.pk_value_weight_decay=${T5_PK_VALUE_WEIGHT_DECAY}"
  "pkm.t5_seq2seq.pk_mem_gated=${PK_MEM_GATED_BOOL}"
  "pkm.t5_seq2seq.pk_mem_share_values=${PK_MEM_SHARE_VALUES_BOOL}"
)

cleanup_ckpt() {
  if [[ "${CLEANUP_CKPT}" == "1" ]]; then
    rm -rf "${RUN_CKPT_ROOT}"
  fi
}

# Ensure we cleanup on normal exit; keep logs and results regardless
trap cleanup_ckpt EXIT

################################
# Train D0..D3, and test only next D
################################
for train_d in 0 1 2 3; do
  test_d=$((train_d + 1))

  CUR_CKPT="${RUN_CKPT_ROOT}/D${train_d}"
  mkdir -p "${CUR_CKPT}"

  TRAIN_LOG="${TRAIN_LOG_DIR}/${RUN_TAG}_trainD${train_d}.log"

  torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT_BASE + train_d)) train.py \
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
    "${T5_OVERRIDES[@]}" \
    > "${TRAIN_LOG}"

  GROUP_DIR="${DATA_ROOT}/D${test_d}/groups"
  GROUP_FILES=("${GROUP_DIR}"/*.csv)

  for group_file in "${GROUP_FILES[@]}"; do
    group_name=$(basename "${group_file}" .csv)
    TEST_LOG="${TEST_LOG_DIR}/${RUN_TAG}_trainD${train_d}_testD${test_d}_${group_name}.log"

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
      "${T5_OVERRIDES[@]}" \
      > "${TEST_LOG}"
  done
done

# Write exactly ONE jsonl line per hparam run_tag
python "${CODE_ROOT}/docs/write_result_jsonl.py" \
  --params_json "${PARAMS_JSON}" \
  --run_tag "${RUN_TAG}" \
  --test_log_glob "${TEST_LOG_DIR}/${RUN_TAG}_trainD*_testD*_*.log" \
  >> "${RESULT_JSONL}"