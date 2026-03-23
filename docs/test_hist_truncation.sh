#!/bin/bash
# Single test run for history truncation experiment.
# Called by run_hist_truncation_sweep.py with env vars set.
set -euo pipefail

export TRANSFORMERS_NO_TF=1
export USE_TF=0
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

# ── Required env vars (set by run_hist_truncation_sweep.py) ──────────────────
CKPT_PATH=${CKPT_PATH:?need CKPT_PATH}
GROUP_FILE=${GROUP_FILE:?need GROUP_FILE}
TEST_MAX_HIS_LEN=${TEST_MAX_HIS_LEN:?need TEST_MAX_HIS_LEN}
RESULT_JSONL=${RESULT_JSONL:-./log/hist_truncation/result.jsonl}

# ── Optional env vars with defaults ──────────────────────────────────────────
DATASET=${DATASET:-Toys_and_Games}
DATA_ROOT=${DATA_ROOT:-/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data}
AMAZON_ROOT=${AMAZON_ROOT:-/mlx_devbox/users/zhuosong.jiang/playground/memory_dev/data}
BASE_MODEL=${BASE_MODEL:-google-t5/t5-small}
TIME_RANGE=${TIME_RANGE:-2016-10-2018-11}
INDEX_FILE=${INDEX_FILE:-.TIGER-index.json}
CONFIG_FILE=${CONFIG_FILE:-${CODE_ROOT}/configs/train_t5_pkm_warmup.yaml}

# PKM params (forwarded from runner; must match the checkpoint)
T5_PK_ENCODER_LAYERS=${T5_PK_ENCODER_LAYERS:-""}
T5_PK_DECODER_LAYERS=${T5_PK_DECODER_LAYERS:-"2"}
PK_MEM_N_KEYS=${PK_MEM_N_KEYS:-128}
PK_TOPK=${PK_TOPK:-8}
PK_MEM_HEADS=${PK_MEM_HEADS:-4}
PK_MEM_K_DIM=${PK_MEM_K_DIM:-512}
PK_MEM_V_DIM=${PK_MEM_V_DIM:--1}
PK_MEM_GATED=${PK_MEM_GATED:-0}
T5_PK_MEM_SHARE_VALUES=${T5_PK_MEM_SHARE_VALUES:-0}
T5_PK_VALUE_FIXED_LR=${T5_PK_VALUE_FIXED_LR:-0.001}
T5_PK_IS_ENABLED=${T5_PK_IS_ENABLED:-true}

TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-8}
NUM_BEAMS=${NUM_BEAMS:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}

if [[ "${PK_MEM_GATED}" == "1" ]]; then PK_MEM_GATED_BOOL=true; else PK_MEM_GATED_BOOL=false; fi
if [[ "${T5_PK_MEM_SHARE_VALUES}" == "1" ]]; then PK_SV_BOOL=true; else PK_SV_BOOL=false; fi

python test.py \
  config="${CONFIG_FILE}" \
  "model.type=t5_seq2seq" \
  "global.gpu_id=0" \
  "model.ckpt_path=${CKPT_PATH}" \
  "model.tokenizer_path=${CKPT_PATH}" \
  "model.base_model=${BASE_MODEL}" \
  "dataset.name=${DATASET}" \
  "dataset.data_path=${AMAZON_ROOT}" \
  "dataset.test_file=${GROUP_FILE}" \
  "dataset.index_file=${INDEX_FILE}" \
  "dataset.test_max_his_len=${TEST_MAX_HIS_LEN}" \
  "test.batch_size=${TEST_BATCH_SIZE}" \
  "test.num_beams=${NUM_BEAMS}" \
  "test.max_new_tokens=${MAX_NEW_TOKENS}" \
  "test.filter_items=true" \
  "train.model_max_length=${MODEL_MAX_LENGTH}" \
  "pkm.t5_seq2seq.pk_is_enabled=${T5_PK_IS_ENABLED}" \
  "pkm.t5_seq2seq.pk_encoder_layers=${T5_PK_ENCODER_LAYERS}" \
  "pkm.t5_seq2seq.pk_decoder_layers=${T5_PK_DECODER_LAYERS}" \
  "pkm.t5_seq2seq.pk_mem_n_keys=${PK_MEM_N_KEYS}" \
  "pkm.t5_seq2seq.pk_mem_heads=${PK_MEM_HEADS}" \
  "pkm.t5_seq2seq.pk_mem_k_dim=${PK_MEM_K_DIM}" \
  "pkm.t5_seq2seq.pk_mem_v_dim=${PK_MEM_V_DIM}" \
  "pkm.t5_seq2seq.pk_topk=${PK_TOPK}" \
  "pkm.t5_seq2seq.pk_value_fixed_lr=${T5_PK_VALUE_FIXED_LR}" \
  "pkm.t5_seq2seq.pk_mem_gated=${PK_MEM_GATED_BOOL}" \
  "pkm.t5_seq2seq.pk_mem_share_values=${PK_SV_BOOL}"
