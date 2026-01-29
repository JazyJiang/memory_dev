#!/bin/bash
set -e

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

CODE_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/2026-01-test-to-delete
cd "${CODE_ROOT}"

DATASET=Toys_and_Games

# 这里改成你本地 0_process.py 生成 D0~D4 的目录（包含 D0/D1/D2/D3/D4 的那个目录）
DATA_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/2026-01-test-to-delete/data

# 这里改成包含 Toys_and_Games.TIGER-index.json 的目录
AMAZON_ROOT=/mlx_devbox/users/zhuosong.jiang/playground/2026-01-test-to-delete/data

LR=3e-4
WD=0.001
SUFFIX=debug-1gpu
BATCH_SIZE=256
TEST_BATCH_SIZE=100
NUM_BEAMS=20

TIME_RANGE=2016-10-2018-11

mkdir -p ./log/${DATASET}/train
mkdir -p ./log/${DATASET}/test

################################
# Stage 0: Train on D0 (1 GPU)
################################
BASE_MODEL=google-t5/t5-small
PREV_CKPT=${CODE_ROOT}/ckpt/${DATASET}/TIGER-D0-${LR}lr-${WD}wd-${SUFFIX}
mkdir -p "${PREV_CKPT}"

torchrun --nproc_per_node=1 --master_port=2309 finetune.py \
  --base_model ${BASE_MODEL} \
  --output_dir ${PREV_CKPT} \
  --data_path ${AMAZON_ROOT} \
  --dataset ${DATASET} \
  --train_file ${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv \
  --valid_file ${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv \
  --test_file  ${DATA_ROOT}/D0/${DATASET}_5_${TIME_RANGE}.csv \
  --per_device_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --epochs 50 \
  --weight_decay ${WD} \
  --save_and_eval_strategy epoch \
  --index_file .TIGER-index.json \
  > ./log/${DATASET}/train/TIGER-D0-train-1gpu.log

################################
# Forward tests (per-D grouped, single GPU sequential)
################################
train_d=0
CUR_CKPT=${PREV_CKPT}

for test_d in $(seq $((train_d + 1)) 4); do
  echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users, 1 GPU) ====="

  GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
  GROUP_FILES=(${GROUP_DIR}/*.csv)

  for group_file in "${GROUP_FILES[@]}"; do
    group_name=$(basename "${group_file}" .csv)
    logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}-1gpu.log

    echo "  → D${test_d}, ${group_name}"

    python test.py \
      --gpu_id 0 \
      --ckpt_path ${CUR_CKPT} \
      --dataset ${DATASET} \
      --data_path ${AMAZON_ROOT} \
      --train_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
      --valid_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
      --test_file ${group_file} \
      --test_batch_size ${TEST_BATCH_SIZE} \
      --num_beams ${NUM_BEAMS} \
      --index_file .TIGER-index.json \
      --filter_items \
      > "${logfile}"
  done

  echo "===== Finished D${test_d} grouped testing ====="
done

################################
# Stage 1 & 2: Continual finetune (1 GPU)
################################
for train_d in 1 2 3; do
  CUR_CKPT=${CODE_ROOT}/ckpt/${DATASET}/TIGER-D${train_d}-${LR}lr-${WD}wd-${SUFFIX}
  mkdir -p "${CUR_CKPT}"

  torchrun --nproc_per_node=1 --master_port=$((2310 + train_d)) finetune.py \
    --base_model ${PREV_CKPT} \
    --output_dir ${CUR_CKPT} \
    --data_path ${AMAZON_ROOT} \
    --dataset ${DATASET} \
    --train_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
    --valid_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
    --test_file  ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
    --per_device_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --epochs 50 \
    --weight_decay ${WD} \
    --save_and_eval_strategy epoch \
    --index_file .TIGER-index.json \
    > ./log/${DATASET}/train/TIGER-D${train_d}-finetune-1gpu.log

  for test_d in $(seq $((train_d + 1)) 4); do
    echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users, 1 GPU) ====="

    GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
    GROUP_FILES=(${GROUP_DIR}/*.csv)

    for group_file in "${GROUP_FILES[@]}"; do
      group_name=$(basename "${group_file}" .csv)
      logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}-1gpu.log

      echo "  → D${test_d}, ${group_name}"

      python test.py \
        --gpu_id 0 \
        --ckpt_path ${CUR_CKPT} \
        --dataset ${DATASET} \
        --data_path ${AMAZON_ROOT} \
        --train_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
        --valid_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_${TIME_RANGE}.csv \
        --test_file ${group_file} \
        --test_batch_size ${TEST_BATCH_SIZE} \
        --num_beams ${NUM_BEAMS} \
        --index_file .TIGER-index.json \
        --filter_items \
        > "${logfile}"
    done

    echo "===== Finished D${test_d} grouped testing ====="
  done

  PREV_CKPT=${CUR_CKPT}
done
 
PLOT_DIR=./log/${DATASET}/plots
mkdir -p "${PLOT_DIR}"
python ${CODE_ROOT}/plot_test_results.py \
  --log_dir ./log/${DATASET}/test \
  --out_dir "${PLOT_DIR}" \
  --dataset "${DATASET}"