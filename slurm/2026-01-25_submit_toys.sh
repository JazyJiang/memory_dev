#!/bin/bash
#SBATCH --job-name=reasoning
#SBATCH --cpus-per-task=80
#SBATCH --output=/home/xinyulin/reasoning/slurm/log/%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00

unset SLURM_CPU_BIND

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate verification

#!/bin/bash

export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /home/xinyulin/context/code/toys/TIGER 

DATASET=Toys_and_Games
DATA_ROOT=/home/xinyulin/context/data/2026-01-25_5period
AMAZON_ROOT=/home/xinyulin/context/data/amazon

LR=3e-4
WD=0.001
SUFFIX=debug
BATCH_SIZE=256
TEST_BATCH_SIZE=100
NUM_BEAMS=20

mkdir -p ./log/${DATASET}/train
mkdir -p ./log/${DATASET}/test

################################
# Stage 0: Train on D0
################################
BASE_MODEL=google-t5/t5-small
PREV_CKPT=/home/xinyulin/context/code/${DATASET}/TIGER/ckpt/TIGER-D0-${LR}lr-${WD}wd-${SUFFIX}

torchrun --nproc_per_node=8 --master_port=2309 finetune.py \
  --base_model ${BASE_MODEL} \
  --output_dir ${PREV_CKPT} \
  --data_path ${AMAZON_ROOT} \
  --dataset ${DATASET} \
  --train_file ${DATA_ROOT}/D0/${DATASET}_5_2016-10-2018-11.csv \
  --valid_file ${DATA_ROOT}/D0/${DATASET}_5_2016-10-2018-11.csv \
  --test_file  ${DATA_ROOT}/D0/${DATASET}_5_2016-10-2018-11.csv \
  --per_device_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --epochs 50 \
  --weight_decay ${WD} \
  --save_and_eval_strategy epoch \
  --index_file .TIGER-index.json \
  > ./log/${DATASET}/train/TIGER-D0-train.log

################################
# Forward tests (per-D grouped, parallel)
################################
train_d=0
GPU_IDS=(0 1 2 3 4 5 6 7)
CUR_CKPT=${PREV_CKPT}
mkdir -p ./log/${DATASET}/test

for test_d in $(seq $((train_d + 1)) 4); do
  echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users) ====="

  GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
  GROUP_FILES=(${GROUP_DIR}/*.csv)

  pids=()
  gpu_idx=0

  for group_file in "${GROUP_FILES[@]}"; do
    gpu_id=${GPU_IDS[$gpu_idx]}
    gpu_idx=$(( (gpu_idx + 1) % ${#GPU_IDS[@]} ))

    group_name=$(basename "${group_file}" .csv)
    logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}.log

    echo "  → D${test_d}, ${group_name}, GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python test.py \
      --gpu_id 0 \
      --ckpt_path ${CUR_CKPT} \
      --dataset ${DATASET} \
      --data_path ${AMAZON_ROOT} \
      --test_file ${group_file} \
      --test_batch_size ${TEST_BATCH_SIZE} \
      --num_beams ${NUM_BEAMS} \
      --index_file .TIGER-index.json \
      --filter_items \
      > "${logfile}" &

    pids+=($!)
  done

  # wait for all group tests of this D to finish
  for pid in "${pids[@]}"; do
    wait $pid
  done

  echo "===== Finished D${test_d} grouped testing ====="
done



################################
# Stage 1 & 2: Continual finetune
################################
for train_d in 1 2 3; do
  CUR_CKPT=/home/xinyulin/context/code/${DATASET}/TIGER/ckpt/TIGER-D${train_d}-${LR}lr-${WD}wd-${SUFFIX}

  torchrun --nproc_per_node=8 --master_port=$((2310 + train_d)) finetune.py \
    --base_model ${PREV_CKPT} \
    --output_dir ${CUR_CKPT} \
    --data_path ${AMAZON_ROOT} \
    --dataset ${DATASET} \
    --train_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_2016-10-2018-11.csv \
    --valid_file ${DATA_ROOT}/D${train_d}/${DATASET}_5_2016-10-2018-11.csv \
    --test_file  ${DATA_ROOT}/D${train_d}/${DATASET}_5_2016-10-2018-11.csv \
    --per_device_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --epochs 50 \
    --weight_decay ${WD} \
    --save_and_eval_strategy epoch \
    --index_file .TIGER-index.json \
    > ./log/${DATASET}/train/TIGER-D${train_d}-finetune.log

    ################################
    # Forward tests (per-D grouped, parallel)
    ################################
    GPU_IDS=(0 1 2 3 4 5 6 7)

    for test_d in $(seq $((train_d + 1)) 4); do
    echo "===== Testing ${CUR_CKPT} on D${test_d} (grouped users) ====="

    GROUP_DIR=${DATA_ROOT}/D${test_d}/groups
    GROUP_FILES=(${GROUP_DIR}/*.csv)

    pids=()
    gpu_idx=0

    for group_file in "${GROUP_FILES[@]}"; do
        gpu_id=${GPU_IDS[$gpu_idx]}
        gpu_idx=$(( (gpu_idx + 1) % ${#GPU_IDS[@]} ))

        group_name=$(basename "${group_file}" .csv)
        logfile=./log/${DATASET}/test/TIGER-D${train_d}-test-D${test_d}-${group_name}.log

        echo "  → D${test_d}, ${group_name}, GPU ${gpu_id}"

        CUDA_VISIBLE_DEVICES=${gpu_id} python test.py \
        --gpu_id 0 \
        --ckpt_path ${CUR_CKPT} \
        --dataset ${DATASET} \
        --data_path ${AMAZON_ROOT} \
        --test_file ${group_file} \
        --test_batch_size ${TEST_BATCH_SIZE} \
        --num_beams ${NUM_BEAMS} \
        --index_file .TIGER-index.json \
        --filter_items \
        > "${logfile}" &

        pids+=($!)
    done

    # wait for all group tests of this D to finish
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "===== Finished D${test_d} grouped testing ====="
    done


  PREV_CKPT=${CUR_CKPT}
done