#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# User configs (edit here)
# -----------------------
SCRIPT=${SCRIPT:-rectified_flow_train.py}

DATASET=${DATASET:-mvtec}
# CLASS_NAME=${CLASS_NAME:-cable}

# MSFlow checkpoint auto-resolve settings (matches your folder structure)
MSFLOW_WORK_DIR=${MSFLOW_WORK_DIR:-work_dirs}
MSFLOW_VERSION=${MSFLOW_VERSION:-msflow_wide_resnet50_2_avgpool_pl258}
MSFLOW_CKPT_NAME=${MSFLOW_CKPT_NAME:-best_loc_auroc.pt}

# Common args
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-0}
SEED=${SEED:-0}
POOL_TYPE=${POOL_TYPE:-avg}

# Where to write RF outputs (ckpts) and logs
RF_WORK_DIR=${RF_WORK_DIR:-./work_dirs}
LOG_DIR=${LOG_DIR:-./logs_rf_sweep}
mkdir -p "${LOG_DIR}"

# -----------------------
# Sweep grid (edit here)
# -----------------------
RF_EPOCHS_LIST=(50)
RF_LR_LIST=(5e-4)

# Stage-wise RF configs (RF is applied only to stages [0,1] -> 2 stages total)
# Examples:
#   --rf-tdims 64 32 16
#   --rf-depths 4 2 1
RF_TDIMS_LIST=(
  "32 32"
  "64 32"
  "64 64"
  "128 64"
  "128 128"
  "256 128"
  "256 256"
)
RF_DEPTHS_LIST=(
  "1 1"
  "2 1"
  "2 2"
  "3 2"
  "3 3"
  "4 3"
  "4 4"
)

RF_STEPS_LIST=(1)

# CLASS_LIST=(bottle cable capsule carpet grid
#                hazelnut leather metal_nut pill screw
#                tile toothbrush wood zipper)

CLASS_LIST=(wood)

# Optional: choose GPU
# export CUDA_VISIBLE_DEVICES=0

# -----------------------
# Run sweep
# -----------------------
RUN_TS=$(date +"%Y%m%d_%H%M%S")

for RF_EPOCHS in "${RF_EPOCHS_LIST[@]}"; do
  for RF_LR in "${RF_LR_LIST[@]}"; do
    for RF_TDIMS in "${RF_TDIMS_LIST[@]}"; do
      for RF_DEPTHS in "${RF_DEPTHS_LIST[@]}"; do
        for RF_STEPS in "${RF_STEPS_LIST[@]}"; do
          for CLASS_NAME in "${CLASS_LIST[@]}"; do
              
              RUN_ID="cls=${CLASS_NAME}_e=${RF_EPOCHS}_lr=${RF_LR}_tdims=$(echo ${RF_TDIMS} | tr ' ' '-')_depths=$(echo ${RF_DEPTHS} | tr ' ' '-')_steps=${RF_STEPS}_${RUN_TS}"
              LOG_FILE="${LOG_DIR}/${RUN_ID}.txt"

              # 1) Print hyperparams header FIRST (your requirement)
              {
                echo "===== RUN ${RUN_ID} ====="
                echo "rf-epochs=${RF_EPOCHS} rf-lr=${RF_LR} rf-tdims=${RF_TDIMS} rf-depths=${RF_DEPTHS} rf-steps=${RF_STEPS} batch-size=${BATCH_SIZE} workers=${WORKERS} seed=${SEED}"
                echo "dataset=${DATASET} class-name=${CLASS_NAME} pool-type=${POOL_TYPE}"
                echo "msflow-work-dir=${MSFLOW_WORK_DIR} msflow-version=${MSFLOW_VERSION} msflow-ckpt-name=${MSFLOW_CKPT_NAME}"
                echo "rf-work-dir=${RF_WORK_DIR}"
                echo "timestamp=$(date -Iseconds)"
                echo "-------------------------"
              } | tee "${LOG_FILE}"

              # 2) Run and append logs
              python "${SCRIPT}"                 --dataset "${DATASET}"                 --class-name "${CLASS_NAME}"                 --mode train                 --batch-size "${BATCH_SIZE}"                 --workers "${WORKERS}"                 --seed "${SEED}"                 --pool-type "${POOL_TYPE}"                 --msflow-work-dir "${MSFLOW_WORK_DIR}"                 --msflow-version "${MSFLOW_VERSION}"                 --msflow-ckpt-name "${MSFLOW_CKPT_NAME}"                 --work-dir "${RF_WORK_DIR}"                 --rf-epochs "${RF_EPOCHS}"                 --rf-lr "${RF_LR}"                 --rf-tdims ${RF_TDIMS}                 --rf-depths ${RF_DEPTHS}                 --rf-steps "${RF_STEPS}"                 2>&1 | tee -a "${LOG_FILE}"

              echo "" | tee -a "${LOG_FILE}"
              echo "===== DONE ${RUN_ID} =====" | tee -a "${LOG_FILE}"
              echo "" | tee -a "${LOG_FILE}"
            done
          done
        done
      done
    done
done