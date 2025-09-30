#!/bin/bash

export DATA_DIR
export CKPT_DIR
export NSIGHT_LOG_DIR
export NSIGHT_FILE_NAME
export MODE
export LOCAL_GPU_IDS
export NUM_GPUS

###########################################################################
# Problem 0: Generate Nsight log
# Find the correct way to generate Nsight log.
###########################################################################

# Ensure checkpoint and profiling directories exist before launching Nsight.
mkdir -p "$CKPT_DIR"
mkdir -p "$NSIGHT_LOG_DIR"

# Build python argument list dynamically to allow optional flags like worker count.
PYTHON_ARGS=(
    --num_gpu="$NUM_GPUS"
    --data="$DATA_DIR"
    --ckpt="$CKPT_DIR"
    --mode="$MODE"
    --save_ckpt
)

if [ -n "${NUM_WORKERS:-}" ]; then
    PYTHON_ARGS+=(--num_workers "$NUM_WORKERS")
fi

# Run training under Nsight Systems to collect GPU traces.
CUDA_VISIBLE_DEVICES="$LOCAL_GPU_IDS" \
nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --output "${NSIGHT_LOG_DIR}/${NSIGHT_FILE_NAME}" \
    python train_cifar.py "${PYTHON_ARGS[@]}"

# Rename the default Nsight report to the expected extension when available.
QDREP_PATH="${NSIGHT_LOG_DIR}/${NSIGHT_FILE_NAME}.qdrep"
if [ -f "$QDREP_PATH" ]; then
    mv "$QDREP_PATH" "${NSIGHT_LOG_DIR}/${NSIGHT_FILE_NAME}.nsys-rep"
fi