#!/bin/bash
# CliffordNet Training Launch Script for ImageNet-1k
# Configured for 6x 4090D GPUs with FSDP

# Default configuration
DATA_DIR="${DATA_DIR:-./imagenet1k}"
MODEL_SIZE="${MODEL_SIZE:-small}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-90}"
LR="${LR:-1e-3}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "CliffordNet ImageNet-1k Training"
echo "=============================================="
echo "Data directory: ${DATA_DIR}"
echo "Model size: ${MODEL_SIZE}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Total batch size: $((BATCH_SIZE * 6))"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Precision: ${PRECISION}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training with uv
uv run python train_imagenet1k.py \
    --data-dir "${DATA_DIR}" \
    --model-size "${MODEL_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --precision "${PRECISION}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers "${NUM_WORKERS}"
