#!/bin/bash
# =============================================================================
# GPU Stress Test Script for CliffordNet
# Tests each GPU individually (single-GPU) and all GPUs together (multi-GPU DDP)
#
# Usage:
#   ./gpu_stress_test.sh                  # Test GPU 0-7, default settings
#   ./gpu_stress_test.sh --max-gpu 3      # Test GPU 0-3 only
#   ./gpu_stress_test.sh --model-size base # Use a larger model
#   ./gpu_stress_test.sh --skip-single    # Only run multi-GPU test
#   ./gpu_stress_test.sh --skip-multi     # Only run single-GPU tests
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MAX_GPU=7                  # GPU index 0..MAX_GPU (inclusive), 0-7 = 8 GPUs
MODEL_SIZE="small"
BATCH_SIZE=0               # 0 = auto-detect
EPOCHS=3                   # Just enough to stress test
DATA_DIR="./imagenet1k"
OUTPUT_BASE="./outputs_stress_test"
NUM_WORKERS=4
WANDB_OFFLINE=""
SKIP_SINGLE=false
SKIP_MULTI=false
MASTER_PORT=29500

# ── Parse CLI args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-gpu)       MAX_GPU="$2"; shift 2 ;;
        --model-size)    MODEL_SIZE="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --data-dir)      DATA_DIR="$2"; shift 2 ;;
        --output-dir)    OUTPUT_BASE="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --wandb-offline) WANDB_OFFLINE="--wandb-offline"; shift ;;
        --skip-single)   SKIP_SINGLE=true; shift ;;
        --skip-multi)    SKIP_MULTI=true; shift ;;
        --master-port)   MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$((MAX_GPU + 1))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_BASE}/logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  CliffordNet GPU Stress Test${NC}"
echo -e "${CYAN}================================================================${NC}"
echo -e "  GPUs to test    : 0 - ${MAX_GPU} (${NUM_GPUS} total)"
echo -e "  Model size       : ${MODEL_SIZE}"
echo -e "  Batch size       : ${BATCH_SIZE} (0=auto)"
echo -e "  Epochs           : ${EPOCHS}"
echo -e "  Data dir         : ${DATA_DIR}"
echo -e "  Log dir          : ${LOG_DIR}"
echo -e "  Skip single-GPU  : ${SKIP_SINGLE}"
echo -e "  Skip multi-GPU   : ${SKIP_MULTI}"
echo -e "${CYAN}================================================================${NC}"
echo ""

FAIL_COUNT=0
PASS_COUNT=0
declare -a RESULTS=()

# ── Helper: run a test and record result ─────────────────────────────────────
run_test() {
    local test_name="$1"
    local log_file="$2"
    shift 2
    local cmd=("$@")

    echo -e "${YELLOW}[RUN]${NC} ${test_name}"
    echo "  -> Log: ${log_file}"
    echo "  -> CMD: ${cmd[*]}"
    echo ""

    local start_time
    start_time=$(date +%s)

    if "${cmd[@]}" > "${log_file}" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${GREEN}[PASS]${NC} ${test_name}  (${elapsed}s)"
        RESULTS+=("PASS: ${test_name} (${elapsed}s)")
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${RED}[FAIL]${NC} ${test_name}  (${elapsed}s)"
        echo -e "${RED}       Check log: ${log_file}${NC}"
        # Print last 20 lines of log for quick diagnosis
        echo -e "${RED}       Last 20 lines of log:${NC}"
        tail -20 "${log_file}" | sed 's/^/         /'
        RESULTS+=("FAIL: ${test_name} (${elapsed}s) -> ${log_file}")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# =============================================================================
# Phase 1: Single-GPU tests (one per card, run in parallel)
# =============================================================================

if [[ "${SKIP_SINGLE}" == false ]]; then
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Phase 1: Single-GPU Stress Tests (parallel, one per card)${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo ""

    declare -a PIDS=()
    declare -a GPU_IDS=()
    declare -a LOG_FILES=()
    declare -a TEST_NAMES=()

    for gpu_id in $(seq 0 "${MAX_GPU}"); do
        test_name="SingleGPU_cuda${gpu_id}"
        log_file="${LOG_DIR}/single_gpu${gpu_id}.log"
        output_dir="${OUTPUT_BASE}/single_gpu${gpu_id}"
        mkdir -p "${output_dir}"

        echo -e "${YELLOW}[LAUNCH]${NC} ${test_name} -> ${log_file}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        uv run python train_imagenet1k.py \
            --data-dir "${DATA_DIR}" \
            --model-size "${MODEL_SIZE}" \
            --batch-size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --num-gpus 1 \
            --num-nodes 1 \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${output_dir}" \
            --wandb-project "CliffordNet" \
            --wandb-run-name "stress_single_gpu${gpu_id}_${TIMESTAMP}" \
            ${WANDB_OFFLINE} \
            > "${log_file}" 2>&1 &

        PIDS+=($!)
        GPU_IDS+=("${gpu_id}")
        LOG_FILES+=("${log_file}")
        TEST_NAMES+=("${test_name}")
    done

    echo ""
    echo -e "${CYAN}Waiting for ${#PIDS[@]} single-GPU tests to complete...${NC}"
    echo ""

    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        gpu_id="${GPU_IDS[$i]}"
        log_file="${LOG_FILES[$i]}"
        test_name="${TEST_NAMES[$i]}"

        if wait "${pid}"; then
            echo -e "${GREEN}[PASS]${NC} ${test_name} (PID ${pid})"
            RESULTS+=("PASS: ${test_name}")
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "${RED}[FAIL]${NC} ${test_name} (PID ${pid})"
            echo -e "${RED}       Check log: ${log_file}${NC}"
            tail -10 "${log_file}" | sed 's/^/         /'
            RESULTS+=("FAIL: ${test_name} -> ${log_file}")
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    echo ""
fi

# =============================================================================
# Phase 2: Multi-GPU DDP test (all GPUs together)
# =============================================================================

if [[ "${SKIP_MULTI}" == false ]]; then
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Phase 2: Multi-GPU DDP Stress Test (${NUM_GPUS} GPUs)${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo ""

    multi_output_dir="${OUTPUT_BASE}/multi_gpu_${NUM_GPUS}"
    mkdir -p "${multi_output_dir}"
    multi_log="${LOG_DIR}/multi_gpu_${NUM_GPUS}.log"

    # Build CUDA_VISIBLE_DEVICES string: "0,1,2,...,MAX_GPU"
    CUDA_DEVS=$(seq -s, 0 "${MAX_GPU}")

    run_test "MultiGPU_DDP_${NUM_GPUS}x" "${multi_log}" \
        env CUDA_VISIBLE_DEVICES="${CUDA_DEVS}" \
        uv run torchrun \
            --standalone \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${MASTER_PORT}" \
            train_imagenet1k.py \
            --data-dir "${DATA_DIR}" \
            --model-size "${MODEL_SIZE}" \
            --batch-size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --num-gpus "${NUM_GPUS}" \
            --num-nodes 1 \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${multi_output_dir}" \
            --wandb-project "CliffordNet" \
            --wandb-run-name "stress_multi_${NUM_GPUS}gpu_${TIMESTAMP}" \
            ${WANDB_OFFLINE}
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  Stress Test Summary${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

for result in "${RESULTS[@]}"; do
    if [[ "${result}" == PASS* ]]; then
        echo -e "  ${GREEN}${result}${NC}"
    else
        echo -e "  ${RED}${result}${NC}"
    fi
done

echo ""
TOTAL=$((PASS_COUNT + FAIL_COUNT))
echo -e "  Total: ${TOTAL}  |  ${GREEN}Passed: ${PASS_COUNT}${NC}  |  ${RED}Failed: ${FAIL_COUNT}${NC}"
echo -e "  Logs : ${LOG_DIR}"
echo ""

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo -e "${RED}Some tests FAILED. Check the log files above for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests PASSED.${NC}"
    exit 0
fi
