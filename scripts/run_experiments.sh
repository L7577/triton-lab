#!/usr/bin/env bash
# Experiment 1 & 2 runner — Triton single-model baseline + dynamic batching
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_LAB="$(dirname "$SCRIPT_DIR")"
MODEL_REPO="$TRITON_LAB/model_repository"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:22.12-py3"
CONTAINER_NAME="triton-lab"
TRITON_HTTP="http://localhost:8000"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
title(){ echo -e "${CYAN}============================================================${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}============================================================${NC}"; }

# --- helpers ---
check_prereqs() {
    if ! docker images "$TRITON_IMAGE" --format "ok" 2>/dev/null | grep -q ok; then
        err "Triton image $TRITON_IMAGE not found. Pull it first."
        exit 1
    fi
    if [ ! -f "$MODEL_REPO/identity_onnx/1/model.onnx" ]; then
        err "Model not found. Run: python3 $TRITON_LAB/generate_onnx.py $TRITON_LAB/identity.onnx"
        err "Then: mkdir -p $MODEL_REPO/identity_onnx/1 && cp $TRITON_LAB/identity.onnx $MODEL_REPO/identity_onnx/1/model.onnx"
        exit 1
    fi
}

set_config() {
    local config_file="$TRITON_LAB/configs/$1"
    cp "$config_file" "$MODEL_REPO/identity_onnx/config.pbtxt"
    log "Config: $(basename "$config_file")"
}

start_triton() {
    log "Starting Triton container..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d --rm --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$MODEL_REPO:/models" \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        "$TRITON_IMAGE" \
        tritonserver --model-repository=/models > /dev/null 2>&1
}

wait_ready() {
    log "Waiting for model READY..."
    for i in $(seq 1 30); do
        if curl -s "$TRITON_HTTP/v2/models/identity_onnx/ready" 2>/dev/null | grep -q "true\|ready"; then
            log "Model ready (took ${i}s)"
            return 0
        fi
        sleep 1
    done
    err "Model failed to become ready within 30s"
    docker logs --tail 30 "$CONTAINER_NAME"
    exit 1
}

stop_triton() {
    log "Stopping Triton..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
}

# GPU snapshot before/after
gpu_snapshot() {
    local label="$1"
    echo "--- GPU snapshot [$label] ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}

# Triton metrics snapshot
triton_metrics() {
    local label="$1"
    echo "--- Triton metrics [$label] ---"
    curl -s "$TRITON_HTTP:8002/metrics" 2>/dev/null | grep -E "^nv_" | head -10 || echo "metrics unavailable"
}

# --- main ---
run_experiment() {
    local exp_name="$1"
    local config_name="$2"
    local sweep="$3"
    local duration="${4:-10}"

    title "$exp_name"
    set_config "$config_name"
    start_triton
    wait_ready

    gpu_snapshot "before"
    triton_metrics "before"

    log "Running benchmark: sweep=$sweep duration=${duration}s"
    python3 "$TRITON_LAB/benchmark.py" \
        --url "$TRITON_HTTP/v2/models/identity_onnx/infer" \
        --model identity_onnx \
        --sweep "$sweep" \
        --duration "$duration"

    gpu_snapshot "after"
    triton_metrics "after"
    stop_triton
}

# ============================================
case "${1:-}" in
    exp1)
        check_prereqs
        run_experiment \
            "Experiment 1: Single-Model Baseline" \
            "exp1_baseline.pbtxt" \
            "1,2,4" \
            "${2:-10}"
        ;;
    exp2-batch)
        check_prereqs
        run_experiment \
            "Experiment 2a: Single-Instance + Dynamic Batching" \
            "exp2_batch.pbtxt" \
            "1,2,4,8,16" \
            "${2:-10}"
        ;;
    exp2-multi)
        check_prereqs
        run_experiment \
            "Experiment 2b: Multi-Instance (2) + Dynamic Batching" \
            "exp2_multi.pbtxt" \
            "1,2,4,8,16" \
            "${2:-10}"
        ;;
    exp2-all)
        check_prereqs
        bash "$0" exp2-batch "${2:-10}"
        bash "$0" exp2-multi "${2:-10}"
        ;;
    all)
        check_prereqs
        bash "$0" exp1 "${2:-10}"
        bash "$0" exp2-batch "${2:-10}"
        bash "$0" exp2-multi "${2:-10}"
        ;;
    *)
        echo "Usage: $0 {exp1|exp2-batch|exp2-multi|exp2-all|all} [duration_sec]"
        exit 1
        ;;
esac
