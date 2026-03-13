#!/bin/bash
# Master script to run both SSD and SGLang NCCL comparison.
#
# Runs sequentially by default (safer for shared GPU resources).
# Captures logs to ssd_nccl_log.txt and sglang_nccl_log.txt.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_nccl_comparison.sh         # sequential, specify GPUs
#   SSD_GPUS=2,3 SGLANG_GPUS=4,5 bash scripts/run_nccl_comparison.sh    # different GPUs per system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${REPO_DIR}"

SSD_LOG="${LOG_DIR}/ssd_nccl_log.txt"
SGLANG_LOG="${LOG_DIR}/sglang_nccl_log.txt"

SSD_GPUS="${SSD_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
SGLANG_GPUS="${SGLANG_GPUS:-${CUDA_VISIBLE_DEVICES:-6,7}}"

echo "============================================================"
echo "NCCL Comparison: SSD vs SGLang"
echo "============================================================"
echo "SSD GPUs:    $SSD_GPUS"
echo "SGLang GPUs: $SGLANG_GPUS"
echo "SSD log:     $SSD_LOG"
echo "SGLang log:  $SGLANG_LOG"
echo "============================================================"

# --- Run SSD ---
echo ""
echo "[1/2] Running SSD..."
(
    cd "$REPO_DIR"
    source /work/avner/git/ssd-orig/.venv/bin/activate
    export CUDA_VISIBLE_DEVICES="$SSD_GPUS"
    export SSD_NCCL_LOG=1
    python scripts/compare_nccl_ssd.py 2>&1
) | tee "$SSD_LOG"
echo ""
echo "[1/2] SSD done. Log saved to $SSD_LOG"

# --- Run SGLang ---
echo ""
echo "[2/2] Running SGLang..."
echo "SGLang server will start (no warmup), then we send a request with max_new_tokens=64."
(
    cd "$REPO_DIR"
    source /work/avner/git/sglang-private-main/.venv/bin/activate
    export CUDA_VISIBLE_DEVICES="$SGLANG_GPUS"
    export SSD_NCCL_LOG=1

    SGLANG_PORT=30000

    # Launch server in background with matching params: page_size=1, K=6, no warmup
    python -m sglang.launch_server \
        --model-path meta-llama/Llama-3.2-1B-Instruct \
        --speculative-algorithm ASYNC_STANDALONE \
        --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
        --speculative-num-steps 6 \
        --page-size 1 \
        --context-length 4096 \
        --tp-size 1 \
        --port $SGLANG_PORT \
        --skip-server-warmup 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready (poll /health endpoint for HTTP 200)
    TIMEOUT=300
    ELAPSED=0
    echo "Waiting for SGLang server to be ready..."
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited early"
            break
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$SGLANG_PORT/health 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo "Server is ready (HTTP 200, elapsed=${ELAPSED}s). Sending request..."
            break
        fi
    done

    # Send request with matching params: prompt, temperature=0, max_new_tokens=64
    curl -s http://localhost:$SGLANG_PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": "The capital city of France is",
            "temperature": 0,
            "max_tokens": 64
        }' &
    CURL_PID=$!

    # Wait for the request to finish (or timeout)
    REQ_TIMEOUT=120
    REQ_ELAPSED=0
    while [ $REQ_ELAPSED -lt $REQ_TIMEOUT ]; do
        if ! kill -0 $CURL_PID 2>/dev/null; then
            echo "Request completed."
            break
        fi
        sleep 2
        REQ_ELAPSED=$((REQ_ELAPSED + 2))
    done
    kill $CURL_PID 2>/dev/null || true
    wait $CURL_PID 2>/dev/null || true

    sleep 5  # let logs flush

    # Kill the server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
) | tee "$SGLANG_LOG"
echo ""
echo "[2/2] SGLang done. Log saved to $SGLANG_LOG"

echo ""
echo "============================================================"
echo "Comparison complete!"
echo "  SSD log:    $SSD_LOG"
echo "  SGLang log: $SGLANG_LOG"
echo ""
echo "Key patterns to grep for:"
echo "  grep 'NCCL_LOG SEND_PREFILL' ssd_nccl_log.txt sglang_nccl_log.txt"
echo "  grep 'NCCL_LOG SEND_SPEC' ssd_nccl_log.txt sglang_nccl_log.txt"
echo "  grep 'NCCL_LOG RECV_SPEC_RESP' ssd_nccl_log.txt sglang_nccl_log.txt"
echo "============================================================"
