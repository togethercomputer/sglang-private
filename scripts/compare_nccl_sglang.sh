#!/bin/bash
# Launch SGLang server with NCCL logging enabled to capture async spec decode data flow.
# Skips default warmup and sends a manual request with max_new_tokens=64.
#
# Usage:
#   bash scripts/compare_nccl_sglang.sh
#
# The server will start, send the request, and stay running.
# Kill with Ctrl+C when done capturing logs.

set -euo pipefail

cd /work/avner/git/sglang-private-main
source /work/avner/git/sglang-private-main/.venv/bin/activate

export SSD_NCCL_LOG=1

SGLANG_PORT=30000

echo "============================================================"
echo "Starting SGLang server with SSD_NCCL_LOG=1"
echo "  page_size=1, speculative_num_steps=6, context_length=4096"
echo "  Prompt: 'The capital city of France is' (max_new_tokens=64)"
echo "============================================================"

python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-algorithm ASYNC_STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
    --speculative-num-steps 6 \
    --page-size 1 \
    --context-length 4096 \
    --tp-size 1 \
    --port $SGLANG_PORT \
    --skip-server-warmup &
SERVER_PID=$!

# Wait for server to be ready
TIMEOUT=300
ELAPSED=0
echo "Waiting for server to be ready..."
while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process exited early"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$SGLANG_PORT/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server is ready (HTTP 200, elapsed=${ELAPSED}s). Sending request..."
        break
    fi
done

# Send request with max_new_tokens=64
echo ""
echo "============================================================"
echo "Sending completion request (max_tokens=64)..."
echo "============================================================"
curl -s http://localhost:$SGLANG_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "The capital city of France is",
        "temperature": 0,
        "max_tokens": 64
    }'
echo ""

# Keep server running for log inspection
echo "============================================================"
echo "Request complete. Server still running. Kill with Ctrl+C."
echo "============================================================"
wait $SERVER_PID
