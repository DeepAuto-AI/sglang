#!/bin/bash
set -euxo pipefail

echo "Starting server..."
export HIP_DEBUG_DELTA_QSA=1
export HIP_DEBUG_RECOMPUTE_SPLIT=0
export TRITON_PRINT_AUTOTUNING=1
export SRT_WARMUP_ALL_SEQ_LENS=0
export HIP_DELTA_ATTENTION_ARGS=dense_decode-window_1024-expsink_256
export HIP_DEBUG_FA3_MIXING_LEN=0
export HIP_DEBUG_FORCE_DENSE_DECODE=0
export HIP_DEBUG_USING_DENSE_PREFILL=0
export PASSKEY_DECODE_LEN=10
export PASSKEY_LEN=100
export SA_BLOCK_SIZE=256
export SA_DECODE_BLOCK_SIZE=128
export HIP_DISABLE_AUTOTUNE=0
export HIP_DEBUG=0
export HIP_DEBUG_BENCH=1
export HIP_DEBUG_CAPTURE_DECORATOR=0
export CUDA_LAUNCH_BLOCKING=0

python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 8090 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype auto \
    --ep-size 2 \
    --tp-size 2 \
    --mem-fraction-static 0.65 \
    --chunked-prefill-size 131072 \
    --max-prefill-tokens 131072 \
    --cuda-graph-bs 1 \
    --context-length 131072 \
    --max-total-tokens 1024000 \
    --attention-backend refine_attn \
    --disable-radix-cache \
    --disable-cuda-graph \
    --max-running-requests 1 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --allow-auto-truncate #>> output.log 2>&1 &
