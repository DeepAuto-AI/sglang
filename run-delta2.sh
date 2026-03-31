HIP_DEBUG_DELTA_QSA_IMSAVE_STATE=0 \
HIP_DEBUG_DELTA_QSA_IMSAVE=0 \
HIP_DEBUG_NEED_CHECKOUT=0 \
HIP_DEBUG_NEED_CHECKOUT_ROOT=/data/jeff/delta/datasave/bsa_pool \
HIP_DEBUG_DELTA_QSA=1 \
BSA_BLOCK_K=64 \
BSA_K=$k \
BSA_EXACT_K=$exact_k \
BSA_WINNER_TREE=$BSA_WINNER_TREE_ \
REVERSE_ITER=$REVERSE_ITER_ \
HIP_DEBUG_RECOMPUTE_SPLIT=0 \
TRITON_PRINT_AUTOTUNING=1 \
SRT_WARMUP_ALL_SEQ_LENS=0 \
HIP_DEBUG_FA3_MIXING_LEN=0 \
HIP_DEBUG_FORCE_DENSE_DECODE=0 \
HIP_DEBUG_USING_DENSE_PREFILL=0 \
PASSKEY_DECODE_LEN=10 \
PASSKEY_LEN=100 \
SA_BLOCK_SIZE=256 \
SA_DECODE_BLOCK_SIZE=128 \
HIP_DISABLE_AUTOTUNE=0 \
HIP_DEBUG=0 \
HIP_DEBUG_BENCH=0 \
HIP_DEBUG_CAPTURE_DECORATOR=0 \
CUDA_LAUNCH_BLOCKING=0 \
    python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --model-path $MODEL_PATH \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.6 \
    --ep-size $N_GPUS \
    --tp-size $N_GPUS \
    --chunked-prefill-size $CONTEXT_LEN \
    --max-prefill-tokens $CONTEXT_LEN \
    --disable-cuda-graph \
    --context-length $CONTEXT_LEN \
    --max-total-tokens $CONTEXT_LEN \
    --hip-attention-config /home/geon/delta/hip-attention-delta/configs/rebuttal/llama31_noextend_dense_decode.json \
    --hip-attention-config-override-json '{"using_extend": false, "__delta_attention_args": "window_0-diff_1-w_16-dense_decode-bsa_meanpool", "__seq_thresh_fa3": 0}' \
    --attention-backend hip_attention \
    --disable-radix-cache \
    --max-running-requests 1 \
    --trust-remote-code

