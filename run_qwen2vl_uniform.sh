#!/bin/bash

export NFRAMES=64
export SGLANG_VLM_CACHE_SIZE_MB=1000
export FRAME_SELECTION_METHOD=uniform

python3 -m sglang.launch_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--host 0.0.0.0 \
--port 30000 \
--tp-size 2 \
--mem-fraction-static 0.4 \
--disable-radix-cache
