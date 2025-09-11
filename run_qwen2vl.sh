#!/bin/bash -l

source /home/geon/.bashrc
source /home/geon/sglang-video/.venv/bin/activate

export NFRAMES=64
export SGLANG_VLM_CACHE_SIZE_MB=1000
export FRAME_SELECTION_METHOD=aks
export AKS_MODEL_CARD=openai/clip-vit-base-patch32

python -m sglang.launch_server \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--host 0.0.0.0 \
--port 30000 \
--tp-size 2 \
--mem-fraction-static 0.4 \
--disable-radix-cache \
--cuda-graph-max-bs 1
