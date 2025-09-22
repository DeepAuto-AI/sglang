#!/bin/bash

MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_MODEL_LEN=131072

# MODEL=Qwen/Qwen3-4B-Instruct-2507
# MAX_MODEL_LEN=262144

CUDA_VISIBLE_DEVICES=3 \
python minference-server-hf.py \
  --model $MODEL \
  --host 0.0.0.0 \
  --port 8082 \
