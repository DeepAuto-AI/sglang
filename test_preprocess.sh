#!/bin/bash -l

#source /home/geon/.bashrc
#source /home/geon/sglang-video/.venv/bin/activate
source /home/park/devel/.venv/bin/activate
export AKS_SAMPLING_RATE=1.0
python test_preprocess.py
