#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

dlrm_pt_bin="python3 dlrm_s_pytorch.py"

CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dlrm_s_pytorch.py \
--use-gpu \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=avazu \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--test-freq=40960 \
--nepochs=1 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--cat-path="./input/avazu_cat.bin" \
--dense-path="" \
--label-path="./input/avazu_label.bin" \
--count-path="./input/avazu_count.bin" \
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"

