#!/bin/bash

#WARNING: must have compiled PyTorch

cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kdd12 \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--print-time \
--test-freq=40960 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--nepochs=1 \
--use-gpu \
--cat-path="./input/kdd12_cat.bin" \
--dense-path="" \
--label-path="./input/kdd12_label.bin" \
--count-path="./input/kdd12_count.bin" \
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"
