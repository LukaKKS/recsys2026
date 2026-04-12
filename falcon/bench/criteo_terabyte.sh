#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch

cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dlrm_s_pytorch.py \
--use-gpu \
--arch-sparse-feature-size=128 \
--arch-mlp-bot="13-512-256-128" \
--arch-mlp-top="1024-1024-512-256-1" \
--data-generation=dataset \
--data-set=terabyte \
--raw-data-file=./input/day \
--processed-data-file=./input/terabyte_processed.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=1.0 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=102400 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--nepochs=1 \
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"
