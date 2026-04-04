#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/train.txt \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
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
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"
