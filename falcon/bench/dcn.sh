#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dcn.py \
--use-gpu \
--arch-sparse-feature-size=128 \
--max-ind-range=40000000 \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/train.txt \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=4096 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"
