#!/usr/bin/env bash
# LEAF (baseline) Cold Start 실험
# 전환이 집중되는 초반 500 배치에서 매 배치마다 AUC 측정

mkdir -p ./logs ./checkpoints

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
  --test-freq=1024 \
  --nepochs=1 \
  --print-time \
  --test-mini-batch-size=16384 \
  --test-num-workers=0 \
  --cat-path="./input/avazu_cat.bin" \
  --dense-path="" \
  --label-path="./input/avazu_label.bin" \
  --count-path="./input/avazu_count.bin" \
  --use-gpu \
  --use-adaptive-encoding \
  --compression-ratio=100 \
  --long-tail-memory-ratio=0.9 \
  --num-batches=50000 \
  --save-model="./checkpoints/leaf_coldstart.pt" \
  2>&1 | tee ./logs/leaf_coldstart.log
