#!/usr/bin/env bash
# FALCON vs LEAF 비교 실험 스크립트
# Avazu CR=10, 100, 1000 / KDD12 CR=10, 100
#
# 사용법:
#   bash run_falcon.sh avazu    → Avazu 실험만
#   bash run_falcon.sh kdd12    → KDD12 실험만
#   bash run_falcon.sh all      → 전체 실험 (기본값)
#
# 결과 로그: ./logs/falcon_<dataset>_CR<ratio>_<method>.log

mkdir -p ./logs ./checkpoints

DATASET=${1:-all}

# ===========================================================================
# 공통 파라미터
# ===========================================================================
COMMON_AVAZU="
  --arch-sparse-feature-size=16
  --arch-mlp-bot=13-512-256-64-16
  --arch-mlp-top=512-256-1
  --data-generation=dataset
  --data-set=avazu
  --loss-function=bce
  --round-targets=True
  --learning-rate=0.1
  --mini-batch-size=128
  --print-freq=1024
  --test-freq=1024
  --nepochs=1
  --print-time
  --test-mini-batch-size=16384
  --test-num-workers=0
  --cat-path=./input/avazu_cat.bin
  --dense-path=
  --label-path=./input/avazu_label.bin
  --count-path=./input/avazu_count.bin
  --use-gpu
  --use-adaptive-encoding
  --long-tail-memory-ratio=0.9
  --num-batches=50000
"

FALCON_PARAMS="
  --use-falcon
  --falcon-K-base=256
  --falcon-lambda-cost=0.1
"

# ===========================================================================
# Avazu 실험
# ===========================================================================
run_avazu() {
    for CR in 10 100 1000; do
        echo "============================================"
        echo " Avazu  CR=${CR}  LEAF baseline"
        echo "============================================"
        CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
            ${COMMON_AVAZU} \
            --compression-ratio=${CR} \
            --save-model="./checkpoints/avazu_leaf_CR${CR}.pt" \
            2>&1 | tee ./logs/falcon_avazu_CR${CR}_leaf.log

        echo "============================================"
        echo " Avazu  CR=${CR}  FALCON"
        echo "============================================"
        CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
            ${COMMON_AVAZU} \
            --compression-ratio=${CR} \
            ${FALCON_PARAMS} \
            --save-model="./checkpoints/avazu_falcon_CR${CR}.pt" \
            2>&1 | tee ./logs/falcon_avazu_CR${CR}_falcon.log
    done
}

# ===========================================================================
# KDD12 실험
# ===========================================================================
run_kdd12() {
    for CR in 10 100; do
        echo "============================================"
        echo " KDD12  CR=${CR}  LEAF baseline"
        echo "============================================"
        CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
            --test-freq=1024 \
            --nepochs=1 \
            --print-time \
            --test-mini-batch-size=16384 \
            --test-num-workers=0 \
            --cat-path="./input/kdd12_cat.bin" \
            --dense-path="" \
            --label-path="./input/kdd12_label.bin" \
            --count-path="./input/kdd12_count.bin" \
            --use-gpu \
            --use-adaptive-encoding \
            --compression-ratio=${CR} \
            --long-tail-memory-ratio=0.9 \
            --num-batches=50000 \
            --save-model="./checkpoints/kdd12_leaf_CR${CR}.pt" \
            2>&1 | tee ./logs/falcon_kdd12_CR${CR}_leaf.log

        echo "============================================"
        echo " KDD12  CR=${CR}  FALCON"
        echo "============================================"
        CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
            --test-freq=1024 \
            --nepochs=1 \
            --print-time \
            --test-mini-batch-size=16384 \
            --test-num-workers=0 \
            --cat-path="./input/kdd12_cat.bin" \
            --dense-path="" \
            --label-path="./input/kdd12_label.bin" \
            --count-path="./input/kdd12_count.bin" \
            --use-gpu \
            --use-adaptive-encoding \
            --compression-ratio=${CR} \
            --long-tail-memory-ratio=0.9 \
            --num-batches=50000 \
            ${FALCON_PARAMS} \
            --save-model="./checkpoints/kdd12_falcon_CR${CR}.pt" \
            2>&1 | tee ./logs/falcon_kdd12_CR${CR}_falcon.log
    done
}

# ===========================================================================
# 결과 요약
# ===========================================================================
summarize() {
    echo ""
    echo "=============================="
    echo " 실험 결과 요약 (AUC)"
    echo "=============================="
    for log in ./logs/falcon_*.log; do
        name=$(basename "$log" .log)
        auc=$(grep "best auc" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+(?= %, best)' | head -1)
        echo "  ${name}: AUC=${auc:-N/A}"
    done
}

# ===========================================================================
# 실행
# ===========================================================================
case "${DATASET}" in
    avazu) run_avazu ;;
    kdd12) run_kdd12 ;;
    all)   run_avazu; run_kdd12 ;;
    *)
        echo "사용법: bash run_falcon.sh [avazu|kdd12|all]"
        exit 1
        ;;
esac

summarize
