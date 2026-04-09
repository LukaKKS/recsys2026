#!/usr/bin/env bash
# Cold Start 실험 결과 추출
# [COLDSTART] batch=N auc=X.XXXX loss=X.XXXX 에서 배치번호와 AUC 파싱

mkdir -p ./logs

grep "COLDSTART" ./logs/leaf_coldstart.log \
  | awk -F'[ =]' '{print $3, $5}' \
  > ./logs/leaf_auc_by_batch.txt

grep "COLDSTART" ./logs/cls_coldstart.log \
  | awk -F'[ =]' '{print $3, $5}' \
  > ./logs/cls_auc_by_batch.txt

echo "LEAF AUC:"
cat ./logs/leaf_auc_by_batch.txt

echo ""
echo "CLS-LEAF AUC:"
cat ./logs/cls_auc_by_batch.txt
