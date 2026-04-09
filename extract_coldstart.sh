#!/usr/bin/env bash
# Cold Start 실험 결과 추출
# [COLDSTART] 라인에서 배치번호와 AUC를 파싱하여 txt 파일로 저장
# 출력 형식: <배치번호> <AUC>

mkdir -p ./logs

extract() {
    local log_file=$1
    local out_file=$2
    local label=$3

    if [ ! -f "$log_file" ]; then
        echo "[WARNING] 로그 파일 없음: $log_file"
        return
    fi

    # [COLDSTART] batch=N auc=X.XXXX loss=X.XXXX 에서 배치번호와 AUC 추출
    grep "\[COLDSTART\]" "$log_file" \
        | awk '{
            for (i=1; i<=NF; i++) {
                if ($i ~ /^batch=/) { split($i, a, "="); batch=a[2] }
                if ($i ~ /^auc=/)   { split($i, a, "="); auc=a[2]   }
            }
            print batch, auc
        }' > "$out_file"

    echo "[$label] 추출 완료 → $out_file ($(wc -l < "$out_file") 배치)"
}

extract "./logs/leaf_coldstart.log" "./logs/leaf_auc_by_batch.txt" "LEAF"
extract "./logs/cls_coldstart.log"  "./logs/cls_auc_by_batch.txt"  "CLS-LEAF"

echo ""
echo "=== LEAF AUC (초반 10배치) ==="
head -10 ./logs/leaf_auc_by_batch.txt 2>/dev/null || echo "(파일 없음)"

echo ""
echo "=== CLS-LEAF AUC (초반 10배치) ==="
head -10 ./logs/cls_auc_by_batch.txt 2>/dev/null || echo "(파일 없음)"
