#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Optional: schedule reverse on batch index j (0-based) by passing extra flags via "$@":
#   - run only while j < N:   --reverse-stop-batch=N
#   - run only for j >= M:   --reverse-start-batch=M

python dlrm_s_pytorch.py \
  --use-adaptive-encoding \
  --use-cls \
  --use-reverse-transfer \
  --cls-alpha-min=0.1 \
  --cls-alpha-max=0.9 \
  --cls-surge-k=24 \
  --cls-decay-freq=10000 \
  --cls-decay-rate=0.8 \
  --cls-decay-grace=100 \
  --reverse-beta-min=0.05 \
  --reverse-beta-max=0.3 \
  --reverse-sim-threshold-min=0.1 \
  --reverse-sim-threshold=0.5 \
  --reverse-freq=1000 \
  --reverse-top-k-fields=3 \
  --reverse-auto-select-min-fields=4 \
  --compression-ratio=100 \
  "$@"

