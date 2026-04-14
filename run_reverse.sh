#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

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
  --reverse-sim-threshold=0.5 \
  --compression-ratio=100 \
  "$@"

