#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python dlrm_s_pytorch.py \
  --use-adaptive-encoding \
  --use-similarity-hashing \
  --sim-update-freq=1000 \
  --compression-ratio=100 \
  "$@"
