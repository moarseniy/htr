#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_ENV="${QWEN_HRT_ENV_FILE:-configs/qwen-htr-train.env}"
if [[ -f "$TRAIN_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$TRAIN_ENV"
  set +a
fi

: "${QWEN_TRAIN_MODEL_DIR:?Set QWEN_TRAIN_MODEL_DIR to the local BF16 Qwen3.5-27B checkpoint}"
: "${QWEN_HRT_OUTPUT_DIR:?Set QWEN_HRT_OUTPUT_DIR to the training output directory}"
QWEN_HRT_MERGED_DIR="${QWEN_HRT_MERGED_DIR:-${QWEN_HRT_OUTPUT_DIR}/merged}"

swift export \
  --model "$QWEN_TRAIN_MODEL_DIR" \
  --adapters "${QWEN_HRT_OUTPUT_DIR}/last" \
  --merge_lora true \
  --output_dir "$QWEN_HRT_MERGED_DIR"

echo "Merged model: $QWEN_HRT_MERGED_DIR"
