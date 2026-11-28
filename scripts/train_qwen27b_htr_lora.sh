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
: "${QWEN_HRT_TRAIN_DATA:?Set QWEN_HRT_TRAIN_DATA to the converted train JSONL}"
: "${QWEN_HRT_OUTPUT_DIR:?Set QWEN_HRT_OUTPUT_DIR to a new output directory}"

if [[ ! -f "$QWEN_TRAIN_MODEL_DIR/config.json" ]]; then
  echo "Model directory is missing config.json: $QWEN_TRAIN_MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$QWEN_HRT_TRAIN_DATA" ]]; then
  echo "Training JSONL not found: $QWEN_HRT_TRAIN_DATA" >&2
  exit 2
fi
if [[ "$QWEN_TRAIN_MODEL_DIR" =~ [Ff][Pp]8 ]]; then
  echo "Use the BF16 Qwen3.5-27B checkpoint for this pilot, not the FP8 inference checkpoint." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft \
  --model "$QWEN_TRAIN_MODEL_DIR" \
  --train_type lora \
  --dataset "$QWEN_HRT_TRAIN_DATA" \
  --torch_dtype bfloat16 \
  --tuner_backend peft \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --max_length 16384 \
  --gradient_checkpointing true \
  --fsdp fsdp2 \
  --save_steps 50 \
  --save_total_limit 2 \
  --create_checkpoint_symlink true \
  --logging_steps 10 \
  --add_version false \
  --output_dir "$QWEN_HRT_OUTPUT_DIR"
