#!/usr/bin/env bash
set -euo pipefail

EPOCHS=${EPOCHS:-10}
LINEAR_EPOCHS=${LINEAR_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
REPLAY_SIZE=${REPLAY_SIZE:-500}
DEVICE=${DEVICE:-auto}
LOG_DIR=${LOG_DIR:-./logs/ablation}
SEED=${SEED:-42}
EXTRA_ARGS=${EXTRA_ARGS:-""}

run_one() {
  local name=$1
  local patch_mode=$2
  local codebook_mode=$3
  local stamp
  local run_id
  local run_name

  stamp=$(date +%Y%m%d_%H%M%S)
  run_id=$(uuidgen | tr 'A-Z' 'a-z' | cut -c1-8)
  run_name="${name}_${stamp}_${run_id}"

  python train.py \
    --epochs "${EPOCHS}" \
    --linear-epochs "${LINEAR_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --replay-size "${REPLAY_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --log-dir "${LOG_DIR}" \
    --run-name "${run_name}" \
    --patch-mode "${patch_mode}" \
    --codebook-mode "${codebook_mode}" \
    --patch-prototype-mode class_mean_ema \
    ${EXTRA_ARGS}
}

run_one "normal_etf_fixed" "normal" "fixed"
run_one "roi_etf_fixed" "roi" "fixed"
run_one "normal_etf_ema_mean" "normal" "ema_mean"
