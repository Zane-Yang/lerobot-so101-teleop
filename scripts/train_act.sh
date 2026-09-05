#!/usr/bin/env bash
# Train ACT locally (fits RTX 3060 6 GB). Override env vars to reuse on other datasets.
set -euo pipefail

REPO_ID="${REPO_ID:-zane/pick_place_block_clean}"
ROOT="${ROOT:-./datasets/pick_place_block_clean}"
OUT_DIR="${OUT_DIR:-output/train/act_pick_place_block}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-30000}"

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="${OUT_DIR}" \
  --batch_size="${BATCH}" \
  --steps="${STEPS}" \
  --policy.push_to_hub=false