#!/usr/bin/env bash
# Real-robot autonomous eval (base) then optional episode recording.
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-output/train/act_pick_place_block/checkpoints/last/pretrained_model}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
CAMERA_ID="${CAMERA_ID:-2}"
TASK="${TASK:-Pick up the candy and place it into the lid}"
MODE="${MODE:-base}"   # base = autonomous no-record, episodic = record N episodes

if [[ "${MODE}" == "episodic" ]]; then
  REPO_ID="${REPO_ID:-zane/eval_pick_place_block}"
  ROOT="${ROOT:-./datasets/eval_pick_place_block}"
  N_EPISODES="${N_EPISODES:-10}"
  lerobot-rollout \
    --strategy.type=episodic \
    --policy.path="${POLICY_PATH}" \
    --robot.type=so101_follower \
    --robot.port="${ROBOT_PORT}" \
    --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_ID}, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${ROOT}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes="${N_EPISODES}" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --display_data=true
else
  lerobot-rollout \
    --strategy.type=base \
    --policy.path="${POLICY_PATH}" \
    --robot.type=so101_follower \
    --robot.port="${ROBOT_PORT}" \
    --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_ID}, width: 640, height: 480, fps: 30}}" \
    --task="${TASK}" \
    --duration=60 \
    --display_data=true
fi