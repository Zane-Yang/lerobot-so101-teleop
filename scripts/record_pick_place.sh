#!/usr/bin/env bash
# Record a real pick-and-place dataset with the gamepad joint teleoperator.
# Customize PORT / CAMERA / REPO / ROOT / TASK to your arm before running.
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-zane_so101_follower}"
CAMERA_ID="${CAMERA_ID:-2}"
REPO_ID="${REPO_ID:-zane/pick_place_block}"
ROOT="${ROOT:-./datasets/pick_place_block}"
TASK="${TASK:-Pick up the candy and place it into the lid}"
N_EPISODES="${N_EPISODES:-50}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --teleop.type=gamepad_joints \
  --teleop.id=gamepad \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_ID}, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${ROOT}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${N_EPISODES}" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true