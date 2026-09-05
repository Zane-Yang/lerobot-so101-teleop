# Reproduction Pipeline

This file is a “from zero to real-robot policy” walkthrough. It mirrors the
exact steps used while developing this project, so a reviewer (or you) can
reproduce the same clean 6-DoF pick-and-place policy on an SO-101

---

## 1. Get the upstream

LeRobot is the base. Work from a **source checkout** (editable install) so that
the custom teleoperator classes and small integration hooks resolve as Python
modules:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# create/activate conda env once
pip install -e ".[feetech,dataset,kinematics]"
```

> The code in this repo (`lerobot_mods/`) is the **self-authored delta** that we
> applied on top of the above source tree. It is not a fork of upstream.

---

## 2. Hardware bring-up (one-time)

1. Enumerate the serial bus & set IDs once:
   ```bash
   lerobot-find-port
   lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
   ```
   (Customizer notes: verify `sts3215` baud / IDs per your build.)

2. Calibrate the follower (this writes a calibration JSON for your arm id):
   ```bash
   lerobot-calibrate \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=zane_so101_follower
   ```

3. Find the camera device:
   ```bash
   ls /dev/video*
   ```
   Pick the one reporting 640×480 @ 30 and point `--robot.cameras` to its
   `/dev/videoN`; we used `index_or_path: 2`.

---

## 3. Install our custom teleoperator files

Copy the content of the two `lerobot_mods/` folders over the matching locations
in the upstream **source** tree, or import them directly as a plugin. Concretely:

```
lerobot_mods/keyboard/  ->  lerobot/teleoperators/keyboard/   (extend existing)
lerobot_mods/gamepad/   ->  lerobot/teleoperators/gamepad/    (extend existing)
```

The classes register as `keyboard_joints` / `gamepad_joints` and are picked up
by the factory used in `lerobot-teleoperate` / `lerobot-record`.

---

## 4. Record a dataset (gamepad)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=zane_so101_follower \
  --teleop.type=gamepad_joints \
  --teleop.id=gamepad \
  --robot.cameras='{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id=zane/pick_place_block \
  --dataset.root=./datasets/pick_place_block \
  --dataset.single_task="Pick up the candy and place it into the lid" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true
```

Expected artifacts:
- `meta/info.json` – dataset codebase_version 3.0, fps 30,
  `action`/`observation.state` = 6-DoF, `observation.images.front` 640×480.
- one or more video chunks under `videos/observation.images.front/`.

> Tip: the arm joints use **degrees**, the gripper uses **0–100**, consistent
> with LeRobot’s SO normalization.

---

## 5. Inspect and prune episodes (optional)

### 5.1 Export an episode preview with pyAV

If upstream `lerobot-dataset-viz` is unavailable due to the `torchcodec`
linking problem (see TROUBLESHOOTING), use the shipped example:

```bash
python examples/dataset_episode_viewer.py \
  --repo-id zane/pick_place_block \
  --root ./datasets/pick_place_block \
  --episode 0 \
  --out preview_ep0.mp4
```

It renders the camera view plus joint `action`/`state` curves on the same
video timeline — great for quickly spotting a “wobbly” demonstration.

### 5.2 Remove bad episodes to a clean dataset

```bash
lerobot-edit-dataset \
  --repo_id zane/pick_place_block \
  --root ./datasets/pick_place_block \
  --new_repo_id zane/pick_place_block_clean \
  --new_root ./datasets/pick_place_block_clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[2,7,11]"
```

Keeps the raw set untouched, produces the curated training set.

> The example dataset `pick_place_block_clean` that we used has 48 episodes /
> 25942 frames with zero timestamp mismatches.

---

## 6. Train (ACT)

```bash
lerobot-train \
  --dataset.repo_id=zane/pick_place_block_clean \
  --dataset.root=./datasets/pick_place_block_clean \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=output/train/act_pick_place_block \
  --batch_size=8 \
  --steps=30000 \
  --policy.push_to_hub=false
```

- `video_backend=pyav` is the workaround for a broken/hard-to-link
  `torchcodec` in this environment; training then decodes the AV1 video with
  `pyav`, which is reliable here.
- With a 6 GB laptop GPU, batch 8 & 30k steps fit fine and converge for a
  single clean pick-place behavior.

---

## 7. Real-robot evaluation

### 7.1 Sanity run (autonomous, no recording)

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=output/train/act_pick_place_block/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --task="Pick up the candy and place it into the lid" \
  --duration=60 \
  --display_data=true
```

### 7.2 Statistical evaluation (record N rollouts)

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=output/train/act_pick_place_block/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.repo_id=zane/eval_pick_place_block \
  --dataset.root=./datasets/eval_pick_place_block \
  --dataset.single_task="Pick up the candy and place it into the lid" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true
```

Score success = episode reached goal state without intervention, count / N.