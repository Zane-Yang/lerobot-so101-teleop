# Architecture & Design Notes

This document describes the target system, the two custom teleoperation modes,
and how they plug into the upstream LeRobot control loop. It is the “how it
works” companion to the README.

---

## 1. Target System

### 1.1 Hardware (used for development)

| Component | Detail |
|-----------|--------|
| Arm | SO-101 follower-style 6-DoF arm |
| Joint actuators | 6 × Feetech STS3215 serial-bus servos (IDs 1–6) |
| Motor controller | Serial / USB motor bus (e.g. `/dev/ttyACM0`) |
| Camera | Single webcam, `front` view, 640×480 @ 30 (OpenCV backend) |
| Gamepad | USB analog gamepad (2 analog sticks + buttons) |
| Compute | Single laptop: RTX 3060 Laptop 6 GB VRAM, 16 GB RAM |

> The arm is used in “follower” configuration on the bus; we do **not** use a
> leader arm. Therefore both added teleoperators are **leader-less**.

### 1.2 Software stack

- `lerobot` (upstream repo, sibling clone / direct source checkout)
- `Python 3.12`, `PyTorch` + CUDA
- `feetech` motor SDK
- `pyav` for robust video decode/encode
- Optional `rerun`/`foxglove` for visualization

---

## 2. How the upstream loop works (in one picture)

```
loop:
  obs = robot.get_observation()            # 6 joint pos + camera frames
  act = teleop.get_action(obs)             # teleoperator input
  robot.send_action(act)                   # sync write Goal_Position
```

- Teleoperator outputs are validated against `robot.action_features` and are
  expected to be **`.pos`** entries for joint-level control (SO-101).
- `robot.send_action` ingests these and writes goal positions to the serial bus.

---

## 3. Why a leader-less teleoperator?

A classic LeRobot SO-100/101 setup uses a **leader arm** as “human-in-the-loop”
teleoperator mirroring the follower. That requires a second arm and puts the
operator physically on hardware. We removed that dependency:

1. **No extra arm needed** — only one follower arm + one USB gamepad/keyboard.
2. **Lower entry cost** — operator needs no second robot to practice gesture
   mirroring.
3. **Direct joint control** — joints are commanded in degrees (SO arm native
   units), keeping the same action space used by ACT policies.

The result is still a full “teleoperation → dataset → policy” pipeline, because
the output of our teleoperators is exactly the `.pos` action structure that
`lerobot-record` stores.

---

## 4. Mode A: `gamepad_joints` (analog velocity mapping)

### 4.1 Idea

Map the two analog sticks to four joints (one stick X/Y per two joints), and
use shoulder buttons for a 5th joint and the gripper. A **deadzone** ignores
stick noise; otherwise deflection amplitude → joint velocity.

### 4.2 Default mapping (config overridable)

| Controller input | Joint | Direction / note |
|------------------|-------|------------------|
| Left stick X | `shoulder_pan` | left / right |
| Left stick Y | `shoulder_lift` | up / down (often software-inverted) |
| Right stick X | `wrist_roll` | rotate |
| Right stick Y | `elbow_flex` | up / down |
| L1 / R1 | `wrist_flex` | open / close wrist |
| L2 / R2 | `gripper` | open (L2) / close (R2) |

> Because gamepads differ in axis/button numbering, we ship
> `examples/gamepad_probe.py` to print the live IDs the host system reports, so
> users can remap config keys to their exact hardware.

### 4.3 Implementation notes

- Config: `KeyboardEndEffector`-style dataclass registers as
  `GamepadJointsTeleopConfig` under subclass name `gamepad_joints`.
- Teleoperator class `GamepadJointsTeleop`:
  - `get_action(obs)` seeds targets from current measured joints (idempotently)
    so the first command is **jump-free**.
  - Each frame: read stick axes & buttons via `pygame`, integrate
    `joint_target += axis * speed * dt`.
  - Per-joint optional clip `[min, max]` is applied to guard against runaway.
  - Returns `{f"{motor}.pos": value}` dict — identical shape to LeRobot
    `action_features`, hence compatible with `record_loop` / `teleop_loop`.
- No new dependency on upstream core: it only adds a new teleoperator subclass,
  so the rest of LeRobot is untouched.

### 4.4 Safety features

- `deadzone` prevents drift when sticks are not perfectly centered.
- Optional `joint_limits` prevents commanding beyond a user-defined joint bound.
- A velocity model means the operator can correct course in real time — far
  less “jumpy” than absolute-position snapping from key steps.

---

## 5. Mode B: `keyboard_joints` (incremental step control)

### 5.1 Idea

“One key press = one joint by one step.” Provides a no-gamepad fallback, and a
precise way to nudge single joints by fixed increments (e.g. degree-granular
servo target tuning).

### 5.2 Default mapping

| Joint | + | − |
|-------|---|---|
| `shoulder_pan` | `d` | `a` |
| `shoulder_lift` | `w` | `s` |
| `elbow_flex` | `l` | `j` |
| `wrist_flex` | `i` | `k` |
| `wrist_roll` | `o` | `u` |
| `gripper` | `m` | `b` |

### 5.3 Implementation notes

- Listener supports both **pynput**-style (GUI/desktop) and **terminal/TTY**
  capture so SSH sessions remain usable.
- All key-to-joint mappings and step sizes are exposed in the config, letting a
  user adapt to national keyboard layouts.

---

## 6. Integration points touched in upstream (minimal & clean)

We deliberately keep our own code out of upstream-heavy paths. In a real
upstream checkout we only needed these **small, explicit** hooks:

1. `teleoperators/keyboard/` & `teleoperators/gamepad/` — added config &
   implementation files for the two new subclasses.
2. `teleoperators/utils.py` — factory dispatched the new subclass names
   (`keyboard_joints`, `gamepad_joints`).
3. `scripts/lerobot_teleoperate.py` / `scripts/lerobot_record.py` —
   `get_action` was already the only call used by the loop; we typed the new
   teleoperator to take the current observation, but left the rest of the loop
   untouched.
4. `examples/` and `dataset_tools` fixes — tools for episode inspection and the
   timestamp integrity fix (see TROUBLESHOOTING).

This means the rest of the project can be kept in sync with upstream by
updating only these hooks — exactly the kind of scoped patch a resume should
highlight.

---

## 7. Dataset & training (short)

- Dataset features: `observation.images.front` (640×480, 30fps), `observation.state`, `action` (6-dim).
- ACT config: ResNet18 backbone, chunk size 100, VAE latent 32, batch 8, 30k step.
- Trained locally on one RTX 3060 6 GB; inference executed by `lerobot-rollout` on the real arm.

See `docs/PIPELINE.md` for exact flags and file layout of a train run.