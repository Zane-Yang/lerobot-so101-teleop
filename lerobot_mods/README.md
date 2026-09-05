# `lerobot_mods` — self-authored delta on top of upstream LeRobot

These folders contain the **small set of files we added or hand-tuned** on top of
Hugging Face LeRobot to make leader-less keyboard/gamepad teleoperation and the
dataset inspection tools work. They are organized so a reviewer can see, at a
glance, exactly what is custom work vs. upstream boilerplate.

| Folder | What it is |
|--------|-----------|
| `gamepad_teleop/` | `gamepad_joints` teleoperator: config + class + package exports |
| `keyboard_teleop/` | `keyboard_joints` teleoperator: config + class + package exports |

## How these integrate

LeRobot discovers teleoperators via `@TeleoperatorConfig.register_subclass("<name>")`
decorators executed at import time, and instantiates them through the factory in
`lerobot/teleoperators/utils.py`. To apply:

1. Copy both folders into the matching upstream directory, i.e.:

   ```
   lerobot_mods/gamepad_teleop/configuration_gamepad.py  -> lerobot/teleoperators/gamepad/configuration_gamepad.py
   lerobot_mods/gamepad_teleop/teleop_gamepad.py         -> lerobot/teleoperators/gamepad/teleop_gamepad.py
   lerobot_mods/gamepad_teleop/__init__.py               -> lerobot/teleoperators/gamepad/__init__.py
   ```

   and analogously for `keyboard_teleop`.

2. Make sure the module is imported before CLI parsing runs. Both `lerobot-record`
   and `lerobot-teleoperate` do this. If `--teleop.type=gamepad_joints` is not
   listed in `--help`, import `lerobot.teleoperators.gamepad` (or `keyboard`)
   in the entry script.

3. The loop calls `teleop.get_action(obs)`; for our teleoperators the current
   observation is used to jump-free seed the first absolute joint target. The
   scripts/entry points from this repo already carry this call.

The transport, dataset, training and rollout remain upstream LeRobot — we never
forked the learning stack.