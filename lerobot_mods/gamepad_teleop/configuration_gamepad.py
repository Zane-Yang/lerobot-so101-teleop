#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("gamepad_joints")
@dataclass
class GamepadJointsTeleopConfig(TeleoperatorConfig):
    """Configuration for the gamepad joint-angle teleoperator.

    Drives a follower arm's individual joints from an analog gamepad. Each
    joystick axis controls one joint by velocity: a larger stick deflection
    moves that joint faster. The D-pad and shoulder buttons control the
    remaining joints. Targets are held when the sticks are released (deadzone).

    Attributes:
        joystick_index: Which connected pygame joystick to use.
        deadzone: Stick deflection below this magnitude is ignored (drift).
        invert: Per-joint sign inversion.
        joint_speeds: Per-joint max speed (degrees/sec for arm joints,
            gripper units/sec for the gripper).
        joint_limits: Optional per-joint ``[min, max]`` clipping for the
            commanded target.
        axis_joints: Joystick axis index -> joint name.
        button_joints: Button index -> joint direction string. Prefix '-' to
            move the joint in the negative direction.
    """

    joystick_index: int = 0
    deadzone: float = 0.15
    invert: dict[str, bool] = field(
        default_factory=lambda: {
            "shoulder_pan": False,
            "shoulder_lift": True,
            "wrist_flex": False,
            "elbow_flex": True,
            "wrist_roll": False,
        }
    )
    joint_speeds: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 60.0,
            "shoulder_lift": 60.0,
            "elbow_flex": 60.0,
            "wrist_flex": 60.0,
            "wrist_roll": 90.0,
            "gripper": 60.0,
        }
    )
    joint_limits: dict[str, list[float]] = field(default_factory=dict)
    # Joystick axis -> joint. The four physical analog axes (two sticks) map to
    # four joints; the remaining joints are driven by buttons (see below).
    axis_joints: dict[int, str] = field(
        default_factory=lambda: {
            0: "shoulder_pan",  # left stick X
            1: "shoulder_lift",  # left stick Y
            2: "elbow_flex",
            3: "wrist_roll",  # right stick X
        }
    )
    # Button -> joint direction while held. Prefix '-' inverts the direction.
    # For SO grippers, increasing gripper.pos closes the gripper.
    button_joints: dict[int, str] = field(
        default_factory=lambda: {
            5: "-wrist_flex",  # L1
            7: "+wrist_flex",  # L2
            4: "-gripper",  # R1 (open)
            6: "+gripper",  # R2 (close)
        }
    )
