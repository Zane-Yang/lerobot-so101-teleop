#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Configuration for keyboard teleoperators."""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""

    # TODO(Steven): Consider setting in here the keys that we want to capture/listen


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """Configuration for keyboard end-effector teleoperator.

    Used for controlling robot end-effectors with keyboard inputs.

    Attributes:
        use_gripper: Whether to include gripper control in actions
    """

    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_joints")
@dataclass
class KeyboardJointsTeleopConfig(TeleoperatorConfig):
    """Configuration for the keyboard joint-angle teleoperator.

    Drives a follower arm one joint at a time from the keyboard. Each joint has a
    positive and a negative key; each key press moves that joint by ``step_sizes``
    (in the robot's native units — degrees for SO joints, 0-100 for gripper).
    The target is held between key presses.

    Attributes:
        step_sizes: Per-joint step size in the robot's native units. Missing joints
            fall back to ``default_step``.
        default_step: Step size used when a joint is absent from ``step_sizes``.
        joint_limits: Optional per-joint ``[min, max]`` clipping for the commanded
            target. Missing joints are not clipped (the motor's own position limits
            still apply on the hardware).
    """

    step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 2.0,
            "shoulder_lift": 2.0,
            "elbow_flex": 2.0,
            "wrist_flex": 2.0,
            "wrist_roll": 2.0,
            "gripper": 4.0,
        }
    )
    default_step: float = 2.0
    joint_limits: dict[str, list[float]] = field(default_factory=dict)
    # Per-joint key bindings. Each value maps "+" to the key that increases the
    # joint value and "-" to the key that decreases it.
    joint_keys: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "shoulder_pan": {"+": "d", "-": "a"},
            "shoulder_lift": {"+": "w", "-": "s"},
            "elbow_flex": {"+": "l", "-": "j"},
            "wrist_flex": {"+": "i", "-": "k"},
            "wrist_roll": {"+": "o", "-": "u"},
            "gripper": {"+": "m", "-": "b"},
        }
    )


@TeleoperatorConfig.register_subclass("keyboard_rover")
@dataclass
class KeyboardRoverTeleopConfig(TeleoperatorConfig):
    """Configuration for keyboard rover teleoperator.

    Used for controlling mobile robots like EarthRover Mini Plus with WASD controls.

    Attributes:
        linear_speed: Default linear velocity magnitude (-1 to 1 range for SDK robots)
        angular_speed: Default angular velocity magnitude (-1 to 1 range for SDK robots)
        speed_increment: Amount to increase/decrease speed with +/- keys
        turn_assist_ratio: Forward motion multiplier when turning with A/D keys (0.0-1.0)
        angular_speed_ratio: Ratio of angular to linear speed for synchronized adjustments
        min_linear_speed: Minimum linear speed when decreasing (prevents zero speed)
        min_angular_speed: Minimum angular speed when decreasing (prevents zero speed)
    """

    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    turn_assist_ratio: float = 0.3
    angular_speed_ratio: float = 0.6
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.05
