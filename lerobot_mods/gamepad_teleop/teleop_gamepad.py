# !/usr/bin/env python

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

import sys
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pygame_available, require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadJointsTeleopConfig, GamepadTeleopConfig

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn't reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass


class GamepadJointsTeleop(Teleoperator):
    """Analog gamepad teleoperator that drives individual joints by velocity.

    Each joystick axis controls one joint; the deflection magnitude sets the
    joint speed and the sign sets the direction. The remaining joints are
    controlled by holding shoulder buttons. This produces absolute ``{joint}.pos``
    targets (seeded from the current observation), so it works directly with
    :meth:`Robot.send_action` without extra processor steps, exactly like the
    keyboard joint teleoperator.
    """

    config_class = GamepadJointsTeleopConfig
    name = "gamepad_joints"

    def __init__(self, config: GamepadJointsTeleopConfig):
        require_package("pygame", extra="gamepad")
        super().__init__(config)
        self.config = config
        self.motor_names = list(
            dict.fromkeys(
                list(config.axis_joints.values())
                + [name.lstrip("+-") for name in config.button_joints.values()]
            )
        )
        self.joystick = None
        self._targets: dict[str, float] | None = None
        self._last_time: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.joystick is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for gamepad teleoperation.")

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect a USB gamepad and try again.")

        self.joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self.joystick.init()
        print(f"Initialized gamepad: {self.joystick.get_name()}")

        print("\nGamepad joint teleoperation ready:")
        for axis, joint in sorted(self.config.axis_joints.items()):
            direction = "inverted" if self.config.invert.get(joint, False) else "normal"
            print(f"  axis {axis:<2} -> {joint:<14} ({direction})")
        for button, spec in sorted(self.config.button_joints.items()):
            joint = spec.lstrip("+-")
            direction = "negative" if spec.startswith("-") else "positive"
            print(f"  button {button:<2} -> {joint:<14} ({direction})")
        print("Each joint is velocity-controlled: full deflection = full speed, release = hold.")

    def _sync_targets(self, obs: RobotObservation | None) -> None:
        if self._targets is not None or obs is None:
            return
        self._targets = {name: float(obs[f"{name}.pos"]) for name in self.motor_names}
        self._last_time = time.perf_counter()

    def _deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.config.deadzone else value

    @check_if_not_connected
    def get_action(self, obs: RobotObservation | None = None) -> RobotAction:
        self._sync_targets(obs)
        if self._targets is None:
            return {}

        # Pump pygame events so joystick state is fresh.
        pygame.event.pump()

        now = time.perf_counter()
        dt = 0.0 if self._last_time is None else (now - self._last_time)
        self._last_time = now

        # Axis-driven joints (velocity = deflection * max speed * direction).
        for axis, joint in self.config.axis_joints.items():
            try:
                value = self._deadzone(self.joystick.get_axis(axis))
            except pygame.error:
                value = 0.0
            sign = -1.0 if self.config.invert.get(joint, False) else 1.0
            speed = self.config.joint_speeds.get(joint, 30.0)
            self._targets[joint] += value * sign * speed * dt

        # Button-driven joints (constant speed while held).
        for button, spec in self.config.button_joints.items():
            try:
                pressed = self.joystick.get_button(button)
            except pygame.error:
                pressed = False
            if not pressed:
                continue
            direction = -1.0 if spec.startswith("-") else 1.0
            joint = spec.lstrip("+-")
            speed = self.config.joint_speeds.get(joint, 30.0)
            self._targets[joint] += direction * speed * dt

        # Clip per-joint targets.
        for joint, limits in self.config.joint_limits.items():
            if joint in self._targets:
                low, high = limits
                self._targets[joint] = max(low, min(high, self._targets[joint]))

        return {f"{name}.pos": float(self._targets[name]) for name in self.motor_names}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.joystick is not None:
            self.joystick.quit()
            self.joystick = None
        if pygame is not None and pygame.joystick.get_init():
            pygame.joystick.quit()