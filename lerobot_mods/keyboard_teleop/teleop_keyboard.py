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

import logging
import time
from queue import Queue
from typing import Any

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener, pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardJointsTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }


class KeyboardJointsTeleop(Teleoperator):
    """
    Keyboard teleoperator that drives individual joints by fixed increments.

    Intended for a follower-only setup (no leader arm): each joint is moved in its
    native units (degrees for the SO body joints, 0-100 for the gripper) by tapping
    its configured positive/negative keys. The commanded absolute joint target is
    held between key presses, so releasing a key stops movement.

    Unlike :class:`KeyboardTeleop` (which needs pynput on X11/macOS/Windows), this
    class consumes discrete key taps through :func:`create_key_listener`, so it also
    works over Wayland terminals and SSH. The live measured joints are used to seed
    the held target, so the teleoperate/record loops pass the current observation to
    :meth:`get_action`.
    """

    config_class = KeyboardJointsTeleopConfig
    name = "keyboard_joints"

    def __init__(self, config: KeyboardJointsTeleopConfig):
        super().__init__(config)
        self.config = config
        self.motor_names = list(config.joint_keys)
        self._targets: dict[str, float] | None = None
        self._pending_deltas: dict[str, float] = {}
        self._key_to_joint_sign: dict[str, tuple[str, int]] = {}
        self.listener = None
        self.recording_events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.listener is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        self._key_to_joint_sign = {}
        for joint, keys in self.config.joint_keys.items():
            plus_key = keys.get("+")
            minus_key = keys.get("-")
            if plus_key:
                self._key_to_joint_sign[plus_key] = (joint, 1)
            if minus_key:
                self._key_to_joint_sign[minus_key] = (joint, -1)

        def on_key(name: str) -> None:
            key = name.lower() if isinstance(name, str) else name
            # Recording control keys shared between teleoperation and lerobot-record.
            if key in ("right", "n"):
                apply_recording_control("right", self.recording_events)
            elif key in ("left", "r"):
                apply_recording_control("left", self.recording_events)
            elif key in ("esc", "q"):
                apply_recording_control("esc", self.recording_events)

            joint, sign = self._key_to_joint_sign.get(key, (None, 0))
            if joint is None:
                return
            step = self.config.step_sizes.get(joint, self.config.default_step)
            self._pending_deltas[joint] = self._pending_deltas.get(joint, 0.0) + sign * step

        help_text = "; ".join(
            f"{joint}: +={keys.get('+', '')} -={keys.get('-', '')}"
            for joint, keys in self.config.joint_keys.items()
        )
        self.listener = create_key_listener(on_key, controls_help=help_text)
        if self.listener is None:
            raise RuntimeError(
                "Keyboard joint teleoperation requires an interactive terminal or a display "
                "session (X11/Wayland). Run from a real terminal, not a detached/headless process."
            )

        print("\nKeyboard joint teleoperation ready:")
        for joint, keys in self.config.joint_keys.items():
            step = self.config.step_sizes.get(joint, self.config.default_step)
            print(f"  {joint:<14} {keys.get('+', ''):>4} / {keys.get('-', ''):<4}  step={step}")
        print("Each tap moves one joint by its step. Other keys (Esc/q/arrows) are handled by the loop.")

    def _sync_targets(self, obs: RobotObservation | None) -> None:
        if self._targets is not None or obs is None:
            return
        self._targets = {name: float(obs[f"{name}.pos"]) for name in self.motor_names}

    @check_if_not_connected
    def get_action(self, obs: RobotObservation | None = None) -> RobotAction:
        self._sync_targets(obs)

        if self._targets is None:
            return {}

        # Consume key taps accumulated since the last frame.
        deltas = self._pending_deltas
        self._pending_deltas = {}
        for joint, delta in deltas.items():
            self._targets[joint] = self._targets.get(joint, 0.0) + delta

        for joint, limits in self.config.joint_limits.items():
            if joint in self._targets:
                low, high = limits
                self._targets[joint] = max(low, min(high, self._targets[joint]))

        return {f"{name}.pos": float(self._targets[name]) for name in self.motor_names}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
