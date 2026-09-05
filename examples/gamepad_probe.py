#!/usr/bin/env python
"""Print the axes and buttons of the first connected pygame joystick.

Use this to discover the axis/button indices of your gamepad before customizing
``--teleop.axis_joints`` / ``--teleop.button_joints`` for the ``gamepad_joints``
teleoperator.

Run:
    conda activate lerobot
    python examples/gamepad_probe.py

Move each stick and press each button; the values/indices update live. Press
Ctrl-C to exit.
"""

import time

import pygame


def main() -> None:
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Connect a USB gamepad and try again.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Name: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")
    print("\nMove sticks and press buttons. Ctrl-C to exit.\n")

    try:
        while True:
            pygame.event.pump()

            axes = [f"{joystick.get_axis(i):+.2f}" for i in range(joystick.get_numaxes())]
            buttons = [str(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]

            print("axes:   " + "  ".join(f"[{i}]{v}" for i, v in enumerate(axes)))
            print("buttons:" + "  ".join(f"[{i}]{v}" for i, v in enumerate(buttons)))
            print("-" * 78)

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    main()