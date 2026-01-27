#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run VR hand tracking teleoperation with Cloud XR environment."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="VR hand tracking teleoperation for Cloud XR data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Device for interacting with environment. Options: keyboard, spacemouse, gamepad, handtracking",
)
parser.add_argument("--task", type=str, default="Isaac-Franka-JengaBowl-CloudXR-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# Enable XR for hand tracking
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_cloudxr.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """
    Run VR hand tracking teleoperation with Cloud XR environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    # modify configuration for teleoperation
    env_cfg.terminations.time_out = None  # Disable timeout for teleoperation

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

        # Also remove camera observation groups (not handled by remove_camera_configs)
        if hasattr(env_cfg, "observations"):
            if hasattr(env_cfg.observations, "wrist_camera"):
                env_cfg.observations.wrist_camera = None
                print("[INFO] Removed wrist_camera observation group for XR mode")
            if hasattr(env_cfg.observations, "external_camera"):
                env_cfg.observations.external_camera = None
                print("[INFO] Removed external_camera observation group for XR mode")

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # Create teleoperation interface
        sensitivity = args_cli.sensitivity
        teleop_interface = None

        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, {}
            )
        else:
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                print(f"ERROR: Unsupported teleop device: {args_cli.teleop_device}")
                print("Supported devices: keyboard, spacemouse, gamepad")
                print("For handtracking, use IsaacLab's built-in teleop script:")
                print("  /home/tshiamo/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py")
                env.close()
                simulation_app.close()
                return

        print("\n" + "="*80)
        print("Cloud XR Data Collection - VR Hand Tracking")
        print("="*80)
        print(f"Task: {args_cli.task}")
        print(f"Number of environments: {args_cli.num_envs}")
        print(f"Teleop device: {args_cli.teleop_device}")
        print(f"Sensitivity: {args_cli.sensitivity}")
        print("="*80 + "\n")

        # reset environment
        env.reset()
        teleop_interface.reset()

        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # get teleop command
                action = teleop_interface.advance()

                # expand to batch dimension
                actions = action.repeat(env.num_envs, 1)

                # apply actions
                obs, rewards, terminated, truncated, info = env.step(actions)

                # check if simulator is stopped
                if env.unwrapped.sim.is_stopped():
                    break

                # reset on termination
                if terminated.any() or truncated.any():
                    env.reset()
                    teleop_interface.reset()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the simulator
        env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close sim app
        simulation_app.close()
