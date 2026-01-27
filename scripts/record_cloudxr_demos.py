#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record VR hand tracking demonstrations for Cloud XR data collection.

This script records demonstrations using VR hand tracking for the Franka JengaBowl task.
The recorded demonstrations are stored as episodes in an HDF5 file.

Usage:
    # Record with VR hand tracking
    ./isaaclab.sh -p scripts/record_cloudxr_demos.py \
        --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
        --teleop_device handtracking \
        --dataset_file ./datasets/jengabowl_demos.hdf5 \
        --num_demos 50

    # Record with keyboard (for testing)
    ./isaaclab.sh -p scripts/record_cloudxr_demos.py \
        --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
        --teleop_device keyboard \
        --dataset_file ./datasets/jengabowl_demos.hdf5 \
        --num_demos 10
"""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Cloud XR data collection.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Franka-JengaBowl-CloudXR-Single-v0",
    help="Name of the task."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Device for interacting with environment (handtracking, keyboard, spacemouse, gamepad)."
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/jengabowl_demos.hdf5",
    help="File path to export recorded demos."
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=30,
    help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful.",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Prepare app launcher arguments
app_launcher_args = vars(args_cli)

# Enable XR for hand tracking
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# Launch the simulator
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import omni.log

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_cloudxr.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Sleep at the specified rate."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations."""
    output_directory = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    os.makedirs(output_directory, exist_ok=True)
    file_name = os.path.basename(args_cli.dataset_file)
    return output_directory, file_name


def main():
    """Record demonstrations with VR hand tracking."""

    # Setup output directories
    output_directory, file_name = setup_output_directories()

    print("\n" + "="*80)
    print("Cloud XR Data Collection - Recording Demonstrations")
    print("="*80)
    print(f"Task: {args_cli.task}")
    print(f"Teleop device: {args_cli.teleop_device}")
    print(f"Dataset file: {args_cli.dataset_file}")
    print(f"Number of demos: {args_cli.num_demos if args_cli.num_demos > 0 else 'Infinite'}")
    print(f"Step rate: {args_cli.step_hz} Hz")
    print(f"Success steps: {args_cli.num_success_steps}")
    print("="*80 + "\n")

    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env_cfg.env_name = args_cli.task

    # Disable timeout for teleoperation
    env_cfg.terminations.time_out = None

    # Enable XR configuration
    if args_cli.xr:
        # NOTE: Commented out to preserve cameras for multi-modal data capture
        # env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

        # Enable cameras explicitly for multi-modal recording
        if hasattr(env_cfg, 'sim'):
            env_cfg.sim.enable_cameras = True

    # Configure data recording
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_directory
    env_cfg.recorders.dataset_filename = file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Create teleoperation interface
    sensitivity = 1.0
    teleop_interface = None

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        teleop_interface = create_teleop_device(
            args_cli.teleop_device, env_cfg.teleop_devices.devices, {}
        )
    else:
        # Create fallback teleop device
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg, Se3Gamepad, Se3GamepadCfg

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
            print("For handtracking, use IsaacLab's built-in recording script:")
            print("  /home/tshiamo/IsaacLab/scripts/tools/record_demos.py")
            env.close()
            simulation_app.close()
            return

    # Rate limiter for consistent stepping
    rate_limiter = RateLimiter(args_cli.step_hz)

    # Recording statistics
    demo_count = 0
    success_step_count = 0
    total_episodes = 0
    successful_episodes = 0

    # Reset environment
    env.reset()
    teleop_interface.reset()

    print("\n" + "="*80)
    print("Recording started!")
    print("="*80)
    print("Controls:")
    if "handtracking" in args_cli.teleop_device.lower():
        print("  - Use VR hand tracking to control the robot")
        print("  - Pick up the block and place it in the bowl")
        print("  - Release the gripper to complete the task")
    elif "keyboard" in args_cli.teleop_device.lower():
        print("  - Arrow keys: Move end effector")
        print("  - Page Up/Down: Move up/down")
        print("  - 'G': Toggle gripper")
        print("  - 'R': Reset environment")
        print("  - 'Q' or ESC: Quit")
    print("\nPress Ctrl+C to stop recording")
    print("="*80 + "\n")

    try:
        while simulation_app.is_running():
            # Check if we've recorded enough demos
            if args_cli.num_demos > 0 and demo_count >= args_cli.num_demos:
                print(f"\nCompleted {args_cli.num_demos} demonstrations!")
                break

            # Get teleoperation command
            with torch.inference_mode():
                action = teleop_interface.advance()

                # Expand to batch dimension
                actions = action.repeat(env.num_envs, 1)

                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)

                # Check for success
                if "success" in info and info["success"].any():
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        print(f"\n{'='*80}")
                        print(f"SUCCESS! Demo {demo_count + 1} completed")
                        print(f"{'='*80}\n")
                        demo_count += 1
                        successful_episodes += 1
                        success_step_count = 0
                        env.reset()
                        teleop_interface.reset()
                else:
                    success_step_count = 0

                # Check if simulator is stopped
                if env.unwrapped.sim.is_stopped():
                    break

                # Reset on termination/truncation
                if terminated.any() or truncated.any():
                    total_episodes += 1
                    if not (terminated.any() and "success" in info and info["success"].any()):
                        print(f"Episode {total_episodes} ended (not successful)")
                    env.reset()
                    teleop_interface.reset()
                    success_step_count = 0

                # Rate limiting
                rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")

    finally:
        # Print final statistics
        print("\n" + "="*80)
        print("Recording Statistics")
        print("="*80)
        print(f"Total episodes: {total_episodes}")
        print(f"Successful demos: {demo_count}")
        print(f"Success rate: {(demo_count / max(total_episodes, 1)) * 100:.1f}%")
        print(f"Dataset saved to: {args_cli.dataset_file}")
        print("="*80 + "\n")

        # Close environment
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
