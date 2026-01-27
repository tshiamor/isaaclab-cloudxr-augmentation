#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record VR hand tracking demonstrations WITH CAMERAS for visuomotor learning.

This script is specifically designed to keep camera observations while using VR hand tracking.
It does NOT remove cameras like the standard XR teleoperation scripts.

Usage:
    # Record with VR hand tracking and cameras
    ./isaaclab.sh -p scripts/record_cloudxr_with_cameras.py \
        --task Isaac-Franka-JengaBowl-CloudXR-Single-Visuomotor-v0 \
        --teleop_device handtracking \
        --dataset_file ./datasets/jengabowl_visuomotor_demos.hdf5 \
        --num_demos 50
"""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations with cameras for visuomotor learning.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Franka-JengaBowl-CloudXR-Single-Visuomotor-v0",
    help="Name of the task (must be visuomotor version with cameras)."
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
    default="./datasets/jengabowl_visuomotor_demos.hdf5",
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
    default=5,
    help="Number of continuous steps with task success for concluding a demo as successful. Keep gripper open!",
)
parser.add_argument(
    "--auto_start",
    action="store_true",
    help="Automatically start recording without waiting for VR button press. Useful when VR buttons aren't working.",
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

# NOTE: We do NOT import remove_camera_configs since we want to keep cameras!
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import DatasetExportMode

import isaaclab_cloudxr.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_cloudxr.tasks.manager_based.isaaclab_cloudxr.mdp.terminations import block_in_bowl
from isaaclab_cloudxr.tasks.manager_based.isaaclab_cloudxr.mdp.camera_recorders_cfg import VisuomotorRecorderManagerCfg


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
    """Record demonstrations with VR hand tracking and cameras."""

    # Setup output directories
    output_directory, file_name = setup_output_directories()

    print("\n" + "="*80)
    print("Cloud XR Data Collection - WITH CAMERAS (Visuomotor Learning)")
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

    # Keep timeout enabled for proper episode termination
    # env_cfg.terminations.time_out = None  # Commented out to enable timeout

    # **IMPORTANT: We do NOT remove camera configs for visuomotor learning!**
    # Cameras will remain active and record RGB/depth data
    if args_cli.xr:
        # Only set antialiasing, keep cameras
        env_cfg.sim.render.antialiasing_mode = "DLSS"
        print("[INFO] XR enabled with cameras - visuomotor mode active")
        print("[INFO] Cameras will record RGB and depth images")
        print("[INFO] This may reduce performance compared to state-only recording\n")

    # Configure data recording WITH CAMERAS
    # Use VisuomotorRecorderManagerCfg instead of ActionStateRecorderManagerCfg
    # This will record camera RGB + Depth images in addition to states/actions
    env_cfg.recorders = VisuomotorRecorderManagerCfg()
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
            print("Supported devices: keyboard, spacemouse, gamepad, handtracking")
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

    # Recording state control - For XR, wait for user to press START (unless auto_start is enabled)
    if args_cli.auto_start:
        running_recording_instance = True  # Auto-start enabled: begin recording immediately
        print("[INFO] Auto-start enabled - recording will begin immediately")
    else:
        running_recording_instance = not args_cli.xr  # False for XR (wait for start), True for keyboard
    should_reset_recording_instance = False
    callbacks_enabled = False  # Prevent auto-start during initialization

    # Store initial pose to hold when paused
    paused_target_pose = None

    # Track when state changes to capture pose in main loop
    request_state_change = {"start": False, "stop": False}

    # Callback functions for recording control
    def start_recording_instance():
        nonlocal running_recording_instance
        print(f"\n[DEBUG CALLBACK] START callback triggered - callbacks_enabled={callbacks_enabled}")
        if not callbacks_enabled:
            print("[DEBUG CALLBACK] Ignoring START - callbacks not enabled yet")
            return  # Ignore callbacks during initialization
        print(f"[DEBUG CALLBACK] START processing - running was {running_recording_instance}")
        request_state_change["start"] = True
        running_recording_instance = True
        print("\n[INFO] Recording STARTED - Robot is now active!")
        print(f"[DEBUG CALLBACK] running_recording_instance is now {running_recording_instance}")
        print("[INFO] Perform the task: Pick block and place in bowl")

    def stop_recording_instance():
        nonlocal running_recording_instance
        print(f"\n[DEBUG CALLBACK] STOP callback triggered - callbacks_enabled={callbacks_enabled}")
        if not callbacks_enabled:
            print("[DEBUG CALLBACK] Ignoring STOP - callbacks not enabled yet")
            return  # Ignore callbacks during initialization
        print(f"[DEBUG CALLBACK] STOP processing - running was {running_recording_instance}")
        request_state_change["stop"] = True
        running_recording_instance = False
        print("\n[INFO] Recording PAUSED - Robot is frozen")
        print(f"[DEBUG CALLBACK] running_recording_instance is now {running_recording_instance}")

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("\n[INFO] Environment reset requested")

    # Add callbacks to teleop interface (if it supports them)
    if hasattr(teleop_interface, 'add_callback'):
        # For VR: START/STOP/RESET callbacks
        teleop_interface.add_callback("START", start_recording_instance)
        teleop_interface.add_callback("STOP", stop_recording_instance)
        teleop_interface.add_callback("RESET", reset_recording_instance)
        # For keyboard: R key for reset
        teleop_interface.add_callback("R", reset_recording_instance)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    # Step the simulation a few times to initialize properly
    # This ensures IK controller and physics are ready before we start
    print("[INFO] Initializing environment...")
    init_action = None
    for _ in range(10):
        init_action = teleop_interface.advance()
        init_actions = init_action.repeat(env.num_envs, 1)
        # Step with actual VR actions to initialize IK controller
        env.step(init_actions)
        env.sim.render()

    print("[INFO] Environment initialized and ready")

    # Capture initial pose to hold while paused (use last init action)
    paused_target_pose = init_actions.clone() if init_action is not None else None
    if paused_target_pose is not None:
        print(f"[DEBUG] Initial hold pose captured: {paused_target_pose[0][:3].cpu().numpy()}")

    # Enable callbacks now that initialization is complete
    callbacks_enabled = True

    print("\n" + "="*80)
    if args_cli.xr:
        if args_cli.auto_start:
            print("VR Recording Ready - AUTO-START ENABLED")
            print("Robot is ACTIVE - Recording will begin immediately!")
        else:
            print("VR Recording Ready - WAITING FOR START")
            print("Robot is FROZEN - Press START button in VR to activate")
    else:
        print("Recording started WITH CAMERAS!")
    print("="*80)
    print("Camera observations:")
    print("  - Wrist RGB + Depth (480x640)")
    print("  - External RGB + Depth (480x640)")
    print("\nControls:")
    if "handtracking" in args_cli.teleop_device.lower():
        print("  - Use VR hand tracking to control the robot")
        print("  - Pick up the block and place it in the bowl")
        print("  - Release the gripper to complete the task")
        if not args_cli.auto_start:
            print("\n  [VR CONTROLLERS]:")
            print("  - Press MENU/START button to BEGIN recording")
            print("  - Press MENU/STOP button to PAUSE recording")
            print("  - Robot will remain LOCKED until you press START")
        else:
            print("\n  [AUTO-START MODE]:")
            print("  - Robot is active immediately - no button press needed!")
            print("  - VR button controls are disabled")
    elif "keyboard" in args_cli.teleop_device.lower():
        print("  - Arrow keys: Move end effector")
        print("  - Page Up/Down: Move up/down")
        print("  - 'G': Toggle gripper")
        print("  - 'R': Reset environment")
        print("  - 'Q' or ESC: Quit")
    print("\nPress Ctrl+C to stop recording")
    print("="*80 + "\n")

    # Debug counter
    debug_step_counter = 0

    try:
        while simulation_app.is_running():
            # Check if we've recorded enough demos (count all episodes, not just successful)
            if args_cli.num_demos > 0 and total_episodes >= args_cli.num_demos:
                print(f"\n{'='*80}")
                print(f"🎉 DATA COLLECTION COMPLETE!")
                print(f"{'='*80}")
                print(f"Completed {args_cli.num_demos} episodes ({demo_count} successful)")
                print(f"{'='*80}\n")
                break

            # Get teleoperation command
            with torch.inference_mode():
                action = teleop_interface.advance()

                # Expand to batch dimension
                actions = action.repeat(env.num_envs, 1)

                debug_step_counter += 1

                # Capture hold pose when state changes to PAUSED
                if request_state_change["stop"]:
                    request_state_change["stop"] = False
                    # Use the current action as the hold pose
                    paused_target_pose = actions.clone()
                    print(f"[DEBUG] Captured hold pose: {paused_target_pose[0][:3].cpu().numpy()}")

                # Clear start flag (no action needed, just acknowledgment)
                if request_state_change["start"]:
                    request_state_change["start"] = False
                    print(f"[DEBUG] START pressed - running_recording_instance = {running_recording_instance}")
                    print(f"[DEBUG] Current VR action: {actions[0][:3].cpu().numpy()}")

                # ALWAYS step environment to keep IK controller active
                # When paused, send "hold position" command instead of VR actions
                if not running_recording_instance and paused_target_pose is not None:
                    # When paused, command robot to hold the paused pose
                    actions = paused_target_pose
                    if debug_step_counter % 30 == 0:  # Log every second at 30Hz
                        print(f"[DEBUG] Step {debug_step_counter}: PAUSED - using hold pose: {actions[0][:3].cpu().numpy()}")
                else:
                    if debug_step_counter % 30 == 0:  # Log every second at 30Hz
                        print(f"[DEBUG] Step {debug_step_counter}: ACTIVE - running={running_recording_instance}, VR action: {actions[0][:3].cpu().numpy()}")

                # Step environment (always step to keep IK active)
                obs, rewards, terminated, truncated, info = env.step(actions)

                # Only process results when recording is active
                if running_recording_instance:
                    # Manually check for success (since it's not a termination anymore)
                    success_check = block_in_bowl(
                        env,
                        xy_threshold=0.15,
                        min_height=-0.02,
                        max_height=0.10,
                        check_gripper=True,
                        atol=0.020,
                        rtol=0.05,
                    )

                    # Check for success
                    if success_check.any():
                        success_step_count += 1
                        # Show progress when approaching success
                        if success_step_count == 1:
                            print(f"\n[SUCCESS DETECTED] Keep gripper OPEN! ({success_step_count}/{args_cli.num_success_steps} steps)")
                        elif success_step_count % 5 == 0:
                            print(f"[SUCCESS] Holding... ({success_step_count}/{args_cli.num_success_steps} steps)")

                        if success_step_count >= args_cli.num_success_steps:
                            demo_count += 1
                            successful_episodes += 1
                            total_episodes += 1  # Count all episodes, not just failures

                            print(f"\n{'='*80}")
                            print(f"🎉 SUCCESS! Episode {total_episodes}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'} completed!")
                            print(f"{'='*80}")
                            print(f"Block successfully placed in bowl!")
                            print(f"Successful demos: {demo_count} / Total episodes: {total_episodes}")
                            print(f"{'='*80}\n")

                            # Hold for 3 seconds to let the user see the success
                            print(f"[INFO] Holding for 3 seconds before reset...")
                            import time
                            hold_start_time = time.time()
                            while time.time() - hold_start_time < 3.0:
                                # Continue stepping the environment to maintain physics
                                hold_action = teleop_interface.advance()
                                hold_actions = hold_action.repeat(env.num_envs, 1)
                                env.step(hold_actions)
                                env.sim.render()
                                time.sleep(0.033)  # ~30Hz

                            print(f"[INFO] Exporting demonstration to dataset...")
                            # **CRITICAL**: Proper sequence to export episode to HDF5
                            if hasattr(env, 'recorder_manager') and env.recorder_manager is not None:
                                # Step 1: Record the final state before reset
                                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)

                                # Step 2: Mark this episode as successful
                                env.recorder_manager.set_success_to_episodes(
                                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                )

                                # Step 3: Export the episode to HDF5
                                env.recorder_manager.export_episodes(env_ids=[0])
                                print(f"[INFO] ✅ Demo {demo_count} exported to HDF5")
                            else:
                                print(f"[WARNING] ⚠️  No recorder_manager found - demo NOT saved!")

                            print(f"[INFO] Resetting environment to home position...")
                            success_step_count = 0
                            env.reset()
                            teleop_interface.reset()

                            # Give environment time to settle at home position
                            for _ in range(5):
                                settle_action = teleop_interface.advance()
                                settle_actions = settle_action.repeat(env.num_envs, 1)
                                env.step(settle_actions)
                                env.sim.render()

                            # Recapture pose after reset (use last settle action)
                            paused_target_pose = settle_actions.clone()

                            print(f"[INFO] Arm returned to home position")
                            print(f"[INFO] Bowl and block positions randomized")
                            if total_episodes < args_cli.num_demos or args_cli.num_demos == 0:
                                print(f"[INFO] Ready for episode {total_episodes + 1}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'}")
                            print()
                    else:
                        success_step_count = 0

                    # Reset on termination/truncation (but not on success, already handled above)
                    if (terminated.any() or truncated.any()) and success_step_count == 0:
                        total_episodes += 1

                        print(f"\n{'='*80}")
                        print(f"⚠️  Episode {total_episodes}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'} failed")
                        print(f"{'='*80}")

                        # Check reason for failure
                        if truncated.any():
                            print(f"Reason: Timeout (episode too long)")
                        elif terminated.any():
                            print(f"Reason: Block dropped below minimum height")

                        print(f"Successful demos: {demo_count} / Total episodes: {total_episodes}")
                        print(f"Resetting environment to home position...")
                        print(f"{'='*80}\n")

                        env.reset()
                        teleop_interface.reset()
                        success_step_count = 0

                        # Give environment time to settle at home position
                        for _ in range(5):
                            settle_action = teleop_interface.advance()
                            settle_actions = settle_action.repeat(env.num_envs, 1)
                            env.step(settle_actions)
                            env.sim.render()

                        # Recapture pose after reset (use last settle action)
                        paused_target_pose = settle_actions.clone()

                        print(f"[INFO] Arm returned to home position")
                        print(f"[INFO] Bowl and block positions randomized")
                        if total_episodes < args_cli.num_demos or args_cli.num_demos == 0:
                            print(f"[INFO] Ready for episode {total_episodes + 1}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'}")
                        print()

                # Handle manual reset requests
                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    success_step_count = 0

                    # Give environment time to settle at home position
                    for _ in range(5):
                        settle_action = teleop_interface.advance()
                        settle_actions = settle_action.repeat(env.num_envs, 1)
                        env.step(settle_actions)
                        env.sim.render()

                    # Recapture pose after reset (use last settle action)
                    paused_target_pose = settle_actions.clone()
                    print(f"[INFO] Environment reset complete")

                # Check if simulator is stopped
                if env.unwrapped.sim.is_stopped():
                    break

                # Rate limiting
                rate_limiter.sleep(env)

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")

    finally:
        # Print final statistics
        print("\n" + "="*80)
        print("📊 DATA COLLECTION STATISTICS")
        print("="*80)
        print(f"✅ Successful demonstrations: {demo_count}")
        print(f"❌ Failed episodes: {total_episodes - demo_count}")
        print(f"📈 Total episodes attempted: {total_episodes}")
        if total_episodes > 0:
            print(f"🎯 Success rate: {(demo_count / total_episodes) * 100:.1f}%")
        print("="*80)
        print(f"💾 Dataset location: {args_cli.dataset_file}")

        # **VERIFY ACTUAL DATA IN FILE**
        import h5py
        import os
        if os.path.exists(args_cli.dataset_file):
            file_size_mb = os.path.getsize(args_cli.dataset_file) / (1024 * 1024)
            print(f"📁 File size: {file_size_mb:.2f} MB")

            try:
                with h5py.File(args_cli.dataset_file, 'r') as f:
                    saved_demos = len([k for k in f.keys() if k.startswith('demo_')])
                    print(f"📦 Demonstrations in file: {saved_demos}")

                    if saved_demos > 0:
                        # Show first demo structure
                        first_demo = f'demo_{list(f.keys())[0].split("_")[1]}'
                        if first_demo in f:
                            print(f"✅ Dataset contents verified:")
                            for key in f[first_demo].keys():
                                shape = f[first_demo][key].shape if hasattr(f[first_demo][key], 'shape') else 'N/A'
                                print(f"  ✓ {key}: {shape}")
                    else:
                        print("❌ WARNING: File is empty - no demonstrations saved!")
                        print("   The export_episodes() call may be missing or failed")
            except Exception as e:
                print(f"⚠️  Could not read HDF5 file: {e}")
        else:
            print("❌ WARNING: Dataset file does not exist!")
        print("="*80 + "\n")

        if demo_count > 0:
            print(f"✅ Successfully collected {demo_count} demonstrations!")
        elif total_episodes > 0:
            print(f"⚠️  No successful demonstrations collected (0/{total_episodes} attempts)")
        else:
            print(f"ℹ️  No demonstrations recorded")

        print("\nClosing environment...")
        # Close environment
        env.close()
        print("Done!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
