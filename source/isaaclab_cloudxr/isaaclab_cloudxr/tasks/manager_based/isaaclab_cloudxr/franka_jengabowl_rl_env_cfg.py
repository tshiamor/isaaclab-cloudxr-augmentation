# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RL training configuration for Franka JengaBowl task.
Multi-environment setup with reward functions for autonomous learning.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .franka_jengabowl_cloudxr_single_visuomotor_env_cfg import (
    FrankaJengaBowlCloudXRSingleVisuomotorEnvCfg,
    EventCfg as BaseEventCfg,
)


@configclass
class RewardsCfg:
    """Reward terms for RL training."""

    # === Reaching Rewards ===
    # Encourage end effector to move closer to block
    reaching_block = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("block"),
        },
        weight=2.0,
    )

    # === Grasping Rewards ===
    # Reward for closing gripper when near block
    grasping = RewTerm(
        func=mdp.gripper_closed_with_object,
        params={
            "object_cfg": SceneEntityCfg("block"),
            "distance_threshold": 0.05,
        },
        weight=5.0,
    )

    # === Transport Rewards ===
    # Reward for moving block closer to bowl (only when lifted)
    transporting = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.2,
            "minimal_height": 0.05,
            "object_cfg": SceneEntityCfg("block"),
        },
        weight=5.0,
    )

    # === Success Rewards ===
    # Large reward for successful placement in bowl
    success = RewTerm(
        func=mdp.block_in_bowl,
        params={
            "xy_threshold": 0.15,
            "min_height": -0.02,
            "max_height": 0.10,
            "check_gripper": True,
            "atol": 0.020,
            "rtol": 0.05,
        },
        weight=100.0,  # Large bonus for success
    )

    # === Regularization (Penalties) ===
    # Penalize large actions (encourage smooth control)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Penalize joint velocities (encourage energy efficiency)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
    )


@configclass
class TerminationsCfg:
    """Termination terms for RL training."""

    # Timeout after max episode length
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success termination (end episode early on success)
    success = DoneTerm(
        func=mdp.block_in_bowl,
        params={
            "xy_threshold": 0.15,
            "min_height": -0.02,
            "max_height": 0.10,
            "check_gripper": True,
            "atol": 0.020,
            "rtol": 0.05,
        }
    )


@configclass
class EventCfg(BaseEventCfg):
    """Events configuration with increased randomization for RL."""

    # Override block randomization with wider ranges
    randomize_block_position = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.4, 0.8),        # Wider X range
                "y": (-0.3, 0.1),       # Wider Y range
                "z": (0.05, 0.15),      # Vary height
                "yaw": (0.0, 6.28)      # Full rotation
            },
            "min_separation": 0.08,
            "asset_cfgs": [
                SceneEntityCfg("block"),
            ],
        },
    )


@configclass
class FrankaJengaBowlRLEnvCfg(FrankaJengaBowlCloudXRSingleVisuomotorEnvCfg):
    """Configuration for RL training of Franka JengaBowl task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # === RL-specific configuration ===

        # Enable massive parallelization for RL (GPU-accelerated)
        self.scene.num_envs = 4096  # Was 1 for VR
        self.scene.env_spacing = 2.5

        # Enable rewards for RL training
        self.rewards = RewardsCfg()

        # Set terminations (including success termination for RL)
        self.terminations = TerminationsCfg()

        # Use enhanced randomization for better generalization
        self.events = EventCfg()

        # === Optimize for training speed ===

        # Disable cameras for faster training (state observations only)
        # Remove cameras from scene entirely
        self.scene.wrist_camera = None
        self.scene.external_camera = None

        # Remove camera observations
        self.observations.wrist_camera = None
        self.observations.external_camera = None

        # RSL-RL requires concatenated observations (1D flattened vector)
        self.observations.policy.concatenate_terms = True

        # Shorter episodes for faster learning iterations
        self.episode_length_s = 10.0  # Was 30.0 for VR

        # Disable VR teleoperation device (not needed for RL)
        self.teleop_devices = None

        # Disable XR rendering
        self.sim.render.antialiasing_mode = None
