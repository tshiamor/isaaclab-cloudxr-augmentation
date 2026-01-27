# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Franka JengaBowl RL task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


# ===== Custom Rewards for JengaBowl Task =====


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """
    Reward for end effector being close to block.
    Uses exponential kernel: exp(-dist^2 / std^2)
    Returns 1.0 when touching, decreases with distance.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene["ee_frame"]

    # Get positions
    object_pos = object.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Compute Euclidean distance
    distance = torch.norm(object_pos - ee_pos, dim=-1)

    # Exponential reward (closer = higher reward)
    reward = torch.exp(-distance**2 / std**2)

    return reward


def gripper_closed_with_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    distance_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward for closing gripper when near block.
    Encourages grasping behavior.
    """
    robot: Articulation = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene["ee_frame"]

    # Get gripper state
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]

    # Check if ee is near object
    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos - ee_pos, dim=-1)

    # Binary checks
    near_object = distance < distance_threshold
    gripper_closed = gripper_pos < 0.02  # Gripper is closed

    # Reward only when both conditions are true
    reward = (near_object & gripper_closed).float()

    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """
    Reward for moving block closer to bowl.
    Only rewards when block is lifted (height > minimal_height).
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Block position (convert from world to local coordinates)
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)
    object_pos = object_pos_w - env.scene.env_origins  # Local coordinates

    # Bowl position (local visual center)
    bowl_pos_local = torch.tensor([0.6, 0.25, 0.0], device=env.device)
    bowl_pos = bowl_pos_local.unsqueeze(0).repeat(env.num_envs, 1)

    # Compute XY distance to bowl (both in local coordinates)
    distance = torch.norm(object_pos[:, :2] - bowl_pos[:, :2], dim=-1)

    # Exponential reward for proximity
    proximity_reward = torch.exp(-distance**2 / std**2)

    # Only reward if block is lifted (local height)
    block_lifted = object_pos[:, 2] > minimal_height
    reward = proximity_reward * block_lifted.float()

    return reward


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalty for large changes in actions.
    Encourages smooth, controlled movements.
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalty for large joint velocities.
    Encourages energy-efficient motion.
    """
    robot: Articulation = env.scene["robot"]
    # Only arm joints (first 7), not gripper
    return torch.sum(torch.square(robot.data.joint_vel[:, :7]), dim=1)
