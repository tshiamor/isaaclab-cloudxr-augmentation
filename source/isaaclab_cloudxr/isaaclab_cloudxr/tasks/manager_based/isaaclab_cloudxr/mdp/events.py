# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default pose for robots in all envs."""
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add gaussian noise to joint states."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """Sample random object poses with minimum separation."""
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]] = {},
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),  # Default identity quaternion [w, x, y, z]
):
    """Randomize a single static asset's pose (for AssetBaseCfg like bowls).

    This function works with static assets that don't have rigid body physics.
    It directly modifies the USD prim transform.
    Preserves a fixed orientation passed as parameter.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        asset_cfg: Asset configuration (just needs the name)
        pose_range: Position/rotation ranges (position is randomized, rotation is ignored)
        orientation: Fixed orientation quaternion [w, x, y, z] to preserve
    """
    if env_ids is None:
        return

    # Get the asset
    asset = env.scene[asset_cfg.name]

    # Import USD utilities - use pxr directly (part of USD, always available)
    from pxr import Gf, UsdGeom, Usd

    # Get USD stage from InteractiveScene
    stage = env.scene.stage

    # Generate random pose for each environment
    for cur_env in env_ids.tolist():
        # Sample random pose (only position, rotation is preserved)
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        sample = [random.uniform(range[0], range[1]) for range in range_list]

        # Calculate world position (local + env origin)
        local_pos = torch.tensor([sample[0], sample[1], sample[2]], device=env.device)
        world_pos = local_pos + env.scene.env_origins[cur_env, 0:3]

        # Get the prim path for this environment from scene config (not from runtime asset object)
        asset_config = getattr(env.cfg.scene, asset_cfg.name)
        prim_path = asset_config.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{cur_env}")

        # Get the prim
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            # Get xformable interface
            xformable = UsdGeom.Xformable(prim)

            # Set translation
            translate_op = None
            orient_op = None

            # Get existing ops or create new ones
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op

            # Create ops if they don't exist
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = xformable.AddOrientOp()

            # Set values
            translate_op.Set(Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))

            # Use fixed orientation from parameter (preserves bowl facing up)
            # Both Isaac Lab and USD Quatd use [w, x, y, z] convention
            orient_op.Set(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """Randomize object poses in each environment independently."""
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )
