# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import sys
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def block_in_bowl(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    xy_threshold: float = 0.15,  # Block must be within this XY radius of bowl center
    min_height: float = 0.03,  # Block must be at least this high above table
    max_height: float = 0.25,  # Block must be below this height (not thrown too high)
    check_gripper: bool = True,  # Whether to check if gripper is open
    atol=0.0001,
    rtol=0.0001,
):
    """Check if the block is placed in/on the bowl.

    Success criteria:
    - Block is positioned within XY radius of bowl center
    - Block is at appropriate height (in or on bowl)
    - Gripper must be open (block released)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    block: RigidObject = env.scene[block_cfg.name]

    # Get positions in world coordinates
    block_pos_w = block.data.root_pos_w  # (num_envs, 3)

    # Convert to local coordinates by subtracting environment origins
    # This is crucial for multi-environment setups where envs are spread out in a grid
    env_origins = env.scene.env_origins  # (num_envs, 3)
    block_pos = block_pos_w - env_origins

    # Bowl is static XFormPrim (no physics) - use approximate LOCAL visual center position
    # The bowl visual geometry renders at approximately [0.6, 0.25, 0.0] in local frame
    bowl_pos_local = torch.tensor([0.6, 0.25, 0.0], dtype=torch.float32, device=env.device)
    bowl_pos = bowl_pos_local.unsqueeze(0).repeat(env.num_envs, 1)

    # Compute XY distance between block and bowl (both in local coordinates now)
    pos_diff = block_pos - bowl_pos
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)

    # Check if block is within bowl's XY radius
    in_bowl_xy = xy_dist < xy_threshold

    # Check if block is at reasonable height (above table but not too high)
    block_height = block_pos[:, 2]
    reasonable_height = (block_height > min_height) & (block_height < max_height)

    # Check if block is above or at similar height to bowl (block should rest on/in bowl)
    height_diff = block_pos[:, 2] - bowl_pos[:, 2]
    above_bowl = height_diff > -0.05  # Allow some tolerance for settling

    # Combine checks
    in_bowl = in_bowl_xy & reasonable_height & above_bowl

    # Debug output for first environment (optional - can be removed for production)
    if env.num_envs > 0:
        print(f"\n[DEBUG] Success Check (Env 0):", file=sys.stderr)
        print(f"  Block pos: [{block_pos[0, 0]:.3f}, {block_pos[0, 1]:.3f}, {block_pos[0, 2]:.3f}]", file=sys.stderr)
        print(f"  Bowl pos:  [{bowl_pos[0, 0]:.3f}, {bowl_pos[0, 1]:.3f}, {bowl_pos[0, 2]:.3f}] (fixed visual center)", file=sys.stderr)
        print(f"  XY dist: {xy_dist[0]:.3f}m (threshold: {xy_threshold}m) -> {in_bowl_xy[0]}", file=sys.stderr)
        print(f"  Block height: {block_height[0]:.3f}m (range: {min_height}-{max_height}m) -> {reasonable_height[0]}", file=sys.stderr)
        print(f"  Height diff: {height_diff[0]:.3f}m (min: -0.05m) -> {above_bowl[0]}", file=sys.stderr)
        print(f"  Check gripper: {check_gripper}", file=sys.stderr)

    # Check gripper is open (block released) - optional
    if not check_gripper:
        # Skip gripper check - useful for debugging or VR handtracking
        if env.num_envs > 0:
            print(f"  Gripper check: SKIPPED", file=sys.stderr)
            print(f"  Final success: {in_bowl[0]}\n", file=sys.stderr)
        return in_bowl

    # Check gripper is open (block released)
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        in_bowl = torch.logical_and(suction_cup_is_open, in_bowl)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            gripper_open = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
            )

            # Debug gripper state
            if env.num_envs > 0:
                print(f"  Gripper pos: [{robot.data.joint_pos[0, gripper_joint_ids[0]]:.4f}, "
                      f"{robot.data.joint_pos[0, gripper_joint_ids[1]]:.4f}] (target: {env.cfg.gripper_open_val}, "
                      f"tol: {atol}) -> {gripper_open[0]}", file=sys.stderr)
                print(f"  Final success: {torch.logical_and(gripper_open, in_bowl)[0]}\n", file=sys.stderr)

            in_bowl = torch.logical_and(gripper_open, in_bowl)
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return in_bowl
