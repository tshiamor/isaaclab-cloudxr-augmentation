# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize block position
    # Keep block elevated to prevent physics glitches
    # Moved farther from robot (x >= 0.5m) for easier control
    randomize_block_position = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.5, 0.7),        # Moved farther: 0.5m to 0.7m from robot
                "y": (-0.227, -0.027),  # Slightly randomize around -0.127
                "z": (0.1, 0.1),        # Keep elevated
                "yaw": (0.0, 6.28)      # Random rotation
            },
            "min_separation": 0.08,
            "asset_cfgs": [
                SceneEntityCfg("block"),
            ],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for jengaBowl task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for jengaBowl task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    block_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("block")}
    )

    success = DoneTerm(
        func=mdp.block_in_bowl,
        params={
            "xy_threshold": 0.15,  # Stricter: 15cm radius (block must be IN the bowl)
            "min_height": 0.01,   # Lower minimum
            "max_height": 1.0,    # Higher maximum
            "atol": 0.015,        # Gripper tolerance: gripper must be within 0.015 of 0.04 (open position)
            "rtol": 0.01,         # Relative tolerance
            "check_gripper": True,  # Enable gripper check - must be open to succeed
        }
    )


@configclass
class FrankaJengaBowlCloudXREnvCfg(StackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Ensure gravity is enabled
        self.sim.gravity = (0.0, 0.0, -9.81)

        # Set observations for single block task
        self.observations = ObservationsCfg()

        # Set terminations for single block task
        self.terminations = TerminationsCfg()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties
        block_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Robot origin at (0,0,0), Table at (0.5, 0.0, 0)
        # USD file has internal geometry offset, so Block prim position is adjusted to compensate

        # Single jenga block on table surface
        # Block dimensions: 25cm x 75cm x 15cm (scaled 10x from original 2.5cm x 7.5cm x 1.5cm)
        # Using original USD - spawn higher to prevent falling through table
        # Moved farther from robot (x >= 0.5m) for easier control
        self.scene.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.6, -0.127, 0.1],  # Moved farther from robot: 0.6m in x-axis
                rot=[1.0, 0.0, 0.0, 0.0]  # No rotation
            ),
            spawn=UsdFileCfg(
                usd_path="/home/tshiamo/WORSPACE_USDS/_36_wood_block_physics.usd",
                scale=(0.25, 0.75, 0.15),  # Scaled up 10x: 25cm x 75cm x 15cm
                rigid_props=block_properties,
                semantic_tags=[("class", "block")],
            ),
        )

        # Bowl (static asset - doesn't move)
        # Position on table surface, centered and within robot reach
        # Bowl's pivot point is offset, so position changes with rotation to keep it on table
        # Adjusted position: +0.25m in x-axis from original
        self.scene.bowl = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[-0.01, 1.16, -1.14071],  # Bowl position adjusted for better reach
                rot=[0.707, -0.707, 0.0, 0.0]  # -90-degree rotation about x-axis to face upward
            ),
            spawn=UsdFileCfg(
                usd_path="/home/tshiamo/WORSPACE_USDS/_24_bowl.usd",
                scale=(1.0, 1.0, 1.0),  # Default scale
                semantic_tags=[("class", "bowl")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
