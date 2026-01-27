"""Configuration for camera recorders.

This module defines recorder configurations for saving camera observations
in visuomotor learning demonstrations.
"""

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import camera_recorders


@configclass
class AllObservationGroupsRecorderCfg(RecorderTermCfg):
    """Configuration for the all observation groups recorder term."""

    class_type: type[RecorderTerm] = camera_recorders.AllObservationGroupsRecorder


@configclass
class VisuomotorRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration for visuomotor learning with cameras.

    This recorder manager saves:
    - Initial states (robot, objects)
    - Actions (user commands)
    - Processed actions (actual robot commands)
    - States (robot joint states, object poses)
    - ALL observations (including camera RGB + Depth images)
    """

    # Import default recorders from Isaac Lab
    from isaaclab.envs.mdp.recorders.recorders_cfg import (
        InitialStateRecorderCfg,
        PostStepProcessedActionsRecorderCfg,
        PostStepStatesRecorderCfg,
        PreStepActionsRecorderCfg,
    )

    # Default recorders (states, actions)
    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()

    # **CRITICAL: Use custom recorder that saves ALL observation groups**
    # This replaces PreStepFlatPolicyObservationsRecorderCfg which only saves "policy" group
    record_all_observations = AllObservationGroupsRecorderCfg()
