"""Custom recorders for saving camera observations in demonstrations.

This module provides recorder terms that save camera RGB and depth images
in addition to the default state observations.
"""

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.envs import ManagerBasedRLEnv


class AllObservationGroupsRecorder(RecorderTerm):
    """Recorder term that records ALL observation groups (policy, cameras, etc.).

    Unlike the default PreStepFlatPolicyObservationsRecorder which only saves the "policy"
    observation group, this recorder saves all observation groups including cameras.

    This is essential for visuomotor learning where we need RGB/Depth images.
    """

    def record_pre_step(self):
        """Record all observation groups from the environment.

        Returns:
            tuple: ("obs", dict) where dict contains all observation groups
        """
        # Return ALL observation groups, not just "policy"
        # This will include: policy, wrist_camera, external_camera, etc.
        return "obs", self._env.obs_buf


class CameraRGBRecorder(RecorderTerm):
    """Recorder term that records RGB images from a specific camera.

    Args:
        camera_name: Name of the camera observation group (e.g., "wrist_camera")
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_name = cfg.params.get("camera_name", "wrist_camera")

    def record_pre_step(self):
        """Record RGB images from the specified camera.

        Returns:
            tuple: (name, tensor) - RGB images with shape (num_envs, H, W, 3)
        """
        if self.camera_name in self._env.obs_buf:
            camera_obs = self._env.obs_buf[self.camera_name]
            if isinstance(camera_obs, dict) and "rgb" in camera_obs:
                return f"{self.camera_name}_rgb", camera_obs["rgb"]
        return None, None


class CameraDepthRecorder(RecorderTerm):
    """Recorder term that records depth images from a specific camera.

    Args:
        camera_name: Name of the camera observation group (e.g., "wrist_camera")
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_name = cfg.params.get("camera_name", "wrist_camera")

    def record_pre_step(self):
        """Record depth images from the specified camera.

        Returns:
            tuple: (name, tensor) - Depth images with shape (num_envs, H, W)
        """
        if self.camera_name in self._env.obs_buf:
            camera_obs = self._env.obs_buf[self.camera_name]
            if isinstance(camera_obs, dict) and "depth" in camera_obs:
                return f"{self.camera_name}_depth", camera_obs["depth"]
        return None, None
