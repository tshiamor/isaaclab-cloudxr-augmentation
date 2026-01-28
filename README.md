# Isaac Lab Cloud XR - Franka Panda Robotic Learning

External Isaac Lab project for VR hand tracking data collection and vision-based imitation learning with the Franka Panda robot.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Available Environments](#available-environments)
- [VR Hand Tracking Setup](#vr-hand-tracking-setup)
- [Recording Demonstrations](#recording-demonstrations)
- [USD Transform Fix](#usd-transform-fix)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)

---

## Overview

This project provides a complete setup for collecting human demonstrations using VR hand tracking with Cloud XR. The task involves controlling a Franka Panda robot to pick up a wooden block and place it in a bowl.

**Key Features:**
- Single robot environment optimized for VR hand tracking
- Cloud XR integration for remote VR streaming
- HDF5 demonstration recording
- Multiple teleoperation devices (VR, keyboard, spacemouse, gamepad)
- Automatic success detection
- Customizable randomization and scene configuration

---

## Installation

### Prerequisites

1. **Isaac Lab installed** - Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
2. **VR headset** (optional, for hand tracking) - Quest, Vive, Index, etc.
3. **Cloud XR** (optional, for remote streaming)

### Setup

```bash
# 1. Navigate to the project directory
cd $HOME/simulations/franka/isaaclab_cloudxr

# 2. Install the extension in editable mode
$HOME/IsaacLab/isaaclab.sh -p -m pip install -e source/isaaclab_cloudxr

# 3. Verify installation
$HOME/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

---

## Available Environments

### 1. Isaac-Franka-JengaBowl-CloudXR-Single-v0 (Recommended for VR)

**Single robot environment optimized for VR hand tracking**

- **num_envs**: 1 (single robot)
- **Use case**: VR hand tracking, data collection
- **XR enabled**: Yes
- **Timeout**: Disabled (for teleoperation)

**Config file**: `franka_jengabowl_cloudxr_single_env_cfg.py`

### 2. Isaac-Franka-JengaBowl-CloudXR-v0

**Multi-robot environment for parallel training/testing**

- **num_envs**: 4096 (default)
- **Use case**: Parallel RL training, batch testing
- **XR enabled**: Yes
- **Timeout**: 30 seconds

**Config file**: `franka_jengabowl_cloudxr_env_cfg.py`

### 3. Template-Isaaclab-Cloudxr-v0

**Original template (CartPole example)**

- Reference only

---

## VR Hand Tracking Setup

### Quick Start

```bash
cd $HOME/simulations/franka/isaaclab_cloudxr

# Run with VR hand tracking (RECOMMENDED: Use IsaacLab's built-in script)
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --num_envs 1

# Or use custom script (keyboard, spacemouse, gamepad only)
$HOME/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --num_envs 1
```

### Alternative Teleoperation Devices

```bash
# Keyboard teleoperation (for testing without VR)
$HOME/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --num_envs 1

# SpaceMouse
$HOME/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device spacemouse \
    --num_envs 1

# Gamepad
$HOME/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device gamepad \
    --num_envs 1
```

### Using IsaacLab's Built-in Teleoperation Script

```bash
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --num_envs 1
```

### Cloud XR Configuration

The XR anchor position determines where the VR user appears in the simulation:

**Default settings:**
```python
xr.anchor_pos = (-0.1, -0.5, -1.05)  # XYZ position in world
xr.anchor_rot = (0.866, 0, 0, -0.5)   # Quaternion rotation
```

**To customize:**
1. Edit: `source/isaaclab_cloudxr/isaaclab_cloudxr/tasks/manager_based/isaaclab_cloudxr/franka_jengabowl_cloudxr_single_env_cfg.py`
2. Find the `__post_init__` method
3. Update `self.xr.anchor_pos` and `self.xr.anchor_rot`

---

## Recording Demonstrations

### Using IsaacLab's Built-in Recording Script (RECOMMENDED for VR)

```bash
cd $HOME/simulations/franka/isaaclab_cloudxr

# Record 50 demos with VR hand tracking
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/jengabowl_demos.hdf5 \
    --num_demos 50 \
    --step_hz 30

# Record with keyboard (for testing)
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --dataset_file ./datasets/jengabowl_demos_test.hdf5 \
    --num_demos 10
```

### Using the Custom Recording Script (keyboard, spacemouse, gamepad)

```bash
cd $HOME/simulations/franka/isaaclab_cloudxr

# Record with keyboard
$HOME/IsaacLab/isaaclab.sh -p scripts/record_cloudxr_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --dataset_file ./datasets/jengabowl_demos_test.hdf5 \
    --num_demos 10
```

### Recording Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task` | Environment task name | `Isaac-Franka-JengaBowl-CloudXR-Single-v0` |
| `--teleop_device` | Teleoperation device | `handtracking` |
| `--dataset_file` | Output HDF5 file path | `./datasets/jengabowl_demos.hdf5` |
| `--num_demos` | Number of demos to record (0 = infinite) | `0` |
| `--step_hz` | Environment stepping rate in Hz | `30` |
| `--num_success_steps` | Steps with success to conclude demo | `10` |

### Understanding the Recording Output

The script will save demonstrations in HDF5 format with the following structure:

```
jengabowl_demos.hdf5
├── demo_0/
│   ├── observations/
│   │   ├── joint_pos
│   │   ├── joint_vel
│   │   ├── eef_pos
│   │   ├── eef_quat
│   │   └── gripper_pos
│   ├── actions/
│   │   ├── arm_action
│   │   └── gripper_action
│   ├── rewards
│   └── dones
├── demo_1/
│   └── ...
└── metadata
```

### Inspecting Recorded Data

```python
import h5py

# Load the dataset
with h5py.File('./datasets/jengabowl_demos.hdf5', 'r') as f:
    print("Available demonstrations:", list(f.keys()))

    # Access first demo
    demo = f['demo_0']
    print("Demo 0 observations:", list(demo['observations'].keys()))
    print("Demo 0 length:", len(demo['observations']['joint_pos']))
```

---

## USD Transform Fix

The wood block USD file has internal geometry offsets that need to be handled. You have two options:

### Option 1: Keep Current Config (Already Done)

The spawn position is already compensated in the environment config. No action needed!

### Option 2: Fix the USD File

Create a clean USD file with proper root transforms:

```bash
cd $HOME/simulations/franka/isaaclab_cloudxr

# Inspect the USD to see transform issues
$HOME/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    $HOME/WORSPACE_USDS/_36_wood_block_physics.usd

# Fix the USD (creates _fixed.usd)
$HOME/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    $HOME/WORSPACE_USDS/_36_wood_block_physics.usd \
    --fix

# Then update your env config to use the fixed USD
# Edit: franka_jengabowl_cloudxr_single_env_cfg.py
# Change: usd_path="$HOME/WORSPACE_USDS/_36_wood_block_physics_fixed.usd"
```

---

## Customization

### Modify Block Position

Edit: `franka_jengabowl_cloudxr_single_env_cfg.py`

```python
self.scene.block = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Block",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.6, -0.127, 0.1],  # Change X, Y, Z here
        rot=[1.0, 0.0, 0.0, 0.0]
    ),
    # ...
)
```

### Modify Bowl Position

```python
self.scene.bowl = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Bowl",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=[-0.01, 1.16, -1.14071],  # Change X, Y, Z here
        rot=[0.707, -0.707, 0.0, 0.0]
    ),
    # ...
)
```

### Modify Randomization Range

Edit the `randomize_block_position` event in `EventCfg`:

```python
randomize_block_position = EventTerm(
    func=mdp.randomize_object_pose,
    mode="reset",
    params={
        "pose_range": {
            "x": (0.5, 0.7),       # Min, Max X
            "y": (-0.227, -0.027), # Min, Max Y
            "z": (0.1, 0.1),       # Min, Max Z (fixed height)
            "yaw": (0.0, 6.28)     # Random rotation
        },
        "min_separation": 0.08,
        "asset_cfgs": [SceneEntityCfg("block")],
    },
)
```

### Modify Success Criteria

Edit `TerminationsCfg` in your environment config:

```python
success = DoneTerm(
    func=mdp.block_in_bowl,
    params={
        "xy_threshold": 0.15,     # XY radius in meters
        "min_height": 0.01,       # Minimum height above table
        "max_height": 1.0,        # Maximum height
        "atol": 0.015,            # Gripper tolerance
        "rtol": 0.01,             # Relative tolerance
        "check_gripper": True,    # Must open gripper to succeed
    }
)
```

### Add Cameras for Visual Observations

```python
from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

# In your SceneCfg class:
wrist_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="world"
    ),
)
```

---

## Troubleshooting

### VR Headset Not Detected

**Symptoms**: VR headset not recognized, no XR interface

**Solutions**:
- Ensure OpenXR runtime is installed
- Check SteamVR or Oculus software is running
- Verify Cloud XR connection
- Try `--xr` flag explicitly: `--xr true`

### Robot Not Responding to Hand Movements

**Symptoms**: Robot doesn't move when you move your hands

**Solutions**:
- Verify `--teleop_device handtracking` is specified
- Check XR is enabled (auto-enabled for handtracking)
- Adjust sensitivity: `--sensitivity 2.0`
- Ensure VR headset tracking is working
- Check hand tracking is enabled in VR settings

### Block Spawns in Wrong Location

**Symptoms**: Block appears offset or floating

**Solutions**:
- The USD file has internal offsets (normal behavior)
- Position is compensated in the config
- Run `fix_block_usd.py` to inspect/fix transforms
- Check the `pos` value in `RigidObjectCfg.InitialStateCfg`

### Multiple Robots Appear Instead of One

**Symptoms**: Multiple robots/environments visible

**Solutions**:
- Make sure you're using `Isaac-Franka-JengaBowl-CloudXR-Single-v0`
- **NOT** `Isaac-Franka-JengaBowl-CloudXR-v0` (multi-env version)
- Or override with `--num_envs 1`

### Recording Fails or No Data Saved

**Symptoms**: HDF5 file not created or empty

**Solutions**:
- Check dataset directory exists or is created
- Ensure write permissions to output directory
- Complete at least one successful episode
- Check console for error messages
- Verify `--num_demos` is > 0 or set to 0 for infinite

### Low Performance / Lag

**Symptoms**: Low FPS, stuttering, delayed response

**Solutions**:
- Single robot uses less GPU, but Cloud XR streaming requires bandwidth
- Ensure low-latency network connection for Cloud XR
- Reduce rendering quality if needed
- Use DLSS antialiasing (already enabled)
- Close other GPU-intensive applications

### Success Not Detected

**Symptoms**: Task completes but not marked as successful

**Solutions**:
- Check gripper is fully open when releasing block
- Verify block is within XY threshold (15cm radius)
- Check block height is within bounds (0.01m - 1.0m)
- Review `block_in_bowl()` termination function
- Adjust `xy_threshold`, `min_height`, `max_height` parameters

---

## File Structure

```
isaaclab_cloudxr/
│
├── README.md                          # This file
│
├── scripts/
│   ├── teleop_cloudxr.py             # VR teleoperation script
│   ├── record_cloudxr_demos.py       # Custom recording script
│   ├── fix_block_usd.py              # USD transform fix tool
│   ├── list_envs.py                  # List available environments
│   ├── random_agent.py               # Random agent for testing
│   ├── zero_agent.py                 # Zero-action agent
│   ├── brev_cosmos_augment.sh        # Cloud GPU augmentation script
│   └── augment_with_cosmos_transfer2.py  # Cosmos-Transfer2.5 augmentation
│
├── source/isaaclab_cloudxr/
│   └── isaaclab_cloudxr/
│       ├── __init__.py
│       └── tasks/
│           └── manager_based/
│               └── isaaclab_cloudxr/
│                   ├── __init__.py                                    # Environment registration
│                   ├── franka_jengabowl_cloudxr_single_env_cfg.py   # Single robot (VR)
│                   ├── franka_jengabowl_cloudxr_env_cfg.py          # Multi-robot
│                   ├── stack_env_cfg.py                              # Base config
│                   └── mdp/
│                       ├── __init__.py
│                       ├── events.py                # Reset & randomization
│                       ├── observations.py          # State observations
│                       └── terminations.py          # Success/failure checks
│
└── datasets/                          # Created when recording
    └── *.hdf5                         # Recorded demonstrations
```

---

## Quick Reference

### Common Commands

```bash
# VR hand tracking teleoperation (RECOMMENDED)
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking

# Keyboard teleoperation (testing)
$HOME/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard

# Record 50 VR demos (RECOMMENDED)
$HOME/IsaacLab/isaaclab.sh -p \
    $HOME/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/jengabowl_demos.hdf5 \
    --num_demos 50

# List environments
$HOME/IsaacLab/isaaclab.sh -p scripts/list_envs.py

# Inspect USD
$HOME/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    $HOME/WORSPACE_USDS/_36_wood_block_physics.usd
```

### Environment Names

- `Isaac-Franka-JengaBowl-CloudXR-Single-v0` -- Use for VR
- `Isaac-Franka-JengaBowl-CloudXR-v0` -- Multi-robot (4096 envs)
- `Template-Isaaclab-Cloudxr-v0` -- Template reference

---

## Resources

- **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/
- **Isaac Lab GitHub**: https://github.com/isaac-sim/IsaacLab
- **Cloud XR SDK**: https://docs.nvidia.com/cloudxr/

---

## License

This project follows Isaac Lab's license: BSD-3-Clause
