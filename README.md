# Isaac Lab Cloud XR - Franka Panda Robotic Learning

External Isaac Lab project for VR hand tracking data collection and vision-based imitation learning with the Franka Panda robot.

## 🎯 Quick Start

**Recommended Path: Vision-Based Learning with Isaac Lab Mimic** ⭐

1. **[Collect VR Demos](CAMERA_RECORDING_GUIDE.md)** → 49 demos with camera data ✅ DONE
2. **[Annotate Demos](PLAYBACK_AND_ANNOTATION_GUIDE.md)** → Add subtask annotations ← **YOU ARE HERE**
3. **[Generate Dataset](ISAACLAB_MIMIC_WORKFLOW.md)** → 49 → 1000+ augmented demos
4. **[Train Policy](ISAACLAB_MIMIC_WORKFLOW.md#step-3-train-vision-based-policy-with-robomimic)** → Vision-based behavior cloning
5. **[Deploy & Evaluate](ISAACLAB_MIMIC_WORKFLOW.md#step-4-evaluate-trained-policy)** → 75-86% success rate

---

## 📚 Documentation Index

### 🚀 Getting Started

| Guide | Description | When to Use |
|-------|-------------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | Fast setup and first demo | New to project |
| **[QUICK_COMMANDS.md](QUICK_COMMANDS.md)** | Common commands reference | Quick lookup |
| **[VR_HANDTRACKING_GUIDE.md](VR_HANDTRACKING_GUIDE.md)** | VR setup and teleoperation | Setting up VR |

### 🎓 Vision-Based Learning Pipeline (RECOMMENDED)

| Guide | Description | Status |
|-------|-------------|--------|
| **[ISAACLAB_MIMIC_WORKFLOW.md](ISAACLAB_MIMIC_WORKFLOW.md)** | Complete Mimic pipeline (annotation → augmentation → training) | ⭐ **START HERE** |
| **[PLAYBACK_AND_ANNOTATION_GUIDE.md](PLAYBACK_AND_ANNOTATION_GUIDE.md)** | Replay demos and annotate subtasks | **Current Step** |
| **[CAMERA_RECORDING_GUIDE.md](CAMERA_RECORDING_GUIDE.md)** | Record demos WITH camera observations | ✅ Completed |
| **[VISION_BASED_PIPELINE.md](VISION_BASED_PIPELINE.md)** | Overview of vision-based learning | Background |
| **[NEXT_STEPS.md](NEXT_STEPS.md)** | What to do after data collection | Planning |

### 🎮 Data Collection

| Guide | Description | When to Use |
|-------|-------------|-------------|
| **[CAMERA_RECORDING_GUIDE.md](CAMERA_RECORDING_GUIDE.md)** | Record demos with wrist + external cameras | Vision-based learning |
| **[VR_HANDTRACKING_GUIDE.md](VR_HANDTRACKING_GUIDE.md)** | VR hand tracking setup and controls | VR teleoperation |
| **[VISUOMOTOR_TASK_DOCUMENTATION.md](VISUOMOTOR_TASK_DOCUMENTATION.md)** | Environment config for camera tasks | Technical reference |

### 🤖 Reinforcement Learning (Alternative Path)

| Guide | Description | Status |
|-------|-------------|--------|
| **[RL_QUICKSTART.md](RL_QUICKSTART.md)** | Fast RL training setup | Alternative to Mimic |
| **[RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md)** | Detailed RL training guide | For RL users |
| **[RL_SETUP_COMPLETE.md](RL_SETUP_COMPLETE.md)** | RL environment setup confirmation | Setup verification |
| **[TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md)** | Training performance analysis | Post-training |

### 🔧 Troubleshooting & Fixes

| Category | Guides | Description |
|----------|--------|-------------|
| **VR Issues** | [VR_HANDTRACKING_FIX.md](VR_HANDTRACKING_FIX.md) | VR connection and tracking fixes |
| **Camera Issues** | [CAMERA_FIX.md](CAMERA_FIX.md), [HDF5_EXPORT_FIX.md](HDF5_EXPORT_FIX.md) | Camera recording and export fixes |
| **Recording Issues** | [DEMO_COUNTING_AND_DEBUG_FIX.md](DEMO_COUNTING_AND_DEBUG_FIX.md), [PAUSE_PLAY_TERMINATION_FIX.md](PAUSE_PLAY_TERMINATION_FIX.md) | Demo recording fixes |
| **Environment Issues** | [BOWL_POSITION_ORIENTATION_FIX.md](BOWL_POSITION_ORIENTATION_FIX.md), [BOWL_STATIC_ASSET_FIX.md](BOWL_STATIC_ASSET_FIX.md) | Scene setup fixes |
| **Control Issues** | [ARM_FREEZE_IK_FIX.md](ARM_FREEZE_IK_FIX.md), [ARM_LOCK_FIX.md](ARM_LOCK_FIX.md), [CALLBACK_FREEZE_FIX.md](CALLBACK_FREEZE_FIX.md) | Robot control fixes |
| **Code Issues** | [IMPORT_ERROR_FIX.md](IMPORT_ERROR_FIX.md), [ATTRIBUTE_ERROR_FIX.md](ATTRIBUTE_ERROR_FIX.md), [MISSING_FUNCTION_FIX.md](MISSING_FUNCTION_FIX.md) | Code and dependency fixes |
| **Complete Fixes** | [ALL_FIXES_COMPLETE.md](ALL_FIXES_COMPLETE.md), [FINAL_FIX_COMPLETE.md](FINAL_FIX_COMPLETE.md) | Summary of all fixes |

### 📖 Advanced Topics

| Guide | Description | When to Use |
|-------|-------------|-------------|
| **[AUGMENTATION_GUIDE.md](AUGMENTATION_GUIDE.md)** | Data augmentation strategies (OLD - Use Mimic instead) | Legacy reference |
| **[MIMICGEN_GUIDE.md](MIMICGEN_GUIDE.md)** | MimicGen comparison (explains incompatibility) | Understanding limitations |
| **[COORDINATE_FRAME_FIX.md](COORDINATE_FRAME_FIX.md)** | Coordinate frame transformations | Advanced debugging |
| **[TERMINATION_AND_BOWL_FIX.md](TERMINATION_AND_BOWL_FIX.md)** | Success criteria tuning | Custom task design |

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
- ✅ Single robot environment optimized for VR hand tracking
- ✅ Cloud XR integration for remote VR streaming
- ✅ HDF5 demonstration recording
- ✅ Multiple teleoperation devices (VR, keyboard, spacemouse, gamepad)
- ✅ Automatic success detection
- ✅ Customizable randomization and scene configuration

---

## Installation

### Prerequisites

1. **Isaac Lab installed** - Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
2. **VR headset** (optional, for hand tracking) - Quest, Vive, Index, etc.
3. **Cloud XR** (optional, for remote streaming)

### Setup

```bash
# 1. Navigate to the project directory
cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

# 2. Install the extension in editable mode
/home/tshiamo/IsaacLab/isaaclab.sh -p -m pip install -e source/isaaclab_cloudxr

# 3. Verify installation
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

---

## Available Environments

### 1. Isaac-Franka-JengaBowl-CloudXR-Single-v0 ⭐ (Recommended for VR)

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
cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

# Run with VR hand tracking (RECOMMENDED: Use IsaacLab's built-in script)
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --num_envs 1

# Or use custom script (keyboard, spacemouse, gamepad only)
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --num_envs 1
```

### Alternative Teleoperation Devices

```bash
# Keyboard teleoperation (for testing without VR)
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --num_envs 1

# SpaceMouse
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device spacemouse \
    --num_envs 1

# Gamepad
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device gamepad \
    --num_envs 1
```

### Using IsaacLab's Built-in Teleoperation Script

```bash
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
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
cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

# Record 50 demos with VR hand tracking
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/jengabowl_demos.hdf5 \
    --num_demos 50 \
    --step_hz 30

# Record with keyboard (for testing)
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard \
    --dataset_file ./datasets/jengabowl_demos_test.hdf5 \
    --num_demos 10
```

### Using the Custom Recording Script (keyboard, spacemouse, gamepad)

```bash
cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

# Record with keyboard
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/record_cloudxr_demos.py \
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
cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

# Inspect the USD to see transform issues
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    /home/tshiamo/WORSPACE_USDS/_36_wood_block_physics.usd

# Fix the USD (creates _fixed.usd)
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    /home/tshiamo/WORSPACE_USDS/_36_wood_block_physics.usd \
    --fix

# Then update your env config to use the fixed USD
# Edit: franka_jengabowl_cloudxr_single_env_cfg.py
# Change: usd_path="/home/tshiamo/WORSPACE_USDS/_36_wood_block_physics_fixed.usd"
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
/home/tshiamo/simulations/franka/isaaclab_cloudxr/
│
├── README.md                          # This file
├── VR_HANDTRACKING_GUIDE.md          # Detailed VR guide
│
├── scripts/
│   ├── teleop_cloudxr.py             # VR teleoperation script
│   ├── record_cloudxr_demos.py       # Custom recording script
│   ├── fix_block_usd.py              # USD transform fix tool
│   ├── list_envs.py                  # List available environments
│   ├── random_agent.py               # Random agent for testing
│   └── zero_agent.py                 # Zero-action agent
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
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking

# Keyboard teleoperation (testing)
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/teleop_cloudxr.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device keyboard

# Record 50 VR demos (RECOMMENDED)
/home/tshiamo/IsaacLab/isaaclab.sh -p \
    /home/tshiamo/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Franka-JengaBowl-CloudXR-Single-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/jengabowl_demos.hdf5 \
    --num_demos 50

# List environments
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/list_envs.py

# Inspect USD
/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/fix_block_usd.py \
    /home/tshiamo/WORSPACE_USDS/_36_wood_block_physics.usd
```

### Environment Names

- `Isaac-Franka-JengaBowl-CloudXR-Single-v0` ← **Use for VR**
- `Isaac-Franka-JengaBowl-CloudXR-v0` ← Multi-robot (4096 envs)
- `Template-Isaaclab-Cloudxr-v0` ← Template reference

---

---

## 📖 Documentation Quick Reference

### I want to...

| Task | Guide to Read |
|------|---------------|
| **Get started from scratch** | [QUICKSTART.md](QUICKSTART.md) |
| **Annotate my 49 demos** | [PLAYBACK_AND_ANNOTATION_GUIDE.md](PLAYBACK_AND_ANNOTATION_GUIDE.md) ← **START HERE** |
| **Understand the complete Mimic workflow** | [ISAACLAB_MIMIC_WORKFLOW.md](ISAACLAB_MIMIC_WORKFLOW.md) |
| **Record more demos with cameras** | [CAMERA_RECORDING_GUIDE.md](CAMERA_RECORDING_GUIDE.md) |
| **Setup VR hand tracking** | [VR_HANDTRACKING_GUIDE.md](VR_HANDTRACKING_GUIDE.md) |
| **Train with RL instead of Mimic** | [RL_QUICKSTART.md](RL_QUICKSTART.md) |
| **Find common commands** | [QUICK_COMMANDS.md](QUICK_COMMANDS.md) |
| **Fix VR connection issues** | [VR_HANDTRACKING_FIX.md](VR_HANDTRACKING_FIX.md) |
| **Fix camera recording issues** | [HDF5_EXPORT_FIX.md](HDF5_EXPORT_FIX.md) |
| **See all available guides** | [📚 Documentation Index](#-documentation-index) (above) |

---

## Resources

### Official Documentation
- **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/
- **Isaac Lab Mimic**: https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/overview/imitation-learning/teleop_imitation.html
- **Isaac Lab GitHub**: https://github.com/isaac-sim/IsaacLab
- **OpenXR Specification**: https://www.khronos.org/openxr/
- **Cloud XR SDK**: https://docs.nvidia.com/cloudxr/

### Project Documentation
- **[Complete Documentation Index](#-documentation-index)** - All guides organized by category
- **[Quick Reference](#-documentation-quick-reference)** - Find the right guide for your task

---

## Current Status

✅ **Data Collection**: 49 high-quality demonstrations (2.8 GB)
🔄 **Annotation**: Ready to annotate with Isaac Lab Mimic
⏳ **Augmentation**: Next step - generate 650-820 augmented demos
⏳ **Training**: Vision-based policy training with robomimic
⏳ **Deployment**: Expected 75-86% success rate

**Next Action**: [Annotate your demos](PLAYBACK_AND_ANNOTATION_GUIDE.md) → Start the Mimic workflow!

---

## Contributing

This is an external Isaac Lab project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project follows Isaac Lab's license: BSD-3-Clause

---

## Support

For issues or questions:
- **Project Documentation**: See [📚 Documentation Index](#-documentation-index) above
- **Isaac Lab GitHub Issues**: https://github.com/isaac-sim/IsaacLab/issues
- **Isaac Lab Forums**: https://forums.developer.nvidia.com/c/omniverse/simulation/69

---

**🚀 Ready to train a vision-based policy? [Start annotating your demos!](PLAYBACK_AND_ANNOTATION_GUIDE.md)**
