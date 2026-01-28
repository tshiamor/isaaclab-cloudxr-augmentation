#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 dataset to LIBERO-compatible robomimic format.

LIBERO uses robomimic's HDF5 format with specific observation keys:
  - obs/agentview_image          : Third-person camera (H, W, 3) uint8
  - obs/robot0_eye_in_hand_image : Wrist camera (H, W, 3) uint8
  - obs/robot0_eef_pos           : End-effector position (3,)
  - obs/robot0_eef_quat          : End-effector quaternion (4,)
  - obs/robot0_gripper_qpos      : Gripper joint positions (2,)
  - obs/robot0_joint_pos         : Joint positions (7,) (arm only)
  - obs/object                   : Object state vector
  - actions                      : 7D actions [-1, 1]
  - rewards                      : Per-step rewards
  - dones                        : Episode termination flags
  - states                       : Flattened simulator state

Usage:
    python scripts/convert_to_libero.py \
        --input datasets/cosmos_generated_202.hdf5 \
        --output datasets/libero_format/block_stacking.hdf5
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path


def flatten_state(demo_states):
    """Flatten all state arrays into a single state vector per timestep."""
    arrays = []
    for key in sorted(demo_states.keys()):
        item = demo_states[key]
        if isinstance(item, h5py.Dataset):
            arrays.append(item[:])
        elif isinstance(item, h5py.Group):
            arrays.append(flatten_state(item))
    return np.concatenate(arrays, axis=-1)


def convert_to_libero(input_file: str, output_file: str, image_size: int = 128):
    """
    Convert Isaac Lab HDF5 to LIBERO-compatible robomimic format.

    Args:
        input_file: Path to Isaac Lab HDF5 dataset
        output_file: Path to output LIBERO-format HDF5
        image_size: Resize images to this size (LIBERO default is 128x128)
    """
    print("=" * 70)
    print("Converting Isaac Lab Dataset to LIBERO Format")
    print("=" * 70)
    print(f"Input:      {input_file}")
    print(f"Output:     {output_file}")
    print(f"Image size: {image_size}x{image_size}")
    print("=" * 70)

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Try importing cv2 for resize, fall back to PIL
    try:
        import cv2
        def resize_images(imgs, size):
            return np.array([cv2.resize(img, (size, size)) for img in imgs])
    except ImportError:
        from PIL import Image
        def resize_images(imgs, size):
            out = []
            for img in imgs:
                pil = Image.fromarray(img)
                pil = pil.resize((size, size), Image.BILINEAR)
                out.append(np.array(pil))
            return np.array(out)

    # env_args for LIBERO compatibility
    env_args = {
        "env_name": "Isaac-Franka-BlockStacking",
        "env_type": 2,  # robomimic env type for generic environments
        "env_kwargs": {
            "task": "block_stacking",
            "robot": "Franka",
            "controller": "joint_velocity",
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": image_size,
            "camera_widths": image_size,
        },
    }

    with h5py.File(input_file, "r") as f_in:
        demos = sorted(
            [k for k in f_in["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        print(f"Found {len(demos)} demonstrations\n")

        with h5py.File(output_file, "w") as f_out:
            # Top-level attributes
            f_out.attrs["env_args"] = json.dumps(env_args)
            f_out.attrs["total"] = len(demos)

            data_grp = f_out.create_group("data")
            total_samples = 0
            successful = 0

            for idx, demo_name in enumerate(demos):
                demo_in = f_in[f"data/{demo_name}"]
                obs_in = demo_in["obs"]
                T = demo_in["actions"].shape[0]
                total_samples += T

                demo_out = data_grp.create_group(demo_name)
                obs_out = demo_out.create_group("obs")

                # --- Images ---
                # agentview_image (third-person / table cam)
                table_cam = obs_in["table_cam"][:]  # (T, 200, 200, 3) uint8
                if table_cam.shape[1] != image_size:
                    table_cam = resize_images(table_cam, image_size)
                obs_out.create_dataset(
                    "agentview_image", data=table_cam, dtype=np.uint8
                )

                # robot0_eye_in_hand_image (wrist cam)
                wrist_cam = obs_in["wrist_cam"][:]  # (T, 200, 200, 3) uint8
                if wrist_cam.shape[1] != image_size:
                    wrist_cam = resize_images(wrist_cam, image_size)
                obs_out.create_dataset(
                    "robot0_eye_in_hand_image", data=wrist_cam, dtype=np.uint8
                )

                # --- Proprioception ---
                obs_out.create_dataset(
                    "robot0_eef_pos", data=obs_in["eef_pos"][:], dtype=np.float32
                )
                obs_out.create_dataset(
                    "robot0_eef_quat", data=obs_in["eef_quat"][:], dtype=np.float32
                )
                obs_out.create_dataset(
                    "robot0_gripper_qpos",
                    data=obs_in["gripper_pos"][:],
                    dtype=np.float32,
                )
                # Arm joints only (first 7 of 9)
                joint_pos = obs_in["joint_pos"][:]
                obs_out.create_dataset(
                    "robot0_joint_pos", data=joint_pos[:, :7], dtype=np.float32
                )

                # --- Object state ---
                if "object" in obs_in:
                    obs_out.create_dataset(
                        "object", data=obs_in["object"][:], dtype=np.float32
                    )
                else:
                    # Build from cube positions + orientations
                    cube_pos = obs_in["cube_positions"][:]  # (T, 9)
                    cube_ori = obs_in["cube_orientations"][:]  # (T, 12)
                    obj_state = np.concatenate([cube_pos, cube_ori], axis=-1)
                    obs_out.create_dataset(
                        "object", data=obj_state, dtype=np.float32
                    )

                # --- Actions ---
                actions = demo_in["actions"][:]  # (T, 7), already [-1, 1]
                demo_out.create_dataset("actions", data=actions, dtype=np.float32)

                # --- Rewards ---
                rewards = np.zeros(T, dtype=np.float32)
                is_success = bool(demo_in.attrs.get("success", False))
                if is_success:
                    rewards[-1] = 1.0
                    successful += 1
                demo_out.create_dataset("rewards", data=rewards, dtype=np.float32)

                # --- Dones ---
                dones = np.zeros(T, dtype=bool)
                dones[-1] = True
                demo_out.create_dataset("dones", data=dones)

                # --- States (flattened simulator state) ---
                if "states" in demo_in:
                    flat_state = flatten_state(demo_in["states"])
                    demo_out.create_dataset(
                        "states", data=flat_state, dtype=np.float32
                    )

                # --- Attributes ---
                demo_out.attrs["num_samples"] = T
                demo_out.attrs["success"] = is_success

                if (idx + 1) % 20 == 0 or idx == len(demos) - 1:
                    print(f"  Processed {idx + 1}/{len(demos)} demos...")

            # --- Train/val mask ---
            mask_grp = f_out.create_group("mask")
            n_train = int(len(demos) * 0.9)
            train_demos = [f"data/{d}" for d in demos[:n_train]]
            val_demos = [f"data/{d}" for d in demos[n_train:]]

            dt = h5py.special_dtype(vlen=str)
            mask_grp.create_dataset(
                "train", data=np.array(train_demos, dtype=object), dtype=dt
            )
            mask_grp.create_dataset(
                "valid", data=np.array(val_demos, dtype=object), dtype=dt
            )

            print(f"\n{'=' * 70}")
            print("Conversion Complete")
            print(f"{'=' * 70}")
            print(f"Demonstrations: {len(demos)}")
            print(f"Successful:     {successful}")
            print(f"Total samples:  {total_samples}")
            print(f"Train/Val:      {n_train}/{len(demos) - n_train}")
            print(f"Output:         {output_file}")
            print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 to LIBERO-compatible format"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input Isaac Lab HDF5 file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output LIBERO-format HDF5 file"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Resize images to NxN (default: 128, LIBERO standard)",
    )
    args = parser.parse_args()

    convert_to_libero(args.input, args.output, args.image_size)


if __name__ == "__main__":
    main()
