#!/usr/bin/env python3
"""
Convert HDF5 robotics dataset to NVIDIA Cosmos Transfer format.

Cosmos expects:
- videos/ folder with MP4 files per episode
- annotations/ folder with JSON files containing per-frame state

Usage:
    python scripts/convert_to_cosmos.py \
        --input datasets/cosmos_generated_202.hdf5 \
        --output datasets/cosmos_transfer_format \
        --camera table_cam \
        --fps 30
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Optional
import cv2
from scipy.spatial.transform import Rotation


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]."""
    # Handle both single quaternion and array of quaternions
    if quat.ndim == 1:
        r = Rotation.from_quat(quat)  # scipy expects [x, y, z, w]
        return r.as_euler('xyz')
    else:
        eulers = []
        for q in quat:
            r = Rotation.from_quat(q)
            eulers.append(r.as_euler('xyz'))
        return np.array(eulers)


def create_video_from_frames(
    frames: np.ndarray,
    output_path: Path,
    fps: int = 30,
    codec: str = 'mp4v'
) -> bool:
    """
    Create MP4 video from numpy array of frames.

    Args:
        frames: (T, H, W, 3) uint8 array
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec (mp4v, avc1, etc.)

    Returns:
        True if successful
    """
    if len(frames) == 0:
        return False

    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    return True


def convert_hdf5_to_cosmos(
    input_path: str,
    output_dir: str,
    camera: str = 'table_cam',
    fps: int = 30,
    include_wrist: bool = True,
    max_demos: Optional[int] = None
):
    """
    Convert HDF5 dataset to Cosmos Transfer format.

    Args:
        input_path: Path to HDF5 file
        output_dir: Output directory
        camera: Primary camera to use (table_cam or wrist_cam)
        fps: Video frame rate
        include_wrist: Also export wrist camera videos
        max_demos: Maximum number of demos to convert (None for all)
    """
    output_path = Path(output_dir)
    videos_dir = output_path / 'videos'
    annotations_dir = output_path / 'annotations'

    # Create wrist camera directory if needed
    if include_wrist:
        wrist_videos_dir = output_path / 'videos_wrist'
        wrist_videos_dir.mkdir(parents=True, exist_ok=True)

    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_path} to Cosmos format...")
    print(f"Output directory: {output_path}")

    with h5py.File(input_path, 'r') as f:
        demos = sorted(f['data'].keys())

        if max_demos:
            demos = demos[:max_demos]

        total = len(demos)
        print(f"Converting {total} demonstrations...")

        metadata = {
            'num_demos': total,
            'fps': fps,
            'camera': camera,
            'source_file': str(input_path),
            'demos': []
        }

        for i, demo_name in enumerate(demos):
            demo = f['data'][demo_name]

            # Get camera frames
            cam_key = f'obs/{camera}'
            if cam_key not in demo:
                print(f"Warning: {cam_key} not found in {demo_name}, skipping")
                continue

            frames = demo[cam_key][:]
            num_frames = len(frames)

            # Get end-effector pose
            eef_pos = demo['obs/eef_pos'][:]  # (T, 3)
            eef_quat = demo['obs/eef_quat'][:]  # (T, 4)

            # Convert quaternion to euler angles
            eef_euler = quaternion_to_euler(eef_quat)  # (T, 3)

            # Get gripper state
            gripper_pos = demo['obs/gripper_pos'][:]  # (T, 2)
            gripper_width = gripper_pos[:, 0] + gripper_pos[:, 1]  # Sum of finger positions

            # Construct state as [x, y, z, roll, pitch, yaw]
            state = np.concatenate([eef_pos, eef_euler], axis=1)  # (T, 6)

            # Create video
            video_filename = f'episode_{i:04d}.mp4'
            video_path = videos_dir / video_filename

            success = create_video_from_frames(frames, video_path, fps=fps)
            if not success:
                print(f"Warning: Failed to create video for {demo_name}")
                continue

            # Create wrist camera video if requested
            if include_wrist and 'obs/wrist_cam' in demo:
                wrist_frames = demo['obs/wrist_cam'][:]
                wrist_video_path = wrist_videos_dir / video_filename
                create_video_from_frames(wrist_frames, wrist_video_path, fps=fps)

            # Create annotation
            annotation = {
                'video_path': f'videos/{video_filename}',
                'demo_name': demo_name,
                'num_frames': int(num_frames),
                'fps': fps,
                'state': state.tolist(),  # [x, y, z, roll, pitch, yaw] per frame
                'gripper_width': gripper_width.tolist(),
                'state_description': {
                    'format': '[x, y, z, roll, pitch, yaw]',
                    'position_unit': 'meters',
                    'orientation_unit': 'radians',
                    'coordinate_frame': 'world'
                }
            }

            # Optionally include additional state info
            if 'obs/joint_pos' in demo:
                annotation['joint_positions'] = demo['obs/joint_pos'][:].tolist()

            # Save annotation
            annotation_filename = f'episode_{i:04d}.json'
            annotation_path = annotations_dir / annotation_filename

            with open(annotation_path, 'w') as ann_file:
                json.dump(annotation, ann_file, indent=2)

            metadata['demos'].append({
                'index': i,
                'name': demo_name,
                'video': video_filename,
                'annotation': annotation_filename,
                'num_frames': int(num_frames)
            })

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Converted {i + 1}/{total} demos")

        # Save metadata
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=2)

        print(f"\nConversion complete!")
        print(f"  Videos: {videos_dir}")
        print(f"  Annotations: {annotations_dir}")
        print(f"  Metadata: {metadata_path}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in videos_dir.glob('*.mp4'))
        print(f"  Total video size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 robotics dataset to NVIDIA Cosmos Transfer format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for Cosmos format data'
    )
    parser.add_argument(
        '--camera', '-c',
        type=str,
        default='table_cam',
        choices=['table_cam', 'wrist_cam'],
        help='Primary camera to use (default: table_cam)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
    )
    parser.add_argument(
        '--include-wrist',
        action='store_true',
        default=True,
        help='Also export wrist camera videos'
    )
    parser.add_argument(
        '--no-wrist',
        action='store_true',
        help='Do not export wrist camera videos'
    )
    parser.add_argument(
        '--max-demos',
        type=int,
        default=None,
        help='Maximum number of demos to convert (default: all)'
    )

    args = parser.parse_args()

    include_wrist = not args.no_wrist

    convert_hdf5_to_cosmos(
        input_path=args.input,
        output_dir=args.output,
        camera=args.camera,
        fps=args.fps,
        include_wrist=include_wrist,
        max_demos=args.max_demos
    )


if __name__ == '__main__':
    main()
