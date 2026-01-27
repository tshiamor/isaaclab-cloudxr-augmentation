#!/usr/bin/env python3
"""
Augment robotics dataset using NVIDIA Cosmos-Transfer1.

This script takes the cosmos_format dataset (202 demos) and generates
augmented variations using Cosmos-Transfer1's edge/depth ControlNet
to create visually diverse training data while preserving the robot motions.

Usage:
    # Augment to 1000 demos (5x augmentation)
    python scripts/augment_with_cosmos_transfer.py \
        --input datasets/cosmos_format \
        --output datasets/cosmos_augmented_1000 \
        --num_augmentations 4 \
        --checkpoint_dir ~/cosmos-transfer1/checkpoints

Requirements:
    - Cosmos-Transfer1 checkpoints downloaded
    - cosmos-transfer1 conda environment activated
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import random


# Prompts for domain randomization - different visual styles
AUGMENTATION_PROMPTS = [
    "A robotic arm performing precise manipulation in a modern laboratory with bright LED lighting and clean white surfaces.",
    "A robotic gripper handling objects in an industrial warehouse setting with metal shelving and natural daylight from windows.",
    "A robot arm working in a home kitchen environment with wooden countertops and warm ambient lighting.",
    "A robotic manipulator in a research facility with blue accent lighting and high-tech equipment in the background.",
    "A robot performing a task in a workshop setting with concrete floors and fluorescent overhead lighting.",
    "A robotic arm operating in a cleanroom environment with soft diffused lighting and sterile white walls.",
    "A robot gripper manipulating objects in an office environment with desk lamps and computer monitors visible.",
    "A robotic system working in a manufacturing plant with conveyor belts and industrial equipment nearby.",
]


def create_controlnet_spec(
    input_video: str,
    prompt: str,
    output_name: str,
    control_type: str = "edge",
    control_weight: float = 1.0
) -> dict:
    """Create a ControlNet specification for Cosmos-Transfer1."""
    spec = {
        "prompt": prompt,
        "input_video_path": input_video,
        output_name: output_name,
    }

    # Add control type configuration
    spec[control_type] = {
        "control_weight": control_weight
    }

    return spec


def run_cosmos_transfer(
    checkpoint_dir: str,
    input_video: str,
    output_video: str,
    prompt: str,
    control_type: str = "edge",
    control_weight: float = 1.0,
    use_distilled: bool = True,
    cosmos_repo: str = "~/cosmos-transfer1"
) -> bool:
    """Run Cosmos-Transfer1 inference using custom no-guardrail script."""
    cosmos_repo = os.path.expanduser(cosmos_repo)
    script_dir = Path(__file__).parent

    # Use our custom inference script that disables guardrail
    inference_script = script_dir / "cosmos_inference_no_guardrail.py"

    cmd = [
        sys.executable,
        str(inference_script),
        "--checkpoint_dir", checkpoint_dir,
        "--input_video", input_video,
        "--output_video", output_video,
        "--prompt", prompt,
        "--control_type", control_type,
        "--control_weight", str(control_weight),
    ]

    if use_distilled:
        cmd.append("--use_distilled")

    env = os.environ.copy()
    env["PYTHONPATH"] = cosmos_repo

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per video
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Timeout during generation")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def augment_dataset(
    input_dir: str,
    output_dir: str,
    checkpoint_dir: str,
    num_augmentations: int = 4,
    control_type: str = "edge",
    use_distilled: bool = True,
    start_idx: int = 0,
    max_demos: Optional[int] = None,
    cosmos_repo: str = "~/cosmos-transfer1"
):
    """
    Augment the dataset using Cosmos-Transfer1.

    Args:
        input_dir: Input cosmos_format directory
        output_dir: Output directory for augmented data
        checkpoint_dir: Path to Cosmos-Transfer1 checkpoints
        num_augmentations: Number of augmented versions per original video
        control_type: Type of control (edge, depth, seg)
        use_distilled: Use distilled model for faster inference
        start_idx: Starting demo index (for resuming)
        max_demos: Maximum demos to process
        cosmos_repo: Path to cosmos-transfer1 repository
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    cosmos_repo = os.path.expanduser(cosmos_repo)
    checkpoint_dir = os.path.expanduser(checkpoint_dir)

    # Create output directories
    (output_path / "videos").mkdir(parents=True, exist_ok=True)
    (output_path / "videos_wrist").mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(parents=True, exist_ok=True)

    # Get list of input videos
    input_videos = sorted(list((input_path / "videos").glob("*.mp4")))

    if max_demos:
        input_videos = input_videos[:max_demos]

    print(f"Found {len(input_videos)} input videos")
    print(f"Will generate {num_augmentations} augmentations each")
    print(f"Total output: {len(input_videos)} original + {len(input_videos) * num_augmentations} augmented = {len(input_videos) * (num_augmentations + 1)} demos")

    # First, copy original videos and annotations
    print("\nCopying original videos...")
    for i, video_path in enumerate(input_videos):
        video_name = video_path.name

        # Copy original video
        shutil.copy(video_path, output_path / "videos" / video_name)

        # Copy wrist video if exists
        wrist_video = input_path / "videos_wrist" / video_name
        if wrist_video.exists():
            shutil.copy(wrist_video, output_path / "videos_wrist" / video_name)

        # Copy annotation
        ann_name = video_path.stem + ".json"
        ann_path = input_path / "annotations" / ann_name
        if ann_path.exists():
            shutil.copy(ann_path, output_path / "annotations" / ann_name)

    print(f"Copied {len(input_videos)} original demos")

    # Generate augmented versions
    print(f"\nGenerating augmented versions using {control_type} control...")

    augmented_count = 0
    total_augmentations = len(input_videos) * num_augmentations

    for video_idx, video_path in enumerate(input_videos):
        if video_idx < start_idx:
            continue

        video_name = video_path.stem
        original_ann_path = input_path / "annotations" / f"{video_name}.json"

        # Load original annotation
        with open(original_ann_path) as f:
            original_ann = json.load(f)

        for aug_idx in range(num_augmentations):
            aug_name = f"{video_name}_aug{aug_idx:02d}"

            # Select random prompt for visual diversity
            prompt = random.choice(AUGMENTATION_PROMPTS)

            # Output video path
            output_video_path = output_path / "videos" / f"{aug_name}.mp4"

            print(f"\n[{augmented_count + 1}/{total_augmentations}] Generating {aug_name}...")

            success = run_cosmos_transfer(
                checkpoint_dir=checkpoint_dir,
                input_video=str(video_path.absolute()),
                output_video=str(output_video_path),
                prompt=prompt,
                control_type=control_type,
                control_weight=1.0,
                use_distilled=use_distilled,
                cosmos_repo=cosmos_repo
            )

            if success and output_video_path.exists():
                # Create annotation for augmented video (same state/actions)
                aug_ann = original_ann.copy()
                aug_ann["video_path"] = f"videos/{aug_name}.mp4"
                aug_ann["demo_name"] = aug_name
                aug_ann["augmentation"] = {
                    "source": video_name,
                    "method": f"cosmos-transfer1-{control_type}",
                    "prompt": prompt,
                    "aug_index": aug_idx
                }

                with open(output_path / "annotations" / f"{aug_name}.json", 'w') as f:
                    json.dump(aug_ann, f, indent=2)

                augmented_count += 1
                print(f"  Success! ({augmented_count}/{total_augmentations})")
            else:
                print(f"  Failed to generate augmentation")

    # Create metadata
    metadata = {
        "num_original": len(input_videos),
        "num_augmented": augmented_count,
        "total_demos": len(input_videos) + augmented_count,
        "augmentation_method": f"cosmos-transfer1-{control_type}",
        "source_dataset": str(input_path.absolute()),
        "prompts_used": AUGMENTATION_PROMPTS
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Augmentation complete!")
    print(f"  Original demos: {len(input_videos)}")
    print(f"  Augmented demos: {augmented_count}")
    print(f"  Total demos: {len(input_videos) + augmented_count}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='Augment robotics dataset using Cosmos-Transfer1'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input cosmos_format directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for augmented data'
    )
    parser.add_argument(
        '--checkpoint_dir', '-c',
        type=str,
        default='~/cosmos-transfer1/checkpoints',
        help='Path to Cosmos-Transfer1 checkpoints'
    )
    parser.add_argument(
        '--num_augmentations', '-n',
        type=int,
        default=4,
        help='Number of augmented versions per original (default: 4)'
    )
    parser.add_argument(
        '--control_type',
        type=str,
        default='edge',
        choices=['edge', 'depth', 'seg'],
        help='Control type for augmentation (default: edge)'
    )
    parser.add_argument(
        '--use_distilled',
        action='store_true',
        default=True,
        help='Use distilled model for faster inference (default: True)'
    )
    parser.add_argument(
        '--no_distilled',
        action='store_true',
        help='Use full model instead of distilled'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting demo index for resuming (default: 0)'
    )
    parser.add_argument(
        '--max_demos',
        type=int,
        default=None,
        help='Maximum demos to process (default: all)'
    )
    parser.add_argument(
        '--cosmos_repo',
        type=str,
        default='~/cosmos-transfer1',
        help='Path to cosmos-transfer1 repository'
    )

    args = parser.parse_args()

    # Handle distilled flag
    use_distilled = not args.no_distilled

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint_dir=args.checkpoint_dir,
        num_augmentations=args.num_augmentations,
        control_type=args.control_type,
        use_distilled=use_distilled,
        start_idx=args.start_idx,
        max_demos=args.max_demos,
        cosmos_repo=args.cosmos_repo
    )


if __name__ == '__main__':
    main()
