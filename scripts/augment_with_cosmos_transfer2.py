#!/usr/bin/env python3
"""
Augment block-stacking demos using Cosmos-Transfer2.5

This script generates multiple augmented versions of each original demo
by applying different text prompts while preserving the action structure
through edge/depth control.

Usage (inside cosmos-transfer2.5 Docker container):
    python /data/augment_with_cosmos_transfer2.py \
        --input /data/input \
        --output /data/output \
        --num_augmentations 4

Requirements:
    - cosmos-transfer2.5 environment
    - GPU with sufficient VRAM (65GB for single-GPU, or multi-GPU setup)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Augmentation prompts for block stacking robot task
AUGMENTATION_PROMPTS = [
    # Environment variations
    "A robot arm in a modern laboratory setting with bright LED lighting, stacking wooden blocks on a white table. Clean industrial environment with equipment visible in the background.",
    "A robotic gripper in a dimly lit warehouse, carefully placing colorful building blocks into a container. Concrete floors and metal shelving visible.",
    "A precise robot manipulator in a sunlit workshop, arranging wooden cubes on a workbench. Natural light streaming through windows, tools and materials around.",
    "A robotic system in a futuristic factory setting with blue ambient lighting, handling geometric blocks. High-tech monitors and displays in the background.",
    # Style variations
    "A mechanical arm in an artist's studio, gently moving painted wooden blocks. Colorful paint splashes on walls, creative chaos surrounding the scene.",
    "A robot arm on a production line, efficiently sorting and stacking product containers. Industrial conveyor belts and safety markings visible.",
    "A delicate robotic hand in a clean room environment, precisely placing sensor components. White walls, particle filters, and sterile equipment around.",
    "A collaborative robot in a home kitchen, organizing food storage containers on the counter. Warm domestic lighting, wooden cabinets visible.",
]


def create_spec_file(video_path: str, prompt: str, output_name: str, output_dir: Path) -> Path:
    """Create a JSON spec file for cosmos-transfer2.5 inference."""
    spec = {
        "name": output_name,
        "prompt": prompt,
        "video_path": video_path,
        "guidance": 3,
        "seed": 2025,
        "edge": {
            "control_weight": 0.9
        },
        "depth": {
            "control_weight": 0.5
        }
    }

    spec_path = output_dir / f"{output_name}_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    return spec_path


def run_inference(spec_file: Path, output_dir: Path, cosmos_repo: Path):
    """Run cosmos-transfer2.5 inference on a spec file."""
    cmd = [
        sys.executable,
        str(cosmos_repo / "examples" / "inference.py"),
        "-i", str(spec_file),
        "-o", str(output_dir),
        "--disable-guardrails"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    return True


def copy_annotation(src_annotation_path: Path, dst_annotation_path: Path, new_video_name: str):
    """Copy annotation file with updated video reference."""
    with open(src_annotation_path, "r") as f:
        annotation = json.load(f)

    # Update video filename reference
    if "video_file" in annotation:
        annotation["video_file"] = new_video_name

    with open(dst_annotation_path, "w") as f:
        json.dump(annotation, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Augment demos with Cosmos-Transfer2.5")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory with videos and annotations")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for augmented dataset")
    parser.add_argument("--num_augmentations", type=int, default=4,
                       help="Number of augmented versions per original demo")
    parser.add_argument("--cosmos_repo", type=str, default="/workspace",
                       help="Path to cosmos-transfer2.5 repository")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    cosmos_repo = Path(args.cosmos_repo)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_out = output_dir / "videos"
    annotations_out = output_dir / "annotations"
    specs_dir = output_dir / "specs"
    videos_out.mkdir(exist_ok=True)
    annotations_out.mkdir(exist_ok=True)
    specs_dir.mkdir(exist_ok=True)

    # Find all input videos
    videos_dir = input_dir / "videos"
    annotations_dir = input_dir / "annotations"

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos to augment")

    total_demos = 0

    for i, video_path in enumerate(video_files):
        demo_name = video_path.stem
        annotation_path = annotations_dir / f"{demo_name}.json"

        print(f"\n[{i+1}/{len(video_files)}] Processing {demo_name}")

        # Copy original video and annotation
        shutil.copy(video_path, videos_out / f"{demo_name}.mp4")
        if annotation_path.exists():
            shutil.copy(annotation_path, annotations_out / f"{demo_name}.json")
        total_demos += 1

        # Generate augmented versions
        for aug_idx in range(args.num_augmentations):
            prompt = AUGMENTATION_PROMPTS[aug_idx % len(AUGMENTATION_PROMPTS)]
            aug_name = f"{demo_name}_aug{aug_idx}"

            print(f"  Generating augmentation {aug_idx + 1}/{args.num_augmentations}: {aug_name}")

            # Create spec file
            spec_file = create_spec_file(
                video_path=str(video_path),
                prompt=prompt,
                output_name=aug_name,
                output_dir=specs_dir
            )

            # Run inference
            temp_out = output_dir / "temp_inference"
            temp_out.mkdir(exist_ok=True)

            if run_inference(spec_file, temp_out, cosmos_repo):
                # Move generated video to output
                generated_video = temp_out / aug_name / f"{aug_name}.mp4"
                if generated_video.exists():
                    shutil.move(str(generated_video), videos_out / f"{aug_name}.mp4")

                    # Copy annotation with updated reference
                    if annotation_path.exists():
                        copy_annotation(
                            annotation_path,
                            annotations_out / f"{aug_name}.json",
                            f"{aug_name}.mp4"
                        )

                    total_demos += 1
                    print(f"    Generated {aug_name}.mp4")
                else:
                    print(f"    Warning: Generated video not found at {generated_video}")
            else:
                print(f"    Warning: Inference failed for {aug_name}")

            # Cleanup temp directory
            shutil.rmtree(temp_out, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Augmentation complete!")
    print(f"Original demos: {len(video_files)}")
    print(f"Total demos after augmentation: {total_demos}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
