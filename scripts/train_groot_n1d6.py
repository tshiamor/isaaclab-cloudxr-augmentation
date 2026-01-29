#!/usr/bin/env python3
"""
Train GR00T N1.6 on Franka block-stacking dataset.

This script:
1. Converts existing GROOT format data to N1.6 compatible format
2. Creates modality config for Franka robot
3. Runs fine-tuning for 30,000 steps
4. Optionally uploads to HuggingFace

Usage:
    cd ~/Isaac-GR00T
    uv run python ~/simulations/franka/isaaclab_cloudxr/scripts/train_groot_n1d6.py

Requirements:
    - Isaac-GR00T repo with uv environment
    - Dataset at ~/simulations/franka/isaaclab_cloudxr/datasets/groot_format
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
DATASET_SRC = Path.home() / "simulations/franka/isaaclab_cloudxr/datasets/groot_format"
DATASET_DST = Path.home() / "simulations/franka/isaaclab_cloudxr/datasets/groot_n1d6_format"
GROOT_REPO = Path.home() / "Isaac-GR00T"
OUTPUT_DIR = Path.home() / "groot_training_output"
HF_MODEL_REPO = "tshiamor/franka-block-stacking-groot-n1d6"

MAX_STEPS = 30000
BATCH_SIZE = 1  # User requested batch_size=1
SAVE_STEPS = 5000


def convert_to_groot_n1d6_format():
    """Convert existing GROOT format to N1.6 compatible format."""
    print("=" * 60)
    print("Step 1: Converting dataset to GR00T N1.6 format")
    print("=" * 60)

    # Create output directory
    DATASET_DST.mkdir(parents=True, exist_ok=True)

    # Copy existing structure
    for subdir in ["data", "videos", "meta"]:
        src = DATASET_SRC / subdir
        dst = DATASET_DST / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {subdir}/")

    # Create modality.json
    # state = [joint_pos(9) + gripper_pos(2)] = 11 dims
    # action = 7 dims (joint velocity)
    modality_config = {
        "state": {
            "single_arm": {"start": 0, "end": 9},  # joint positions
            "gripper": {"start": 9, "end": 11},  # gripper position
        },
        "action": {
            "single_arm": {"start": 0, "end": 7},  # joint velocity actions
        },
        "video": {
            "front": {"original_key": "observation.images.table_cam"},
            "wrist": {"original_key": "observation.images.wrist_cam"},
        },
        "annotation": {
            "annotation.human.action.task_description": {},
        },
    }

    modality_path = DATASET_DST / "meta" / "modality.json"
    with open(modality_path, "w") as f:
        json.dump(modality_config, f, indent=2)
    print(f"  Created modality.json")

    # Create tasks.jsonl
    tasks = [
        {"task_index": 0, "task": "Stack the colored blocks into the bowl"},
    ]
    tasks_path = DATASET_DST / "meta" / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"  Created tasks.jsonl")

    # Update parquet files to add annotation column
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        data_dir = DATASET_DST / "data" / "chunk-000"
        for parquet_file in data_dir.glob("*.parquet"):
            table = pq.read_table(parquet_file)
            columns = table.column_names

            # Check if we need to add annotation column
            if "annotation.human.action.task_description" not in columns:
                # Add task_index column (all 0 since single task)
                n_rows = table.num_rows
                task_indices = pa.array([0] * n_rows, type=pa.int64())

                # Create new table with annotation column
                new_columns = {col: table.column(col) for col in columns}
                new_columns["annotation.human.action.task_description"] = task_indices
                new_columns["task_index"] = task_indices

                # Remove old language_instruction if present
                if "language_instruction" in new_columns:
                    del new_columns["language_instruction"]

                new_table = pa.table(new_columns)
                pq.write_table(new_table, parquet_file)

        print(f"  Updated parquet files with annotation columns")
    except ImportError:
        print("  WARNING: pyarrow not available, skipping parquet update")
        print("  Install with: pip install pyarrow")

    # Rename video directories if needed
    videos_dir = DATASET_DST / "videos"
    for old_name, new_name in [
        ("observation.images.table_cam", "observation.images.front"),
        ("observation.images.wrist_cam", "observation.images.wrist"),
    ]:
        old_path = videos_dir / old_name
        new_path = videos_dir / new_name
        if old_path.exists() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"  Renamed {old_name} -> {new_name}")

    print(f"\n  Dataset ready at: {DATASET_DST}")
    return DATASET_DST


def create_modality_config():
    """Create the Python modality config file for Franka robot."""
    print("\n" + "=" * 60)
    print("Step 2: Creating modality config for Franka robot")
    print("=" * 60)

    config_content = '''"""
Franka Panda modality configuration for GR00T N1.6 fine-tuning.

This config defines:
- Video inputs: front camera (third-person) and wrist camera
- State: 9 joint positions + 2 gripper positions = 11 dims
- Action: 7 joint velocity commands, 16 action horizon
"""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
from gr00t.configs.data.embodiment_configs import register_modality_config


franka_block_stacking_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "front",   # table/third-person camera
            "wrist",   # wrist-mounted camera
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # 9 joint positions
            "gripper",     # 2 gripper positions
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon
        modality_keys=[
            "single_arm",  # 7 joint velocity commands
        ],
        action_configs=[
            # single_arm - joint velocity control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Register under NEW_EMBODIMENT tag
register_modality_config(franka_block_stacking_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''

    config_dir = DATASET_DST / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "franka_config.py"

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"  Created {config_path}")
    return config_path


def run_training(dataset_path: Path, config_path: Path):
    """Run GR00T N1.6 fine-tuning."""
    print("\n" + "=" * 60)
    print("Step 3: Running GR00T N1.6 fine-tuning")
    print("=" * 60)
    print(f"  Dataset: {dataset_path}")
    print(f"  Config: {config_path}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the training command
    cmd = [
        "python",
        str(GROOT_REPO / "gr00t" / "experiment" / "launch_finetune.py"),
        "--base-model-path", "nvidia/GR00T-N1.6-3B",
        "--dataset-path", str(dataset_path),
        "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(config_path),
        "--num-gpus", "1",
        "--output-dir", str(OUTPUT_DIR),
        "--save-total-limit", "5",
        "--save-steps", str(SAVE_STEPS),
        "--max-steps", str(MAX_STEPS),
        "--global-batch-size", str(BATCH_SIZE),
        "--dataloader-num-workers", "4",
        # Color jitter for data augmentation
        "--color-jitter-params", "brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08",
    ]

    print("\n  Command:")
    print("  " + " \\\n    ".join(cmd[:5]) + " \\")
    print("    " + " \\\n    ".join(cmd[5:]))
    print()

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Run training
    os.chdir(GROOT_REPO)
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\n  ERROR: Training failed with exit code {result.returncode}")
        return False

    print(f"\n  Training complete! Checkpoints saved to {OUTPUT_DIR}")
    return True


def upload_to_huggingface():
    """Upload trained model to HuggingFace."""
    print("\n" + "=" * 60)
    print("Step 4: Uploading to HuggingFace")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Find the best/final checkpoint
        checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
        if not checkpoints:
            print("  No checkpoints found!")
            return False

        final_checkpoint = checkpoints[-1]
        print(f"  Uploading {final_checkpoint} to {HF_MODEL_REPO}")

        # Create repo if needed
        try:
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        # Upload
        api.upload_folder(
            folder_path=str(final_checkpoint),
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message=f"Upload GR00T N1.6 fine-tuned on Franka block-stacking ({MAX_STEPS} steps)",
        )

        print(f"\n  Uploaded to: https://huggingface.co/{HF_MODEL_REPO}")
        return True

    except ImportError:
        print("  huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  Upload failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("GR00T N1.6 Training Pipeline")
    print("=" * 60)
    print(f"Source dataset: {DATASET_SRC}")
    print(f"Target dataset: {DATASET_DST}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated time: 2.5-4 hours (batch_size=1, 30K steps)")
    print("=" * 60)

    # Step 1: Convert dataset
    dataset_path = convert_to_groot_n1d6_format()

    # Step 2: Create modality config
    config_path = create_modality_config()

    # Step 3: Run training
    success = run_training(dataset_path, config_path)

    if success:
        # Step 4: Upload to HuggingFace
        upload_to_huggingface()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
