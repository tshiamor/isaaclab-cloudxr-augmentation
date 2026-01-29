#!/usr/bin/env bash
# =============================================================================
# GR00T N1.6 Training Script for Franka Block-Stacking
# =============================================================================
#
# This script trains GR00T N1.6 on your block-stacking dataset.
#
# Expected time: ~2.5-4 hours with batch_size=1, 30K steps on RTX 5090
#
# Usage:
#   bash scripts/run_groot_training.sh
#
# =============================================================================

set -euo pipefail

GROOT_DIR="${HOME}/Isaac-GR00T"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_SRC="${SCRIPT_DIR}/../datasets/groot_format"
DATASET_DST="${SCRIPT_DIR}/../datasets/groot_n1d6_format"
OUTPUT_DIR="${HOME}/groot_training_output"

MAX_STEPS=30000
BATCH_SIZE=1
SAVE_STEPS=5000

echo "============================================="
echo "GR00T N1.6 Training Pipeline"
echo "============================================="
echo "Dataset:    ${DATASET_SRC}"
echo "Output:     ${OUTPUT_DIR}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Estimated:  ~2.5-4 hours"
echo "============================================="
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check GROOT repo
if [ ! -d "${GROOT_DIR}" ]; then
    echo "ERROR: Isaac-GR00T not found at ${GROOT_DIR}"
    echo "Clone it with: git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T"
    exit 1
fi

# Setup uv environment
echo "[Step 1/5] Setting up GR00T environment..."
cd "${GROOT_DIR}"

if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "  Syncing dependencies (this may download ~10GB of models on first run)..."
uv sync --python 3.10 2>&1 | tail -5

# Prepare dataset
echo ""
echo "[Step 2/5] Preparing dataset for GR00T N1.6 format..."

# Run the Python conversion
uv run python - << 'CONVERT_EOF'
import json
import shutil
from pathlib import Path

DATASET_SRC = Path.home() / "simulations/franka/isaaclab_cloudxr/datasets/groot_format"
DATASET_DST = Path.home() / "simulations/franka/isaaclab_cloudxr/datasets/groot_n1d6_format"

print(f"  Source: {DATASET_SRC}")
print(f"  Destination: {DATASET_DST}")

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
modality_config = {
    "state": {
        "single_arm": {"start": 0, "end": 9},
        "gripper": {"start": 9, "end": 11},
    },
    "action": {
        "single_arm": {"start": 0, "end": 7},
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
tasks_path = DATASET_DST / "meta" / "tasks.jsonl"
with open(tasks_path, "w") as f:
    f.write('{"task_index": 0, "task": "Stack the colored blocks into the bowl"}\n')
print(f"  Created tasks.jsonl")

# Update parquet files
import pyarrow as pa
import pyarrow.parquet as pq

data_dir = DATASET_DST / "data" / "chunk-000"
for parquet_file in data_dir.glob("*.parquet"):
    table = pq.read_table(parquet_file)
    columns = table.column_names

    if "annotation.human.action.task_description" not in columns:
        n_rows = table.num_rows
        new_columns = {col: table.column(col) for col in columns}
        new_columns["annotation.human.action.task_description"] = pa.array([0] * n_rows, type=pa.int64())
        new_columns["task_index"] = pa.array([0] * n_rows, type=pa.int64())
        if "language_instruction" in new_columns:
            del new_columns["language_instruction"]
        new_table = pa.table(new_columns)
        pq.write_table(new_table, parquet_file)

print(f"  Updated parquet with annotation columns")

# Rename video directories
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

print(f"\n  Dataset ready!")
CONVERT_EOF

# Create modality config
echo ""
echo "[Step 3/5] Creating modality config..."

CONFIG_DIR="${DATASET_DST}/config"
mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/franka_config.py" << 'CONFIG_EOF'
"""Franka Panda modality configuration for GR00T N1.6."""

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
from gr00t.configs.data.embodiment_configs import register_modality_config

franka_block_stacking_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm"],
        action_configs=[
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

register_modality_config(franka_block_stacking_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
CONFIG_EOF

echo "  Created ${CONFIG_DIR}/franka_config.py"

# Run training
echo ""
echo "[Step 4/5] Starting GR00T N1.6 fine-tuning..."
echo "  This will take approximately 2.5-4 hours"
echo ""

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path "${DATASET_DST}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${CONFIG_DIR}/franka_config.py" \
    --num-gpus 1 \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps ${SAVE_STEPS} \
    --max-steps ${MAX_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo ""
echo "[Step 5/5] Uploading to HuggingFace..."

uv run python - << 'UPLOAD_EOF'
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi

    output_dir = Path.home() / "groot_training_output"
    repo_id = "tshiamor/franka-block-stacking-groot-n1d6"

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        print("  No checkpoints found!")
        exit(1)

    final_checkpoint = checkpoints[-1]
    print(f"  Uploading {final_checkpoint.name} to {repo_id}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(final_checkpoint),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload GR00T N1.6 fine-tuned on Franka block-stacking (30K steps)",
    )
    print(f"\n  Uploaded to: https://huggingface.co/{repo_id}")
except ImportError:
    print("  huggingface_hub not installed, skipping upload")
except Exception as e:
    print(f"  Upload failed: {e}")
UPLOAD_EOF

echo ""
echo "============================================="
echo "Training Complete!"
echo "============================================="
echo "Checkpoints: ${OUTPUT_DIR}"
echo "HuggingFace: https://huggingface.co/tshiamor/franka-block-stacking-groot-n1d6"
echo "============================================="
