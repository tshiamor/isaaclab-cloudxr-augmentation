#!/usr/bin/env bash
# =============================================================================
# LIBERO Training Setup for Cloud GPU (Brev / Shadeform)
# =============================================================================
#
# LIBERO requires PyTorch 1.11 + CUDA 11.3, which is incompatible with
# RTX 5090 (Blackwell, sm_120). This script runs LIBERO training on cloud
# instances with A100/H100/V100 GPUs.
#
# It will:
#   1. Install LIBERO and dependencies
#   2. Download your converted dataset from HuggingFace
#   3. Train BC-RNN / BC-Transformer policies
#   4. Upload trained checkpoints to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_libero_train.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET_REPO="tshiamor/block-stacking-libero"
HF_MODEL_REPO="tshiamor/block-stacking-libero-policy"
WORK_DIR="${HOME}/libero-training"
LIBERO_DIR="${WORK_DIR}/LIBERO"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/outputs"

# Training config
POLICY="bc_transformer_policy"  # bc_rnn_policy, bc_transformer_policy, bc_vilt_policy
EPOCHS=300
BATCH_SIZE=16
SEED=42

echo "============================================="
echo "LIBERO Training Pipeline"
echo "============================================="
echo "Dataset:  ${HF_DATASET_REPO}"
echo "Policy:   ${POLICY}"
echo "Epochs:   ${EPOCHS}"
echo ""

# ---- Check prerequisites ----
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "  export HF_TOKEN='your_token'"
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected."
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---- Step 1: Install Miniconda ----
echo "[Step 1/6] Setting up Python environment..."

if ! command -v conda &>/dev/null; then
    if [ -d "${HOME}/miniconda3" ]; then
        echo "  Miniconda directory exists, activating..."
    else
        echo "  Installing Miniconda..."
        INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        [ ! -f "${INSTALLER}" ] && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${INSTALLER}"
        bash "${INSTALLER}" -b -p "${HOME}/miniconda3"
    fi
fi

if [ -f "${HOME}/miniconda3/bin/conda" ]; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
fi

# Create LIBERO environment
if ! conda env list | grep -q "libero"; then
    echo "  Creating libero conda environment (Python 3.8.13)..."
    conda create -n libero python=3.8.13 -y
fi

conda activate libero
echo "  Python: $(python --version)"

# ---- Step 2: Install LIBERO ----
echo "[Step 2/6] Installing LIBERO..."
mkdir -p "${WORK_DIR}"

# Install PyTorch with CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113 -q 2>/dev/null || \
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117 -q 2>/dev/null || \
echo "  Note: Falling back to latest PyTorch (may need CUDA >= 11.7)"

# Clone and install LIBERO
if [ ! -d "${LIBERO_DIR}" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

cd "${LIBERO_DIR}"
pip install -r requirements.txt -q 2>/dev/null || true
pip install -e . -q

# Install robosuite (LIBERO's simulation backend)
pip install robosuite -q 2>/dev/null || true

# Additional dependencies
pip install h5py opencv-python huggingface_hub -q

echo "  LIBERO installed."

# ---- Step 3: Download dataset ----
echo "[Step 3/6] Downloading dataset from HuggingFace..."
pip install huggingface_hub -q

python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/libero-training/dataset"))
repo_id = os.environ.get("HF_DATASET_REPO", "tshiamor/block-stacking-libero")

print(f"Downloading {repo_id} to {dataset_dir}...")
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print(f"Downloaded to {dataset_dir}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Will check for local dataset file instead.")
PYEOF

# Find the HDF5 file
DATASET_FILE=$(find "${DATASET_DIR}" -name "*.hdf5" -type f | head -1)
if [ -z "${DATASET_FILE}" ]; then
    echo "  ERROR: No HDF5 file found in ${DATASET_DIR}"
    echo "  Upload your dataset first:"
    echo "    python scripts/convert_to_libero.py --input datasets/cosmos_generated_202.hdf5 --output datasets/libero_format/block_stacking.hdf5"
    echo "    huggingface-cli upload ${HF_DATASET_REPO} datasets/libero_format/ --repo-type dataset"
    exit 1
fi

echo "  Dataset: ${DATASET_FILE}"

# Verify dataset
python - <<VERIFYEOF
import h5py, os

f = h5py.File("${DATASET_FILE}", "r")
demos = [k for k in f["data"].keys() if k.startswith("demo_")]
demo0 = f["data/demo_0"]
obs_keys = list(demo0["obs"].keys())
print(f"  Demos: {len(demos)}")
print(f"  Obs keys: {obs_keys}")
print(f"  Actions shape: {demo0['actions'].shape}")

required = ["agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
missing = [k for k in required if k not in obs_keys]
if missing:
    print(f"  WARNING: Missing LIBERO keys: {missing}")
    print(f"  Run convert_to_libero.py first!")
else:
    print(f"  Format: OK (LIBERO-compatible)")
f.close()
VERIFYEOF

# ---- Step 4: Create training config ----
echo "[Step 4/6] Creating training configuration..."
mkdir -p "${OUTPUT_DIR}"

cat > "${WORK_DIR}/train_config.py" << 'TRAINEOF'
#!/usr/bin/env python3
"""
Train a LIBERO-compatible policy on custom block-stacking dataset.

Uses robomimic's training infrastructure directly since we have a
custom dataset (not one of LIBERO's built-in task suites).
"""

import os
import sys
import json
import h5py
import torch
import numpy as np
from pathlib import Path

# Add LIBERO to path
libero_dir = os.environ.get("LIBERO_DIR", os.path.expanduser("~/libero-training/LIBERO"))
sys.path.insert(0, libero_dir)

try:
    import robomimic
    import robomimic.utils.train_utils as TrainUtils
    from robomimic.config import config_factory
    from robomimic.scripts.train import train
    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False
    print("robomimic not found, using standalone training")

try:
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False


def create_robomimic_config(dataset_file, output_dir, policy_type="bc",
                             epochs=300, batch_size=16, seed=42):
    """Create a robomimic training config for our dataset."""

    # Read dataset to get shapes
    with h5py.File(dataset_file, "r") as f:
        demo = f["data/demo_0"]
        action_dim = demo["actions"].shape[-1]
        obs_keys = list(demo["obs"].keys())

        # Get image shape
        if "agentview_image" in obs_keys:
            img_shape = demo["obs/agentview_image"].shape[1:]
        else:
            img_shape = None

    config = {
        "algo_name": policy_type,
        "experiment": {
            "name": f"block_stacking_{policy_type}",
            "validate": True,
            "logging": {
                "terminal_output_to_txt": True,
                "log_tb": True,
            },
            "save": {
                "enabled": True,
                "every_n_epochs": 50,
                "epochs": [],
                "on_best_validation": True,
                "on_best_rollout_return": False,
                "on_best_rollout_success_rate": False,
            },
            "epoch_every_n_steps": 500,
            "validation_epoch_every_n_steps": 50,
            "rollout": {"enabled": False},
        },
        "train": {
            "data": dataset_file,
            "output_dir": output_dir,
            "num_data_workers": 4,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_normalize_obs": False,
            "batch_size": batch_size,
            "num_epochs": epochs,
            "seed": seed,
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": ["agentview_image"],
                },
            },
            "encoder": {
                "rgb": {
                    "core_class": "VisualCore",
                    "core_kwargs": {
                        "feature_dimension": 64,
                        "backbone_class": "ResNet18Conv",
                        "backbone_kwargs": {
                            "pretrained": False,
                            "input_coord_conv": False,
                        },
                        "pool_class": "SpatialSoftmax",
                        "pool_kwargs": {
                            "num_kp": 32,
                            "learnable_temperature": False,
                            "temperature": 1.0,
                            "noise_std": 0.0,
                        },
                    },
                },
            },
        },
    }

    # Add wrist camera if available
    if "robot0_eye_in_hand_image" in obs_keys:
        config["observation"]["modalities"]["obs"]["rgb"].append(
            "robot0_eye_in_hand_image"
        )

    # Policy-specific config
    if policy_type == "bc":
        config["algo"] = {
            "optim_params": {
                "policy": {
                    "learning_rate": {"initial": 1e-4, "decay_factor": 0.1},
                    "regularization": {"L2": 0.0},
                }
            },
            "rnn": {
                "enabled": True,
                "horizon": 10,
                "hidden_dim": 400,
                "rnn_type": "LSTM",
                "num_layers": 2,
            },
            "gmm": {
                "enabled": True,
                "num_modes": 5,
                "min_std": 0.0001,
            },
        }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {config_path}")
    return config_path


def train_standalone(dataset_file, output_dir, epochs=300, batch_size=16, seed=42):
    """Standalone training using PyTorch directly (fallback if robomimic unavailable)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class DemoDataset(Dataset):
        def __init__(self, hdf5_path, split="train"):
            self.f = h5py.File(hdf5_path, "r", swmr=True)
            if "mask" in self.f and split in self.f["mask"]:
                demo_keys = [d.decode() if isinstance(d, bytes) else d
                             for d in self.f["mask"][split][:]]
            else:
                all_demos = sorted(
                    [k for k in self.f["data"].keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                n = int(len(all_demos) * 0.9)
                demo_keys = [f"data/{d}" for d in (all_demos[:n] if split == "train" else all_demos[n:])]

            self.samples = []
            for dk in demo_keys:
                demo = self.f[dk] if dk.startswith("data/") else self.f[f"data/{dk}"]
                T = demo["actions"].shape[0]
                for t in range(T):
                    self.samples.append((dk if dk.startswith("data/") else f"data/{dk}", t))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            dk, t = self.samples[idx]
            demo = self.f[dk]

            # Image observation
            img = demo["obs/agentview_image"][t]  # (H, W, 3) uint8
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # Low-dim observations
            eef_pos = demo["obs/robot0_eef_pos"][t]
            eef_quat = demo["obs/robot0_eef_quat"][t]
            gripper = demo["obs/robot0_gripper_qpos"][t]
            low_dim = np.concatenate([eef_pos, eef_quat, gripper])
            low_dim = torch.from_numpy(low_dim).float()

            # Action
            action = torch.from_numpy(demo["actions"][t]).float()

            return {"image": img, "low_dim": low_dim, "action": action}

    class BCPolicy(nn.Module):
        def __init__(self, img_size, low_dim_size, action_dim):
            super().__init__()
            # Simple CNN encoder
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
            )
            cnn_out = 128 * 4 * 4
            self.mlp = nn.Sequential(
                nn.Linear(cnn_out + low_dim_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh(),
            )

        def forward(self, image, low_dim):
            feat = self.cnn(image)
            x = torch.cat([feat, low_dim], dim=-1)
            return self.mlp(x)

    print("Using standalone PyTorch training (robomimic not available)")
    torch.manual_seed(seed)

    train_dataset = DemoDataset(dataset_file, "train")
    val_dataset = DemoDataset(dataset_file, "valid")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Get dimensions from first sample
    sample = train_dataset[0]
    img_size = sample["image"].shape[-1]
    low_dim_size = sample["low_dim"].shape[0]
    action_dim = sample["action"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(img_size, low_dim_size, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            img = batch["image"].to(device)
            low_dim = batch["low_dim"].to(device)
            action = batch["action"].to(device)

            pred = model(img, low_dim)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                low_dim = batch["low_dim"].to(device)
                action = batch["action"].to(device)
                pred = model(img, low_dim)
                val_loss += criterion(pred, action).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {output_dir}")


def main():
    dataset_file = os.environ.get("DATASET_FILE", "")
    output_dir = os.environ.get("OUTPUT_DIR", os.path.expanduser("~/libero-training/outputs"))
    epochs = int(os.environ.get("EPOCHS", "300"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    seed = int(os.environ.get("SEED", "42"))
    policy_type = os.environ.get("POLICY", "bc_transformer_policy")

    os.makedirs(output_dir, exist_ok=True)

    if HAS_ROBOMIMIC:
        print("Using robomimic training pipeline")
        config_path = create_robomimic_config(
            dataset_file, output_dir,
            policy_type="bc",
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        # Run robomimic training
        train(config_path)
    else:
        train_standalone(dataset_file, output_dir, epochs, batch_size, seed)


if __name__ == "__main__":
    main()
TRAINEOF

# ---- Step 5: Train policy ----
echo "[Step 5/6] Training policy..."
echo "  Policy: ${POLICY}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo ""

cd "${LIBERO_DIR}"

DATASET_FILE="${DATASET_FILE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EPOCHS="${EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
SEED="${SEED}" \
POLICY="${POLICY}" \
LIBERO_DIR="${LIBERO_DIR}" \
    python "${WORK_DIR}/train_config.py"

echo "  Training complete."
echo "  Checkpoints: ${OUTPUT_DIR}"

# ---- Step 6: Upload results ----
echo "[Step 6/6] Uploading checkpoints to HuggingFace..."

python - <<'UPLOADEOF'
import os
from huggingface_hub import HfApi

output_dir = os.environ.get("OUTPUT_DIR", os.path.expanduser("~/libero-training/outputs"))
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/block-stacking-libero-policy")

api = HfApi()

try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

print(f"Uploading {output_dir} to {repo_id}...")
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload LIBERO-trained policy checkpoints",
)
print(f"Upload complete: https://huggingface.co/tshiamor/{repo_id}")
UPLOADEOF

echo ""
echo "============================================="
echo "PIPELINE COMPLETE"
echo "============================================="
echo "Checkpoints: https://huggingface.co/${HF_MODEL_REPO}"
echo "============================================="
