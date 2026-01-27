#!/usr/bin/env bash
# =============================================================================
# Cosmos-Transfer2.5 Augmentation Setup for NVIDIA Brev
# =============================================================================
#
# Run this script on an NVIDIA Brev instance with H100/A100 GPU.
# It will:
#   1. Clone cosmos-transfer2.5
#   2. Pull the dataset from HuggingFace (tshiamor/block-stacking-cosmos-transfer)
#   3. Build the Docker container
#   4. Run augmentation (202 demos -> 1000+ demos)
#   5. Upload results to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_cosmos_augment.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/block-stacking-cosmos-transfer"
HF_OUTPUT_REPO="tshiamor/block-stacking-cosmos-augmented-1000"
NUM_AUGMENTATIONS=4
WORK_DIR="${HOME}/cosmos-augmentation"
COSMOS_REPO="${WORK_DIR}/cosmos-transfer2.5"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/augmented_output"

echo "============================================="
echo "Cosmos-Transfer2.5 Augmentation Pipeline"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Output:  ${HF_OUTPUT_REPO}"
echo "Augmentations per demo: ${NUM_AUGMENTATIONS}"
echo ""

# ---- Check prerequisites ----
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "  export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- Step 1: Install system dependencies + conda ----
echo "[Step 1/7] Installing system dependencies..."
sudo apt-get update -qq || echo "  Warning: apt-get update had errors (non-critical, continuing)"
sudo apt-get install -y -qq git git-lfs curl ffmpeg wget > /dev/null 2>&1 || true
git lfs install

# Install Docker if not present
if ! command -v docker &>/dev/null; then
    echo "  Installing Docker..."
    sudo apt-get install -y -qq docker.io > /dev/null 2>&1 || true
fi

# Install Miniconda if no usable python/pip is available
if ! command -v conda &>/dev/null; then
    echo "  Installing Miniconda..."
    CONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "${CONDA_INSTALLER}" ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${CONDA_INSTALLER}"
    fi
    bash "${CONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    echo "  Miniconda installed to ${HOME}/miniconda3"
fi

# Activate conda in current shell
if [ -f "${HOME}/miniconda3/bin/conda" ]; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    echo "  Conda version: $(conda --version)"
fi

# Ensure pip is available (use conda's pip or system pip)
if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo "  Installing pip via conda..."
    conda install -y pip > /dev/null 2>&1
fi

# Use pip3 if pip is not available
if ! command -v pip &>/dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

# Use python3 if python is not available
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "  Python: $(${PYTHON_CMD} --version)"
echo "  Pip: $(${PIP_CMD} --version)"

# Ensure NVIDIA Container Toolkit is available
if ! sudo docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null 2>&1 || true
    sudo nvidia-ctk runtime configure --runtime=docker || true
    sudo systemctl restart docker || true
fi

# ---- Step 2: Clone Cosmos-Transfer2.5 ----
echo "[Step 2/7] Cloning Cosmos-Transfer2.5..."
mkdir -p "${WORK_DIR}"

if [ ! -d "${COSMOS_REPO}" ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "${COSMOS_REPO}"
    cd "${COSMOS_REPO}"
    git lfs pull
else
    echo "  Already cloned."
fi

# ---- Step 3: Download dataset from HuggingFace ----
echo "[Step 3/7] Downloading dataset from HuggingFace..."
${PIP_CMD} install -q huggingface_hub

${PYTHON_CMD} - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/cosmos-augmentation/dataset"))
hf_dataset = os.environ.get("HF_DATASET", "tshiamor/block-stacking-cosmos-transfer")

print(f"Downloading {hf_dataset} to {dataset_dir}...")
snapshot_download(
    repo_id=hf_dataset,
    repo_type="dataset",
    local_dir=dataset_dir,
)
print(f"Dataset downloaded to {dataset_dir}")
PYEOF

echo "  Dataset contents:"
ls -la "${DATASET_DIR}/"
echo "  Videos: $(ls "${DATASET_DIR}/videos/" | wc -l)"

# ---- Step 4: Build Docker container ----
echo "[Step 4/7] Building Docker container..."
cd "${COSMOS_REPO}"

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "  GPU Compute Capability: ${GPU_ARCH}"

if [ "${GPU_ARCH}" -ge "120" ]; then
    echo "  Blackwell GPU detected, using nightly Dockerfile"
    DOCKERFILE="docker/nightly.Dockerfile"
else
    echo "  Ampere/Hopper GPU detected, using standard Dockerfile"
    DOCKERFILE="Dockerfile"
fi

sudo docker build -f "${DOCKERFILE}" -t cosmos-transfer2.5 . 2>&1 | tail -5

echo "  Docker image built."

# ---- Step 4b: Verify HuggingFace access to gated model ----
echo "  Verifying access to nvidia/Cosmos-Transfer2.5-2B..."
${PYTHON_CMD} - <<'CHECKEOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info("nvidia/Cosmos-Transfer2.5-2B", token=os.environ.get("HF_TOKEN"))
    print(f"  OK: Access granted to {info.id}")
except Exception as e:
    print(f"\n  ERROR: Cannot access nvidia/Cosmos-Transfer2.5-2B")
    print(f"  {e}")
    print(f"\n  You need to accept the license at:")
    print(f"    https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B")
    print(f"  Then re-run this script.")
    sys.exit(1)
CHECKEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "ABORTING: HuggingFace model access denied."
    echo "Go to https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B and accept the license."
    exit 1
fi

# ---- Step 5: Create augmentation script inside workspace ----
echo "[Step 5/7] Preparing augmentation script..."

cat > "${WORK_DIR}/run_augmentation.py" << 'AUGEOF'
#!/usr/bin/env python3
"""
Augment block-stacking demos using Cosmos-Transfer2.5.
Runs inside the cosmos-transfer2.5 Docker container.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AUGMENTATION_PROMPTS = [
    "A robot arm in a modern laboratory setting with bright LED lighting, stacking wooden blocks on a white table. Clean industrial environment with equipment visible in the background.",
    "A robotic gripper in a dimly lit warehouse, carefully placing colorful building blocks into a container. Concrete floors and metal shelving visible.",
    "A precise robot manipulator in a sunlit workshop, arranging wooden cubes on a workbench. Natural light streaming through windows, tools and materials around.",
    "A robotic system in a futuristic factory setting with blue ambient lighting, handling geometric blocks. High-tech monitors and displays in the background.",
    "A mechanical arm in an artist's studio, gently moving painted wooden blocks. Colorful paint splashes on walls, creative chaos surrounding the scene.",
    "A robot arm on a production line, efficiently sorting and stacking product containers. Industrial conveyor belts and safety markings visible.",
    "A delicate robotic hand in a clean room environment, precisely placing sensor components. White walls, particle filters, and sterile equipment around.",
    "A collaborative robot in a home kitchen, organizing food storage containers on the counter. Warm domestic lighting, wooden cabinets visible.",
]

def create_spec_file(video_path, prompt, output_name, specs_dir):
    spec = {
        "name": output_name,
        "prompt": prompt,
        "video_path": str(video_path),
        "guidance": 3,
        "seed": 2025,
        "edge": {"control_weight": 0.9},
    }
    spec_path = Path(specs_dir) / f"{output_name}_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path

def run_inference(spec_file, output_dir):
    cmd = [
        sys.executable,
        "/workspace/examples/inference.py",
        "-i", str(spec_file),
        "-o", str(output_dir),
        "--disable-guardrails",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        return False
    return True

def main():
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    num_aug = int(os.environ.get("NUM_AUGMENTATIONS", "4"))

    videos_dir = input_dir / "videos"
    annotations_dir = input_dir / "annotations"

    # Output structure
    videos_out = output_dir / "videos"
    annotations_out = output_dir / "annotations"
    specs_dir = output_dir / "specs"
    videos_wrist_out = output_dir / "videos_wrist"
    for d in [videos_out, annotations_out, specs_dir, videos_wrist_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    metadata_src = input_dir / "metadata.json"
    if metadata_src.exists():
        shutil.copy(metadata_src, output_dir / "metadata.json")

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} source videos")
    print(f"Target: {len(video_files)} originals + {len(video_files) * num_aug} augmented = {len(video_files) * (1 + num_aug)} total")

    total = 0
    failed = 0

    for i, video_path in enumerate(video_files):
        demo_name = video_path.stem
        annotation_path = annotations_dir / f"{demo_name}.json"
        wrist_video = input_dir / "videos_wrist" / f"{demo_name}.mp4"

        print(f"\n[{i+1}/{len(video_files)}] {demo_name}")

        # Copy original
        shutil.copy(video_path, videos_out / f"{demo_name}.mp4")
        if annotation_path.exists():
            shutil.copy(annotation_path, annotations_out / f"{demo_name}.json")
        if wrist_video.exists():
            shutil.copy(wrist_video, videos_wrist_out / f"{demo_name}.mp4")
        total += 1

        # Generate augmentations
        for aug_idx in range(num_aug):
            prompt = AUGMENTATION_PROMPTS[aug_idx % len(AUGMENTATION_PROMPTS)]
            aug_name = f"{demo_name}_aug{aug_idx}"
            print(f"  [{aug_idx+1}/{num_aug}] {aug_name}")

            spec_file = create_spec_file(video_path, prompt, aug_name, specs_dir)
            temp_out = output_dir / "temp_inference"
            temp_out.mkdir(exist_ok=True)

            try:
                if run_inference(spec_file, temp_out):
                    # Find generated video
                    generated = temp_out / f"{aug_name}.mp4"
                    if not generated.exists():
                        generated = temp_out / aug_name / f"{aug_name}.mp4"

                    if generated.exists():
                        shutil.move(str(generated), str(videos_out / f"{aug_name}.mp4"))
                        if annotation_path.exists():
                            ann = json.loads(annotation_path.read_text())
                            if "video_file" in ann:
                                ann["video_file"] = f"{aug_name}.mp4"
                            (annotations_out / f"{aug_name}.json").write_text(json.dumps(ann, indent=2))
                        total += 1
                        print(f"    OK: {aug_name}.mp4")
                    else:
                        print(f"    WARN: output video not found")
                        failed += 1
                else:
                    print(f"    FAIL: inference error")
                    failed += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                failed += 1

            shutil.rmtree(temp_out, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Done! Total: {total} demos ({len(video_files)} original + {total - len(video_files)} augmented)")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
AUGEOF

# ---- Step 6: Run augmentation ----
echo "[Step 6/7] Running augmentation..."
echo "  This will run cosmos-transfer2.5 inference for each video."
echo "  With ${NUM_AUGMENTATIONS} augmentations per demo, expect ~808 inference runs."
echo ""

sudo docker run --rm \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    -v "${COSMOS_REPO}":/workspace \
    -v "${DATASET_DIR}":/data/input:ro \
    -v "${OUTPUT_DIR}":/data/output \
    -v "${WORK_DIR}/run_augmentation.py":/data/run_augmentation.py:ro \
    -v "${HOME}/.cache":/root/.cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -e HF_HOME="/root/.cache/huggingface" \
    -e INPUT_DIR=/data/input \
    -e OUTPUT_DIR=/data/output \
    -e NUM_AUGMENTATIONS="${NUM_AUGMENTATIONS}" \
    cosmos-transfer2.5 \
    bash -c "huggingface-cli login --token \${HF_TOKEN} 2>/dev/null || true; python3 /data/run_augmentation.py"

echo "  Augmentation complete."
echo "  Output videos: $(ls "${OUTPUT_DIR}/videos/" 2>/dev/null | wc -l)"

# ---- Step 7: Upload to HuggingFace ----
echo "[Step 7/7] Uploading results to HuggingFace..."

${PYTHON_CMD} - <<'UPLOADEOF'
import os
from huggingface_hub import HfApi

output_dir = os.environ.get("OUTPUT_DIR", os.path.expanduser("~/cosmos-augmentation/augmented_output"))
repo_id = os.environ.get("HF_OUTPUT_REPO", "tshiamor/block-stacking-cosmos-augmented-1000")

api = HfApi()

# Create dataset repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

print(f"Uploading {output_dir} to {repo_id}...")
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload cosmos-augmented dataset (202 original + augmented)",
)
print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
UPLOADEOF

echo ""
echo "============================================="
echo "PIPELINE COMPLETE"
echo "============================================="
echo "Dataset: https://huggingface.co/datasets/${HF_OUTPUT_REPO}"
echo "============================================="
