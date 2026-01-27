# Cosmos-Transfer2.5 Dataset Augmentation Guide

## Overview

This guide explains how to use Cosmos-Transfer2.5 to augment your block-stacking dataset from 202 demos to 1000+ demos.

**Why Cosmos-Transfer2.5 instead of Cosmos-Transfer1?**
- Cosmos-Transfer2.5 has **native Blackwell GPU support** (RTX 5090)
- Uses newer CUDA 12.8.1 / PyTorch 25.10
- 2B parameter model (smaller than 7B, but still requires significant VRAM)

## Hardware Requirements

| Configuration | VRAM Required | Supported |
|--------------|---------------|-----------|
| Single H100/B200 | 65.4 GB | Yes |
| 2x RTX 5090 (multi-GPU) | 64 GB total | Likely (with offloading) |
| Single RTX 5090 | 32 GB | No (insufficient VRAM) |
| A100 80GB | 80 GB | Yes |

## Option 1: Docker on Multi-GPU Setup (Local)

If you have 2+ RTX 5090s or similar GPUs:

```bash
cd ~/cosmos-transfer2.5

# Build Docker for Blackwell
docker build -f docker/nightly.Dockerfile -t cosmos-transfer2.5-blackwell .

# Run with multi-GPU
docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    -v ~/cosmos-transfer2.5:/workspace \
    -v ~/simulations/franka/isaaclab_cloudxr/datasets/cosmos_format:/data/input \
    -v ~/simulations/franka/isaaclab_cloudxr/datasets/cosmos_augmented:/data/output \
    -v ~/.cache:/root/.cache \
    -e HF_TOKEN="$HF_TOKEN" \
    cosmos-transfer2.5-blackwell

# Inside container, run with torchrun for multi-GPU
torchrun --nproc_per_node=2 examples/inference.py \
    -i /data/input/spec.json \
    -o /data/output
```

## Option 2: Remote Machine (Cloud/Cluster)

### Step 1: Transfer Dataset to Remote

```bash
# On your local machine
rsync -avz --progress \
    ~/simulations/franka/isaaclab_cloudxr/datasets/cosmos_format/ \
    user@remote-server:/data/cosmos_input/

# Also transfer the augmentation script
scp ~/simulations/franka/isaaclab_cloudxr/scripts/augment_with_cosmos_transfer2.py \
    user@remote-server:/data/
```

### Step 2: Setup on Remote Machine

```bash
# SSH to remote
ssh user@remote-server

# Clone cosmos-transfer2.5
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git lfs pull

# Build Docker
# For H100/A100:
docker build -f Dockerfile -t cosmos-transfer2.5 .

# For Blackwell:
docker build -f docker/nightly.Dockerfile -t cosmos-transfer2.5 .
```

### Step 3: Run Augmentation

```bash
docker run -it --rm \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    -v /path/to/cosmos-transfer2.5:/workspace \
    -v /data/cosmos_input:/data/input \
    -v /data/cosmos_output:/data/output \
    -v ~/.cache:/root/.cache \
    -e HF_TOKEN="your_hf_token" \
    cosmos-transfer2.5

# Inside container
python /data/augment_with_cosmos_transfer2.py \
    --input /data/input \
    --output /data/output \
    --num_augmentations 4 \
    --cosmos_repo /workspace
```

### Step 4: Transfer Results Back

```bash
# On remote, when done
rsync -avz --progress \
    /data/cosmos_output/ \
    user@local-machine:~/simulations/franka/isaaclab_cloudxr/datasets/cosmos_augmented/
```

## Option 3: Cloud GPU Services

### NVIDIA DGX Cloud / NGC

```bash
# Pull NGC container (if available)
docker pull nvcr.io/nvidia/cosmos-transfer2.5:latest

# Or use NVIDIA Brev/DGX Cloud with H100s
```

### AWS/GCP/Azure with H100

1. Launch instance with H100 GPU (p5.xlarge on AWS)
2. Follow "Remote Machine" steps above
3. Use spot instances to reduce cost

### Lambda Labs / RunPod

These services offer H100 rentals at lower cost:
- Lambda Labs: ~$2.50/hr for H100
- RunPod: ~$2.00/hr for H100

## Manual Inference Commands

To run inference on individual videos:

```bash
# Create a spec file (robot_spec.json)
cat > robot_spec.json << 'EOF'
{
    "name": "demo_001_aug0",
    "prompt": "A robot arm in a modern laboratory setting with bright LED lighting, stacking wooden blocks on a white table.",
    "video_path": "/data/input/videos/demo_001.mp4",
    "guidance": 3,
    "edge": {
        "control_weight": 0.9
    }
}
EOF

# Run inference
python examples/inference.py -i robot_spec.json -o outputs/

# For multi-GPU
torchrun --nproc_per_node=2 examples/inference.py -i robot_spec.json -o outputs/
```

## Control Types

| Control | Description | Use Case |
|---------|-------------|----------|
| `edge` | Canny edge detection | Preserve structure and shapes |
| `depth` | Depth estimation | Preserve 3D layout |
| `seg` | Semantic segmentation | Preserve object classes |
| `vis` | Blur/visual features | Style transfer |

For robot manipulation, **edge + depth** typically works best:
```json
{
    "edge": {"control_weight": 0.9},
    "depth": {"control_weight": 0.5}
}
```

## Expected Output

From 202 original demos with 4 augmentations each:
- Original: 202 demos
- Augmented: 202 × 4 = 808 demos
- Total: 1010 demos

## Upload to HuggingFace

After augmentation:

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
huggingface-cli upload tshiamor/block-stacking-cosmos-augmented-1000 \
    ~/simulations/franka/isaaclab_cloudxr/datasets/cosmos_augmented/
```

## Troubleshooting

### Out of Memory
- Use multi-GPU with torchrun
- Reduce `guidance` value (0-7, lower = less memory)
- Reduce video resolution in spec file

### Slow Inference
- Enable guardrail offloading: `--offload-guardrail-models`
- Disable guardrails for faster inference: `--disable-guardrails`

### Docker Permission Issues
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Files Created

- `scripts/augment_with_cosmos_transfer2.py` - Augmentation script
- `COSMOS_TRANSFER2_SETUP.md` - This documentation

## Repository Locations

- Cosmos-Transfer2.5: `~/cosmos-transfer2.5/`
- Original Dataset: `datasets/cosmos_format/`
- Augmented Dataset: `datasets/cosmos_augmented/` (to be created)
