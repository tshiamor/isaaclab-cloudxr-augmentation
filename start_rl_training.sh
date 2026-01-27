#!/bin/bash
#
# Quick script to start RL training for Franka JengaBowl task
#

cd /home/tshiamo/simulations/franka/isaaclab_cloudxr

echo "========================================"
echo "Starting RL Training"
echo "Task: Isaac-Franka-JengaBowl-RL-v0"
echo "Environments: 4096"
echo "Mode: Headless (no visualization)"
echo "========================================"
echo ""

/home/tshiamo/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Franka-JengaBowl-RL-v0 \
    --num_envs 4096 \
    --headless

echo ""
echo "========================================"
echo "Training session ended"
echo "========================================"
