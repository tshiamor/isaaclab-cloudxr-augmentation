#!/usr/bin/env python3
"""
Train robomimic policy with custom isaaclab_cloudxr extension.

This script properly initializes Isaac Sim before importing the custom extension.
"""

import argparse

# Parse args first (before any imports that need Isaac Sim)
parser = argparse.ArgumentParser(description="Train robomimic policy with custom extension")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
parser.add_argument("--algo", type=str, default="bc", help="Algorithm (bc, bc_rnn, etc)")
parser.add_argument("--name", type=str, default=None, help="Experiment name")
parser.add_argument("--log_dir", type=str, default="logs/robomimic", help="Log directory")
parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
parser.add_argument("--normalize_training_actions", action="store_true", help="Normalize actions during training")
args, unknown = parser.parse_known_args()

# Initialize Isaac Sim AppLauncher
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli={"headless": True})
simulation_app = app_launcher.app

# NOW import custom extension (after Isaac Sim is initialized)
print(f"[INFO] Loading custom extension: isaaclab_cloudxr")
import isaaclab_cloudxr
print(f"[INFO] Custom extension loaded!")

# Import gymnasium to verify environment is registered
import gymnasium as gym
print(f"[INFO] Checking if environment '{args.task}' is registered...")
try:
    env_spec = gym.spec(args.task)
    print(f"[INFO] ✓ Environment found: {args.task}")
except:
    print(f"[ERROR] Environment '{args.task}' not found!")
    print(f"[INFO] Available JengaBowl environments:")
    for spec in gym.envs.registry.values():
        if 'JengaBowl' in spec.id:
            print(f"  - {spec.id}")
    simulation_app.close()
    exit(1)

# Now run the actual training script
import sys
import os
sys.path.insert(0, "/home/tshiamo/IsaacLab/scripts/imitation_learning/robomimic")

# Import the train module
from train import main

# Create args object for train.py
class TrainArgs:
    def __init__(self):
        self.task = args.task
        self.dataset = args.dataset
        self.algo = args.algo
        self.name = args.name
        self.log_dir = args.log_dir
        self.epochs = args.epochs
        self.normalize_training_actions = args.normalize_training_actions

# Run training
try:
    train_args = TrainArgs()
    main(train_args)
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    simulation_app.close()
