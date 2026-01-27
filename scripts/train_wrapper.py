#!/usr/bin/env python3
"""
Wrapper to train robomimic policy with custom extension loaded.
"""

import sys
import os

# Add IsaacLab to path
sys.path.insert(0, "/home/tshiamo/IsaacLab/scripts/imitation_learning/robomimic")

# Import custom extension FIRST to register environments
print("[INFO] Loading isaaclab_cloudxr extension...")
import isaaclab_cloudxr
print("[INFO] Custom extension loaded!")

# Now import and run the training script
from train import main, parse_cli_args

if __name__ == "__main__":
    args = parse_cli_args()
    main(args)
