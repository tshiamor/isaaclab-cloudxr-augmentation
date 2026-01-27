#!/usr/bin/env python3
"""
Custom Cosmos-Transfer1 inference script without guardrail.

This script runs Cosmos-Transfer1 inference with guardrail disabled,
allowing generation without needing to download Cosmos-Guardrail1.

Based on: cosmos_transfer1/diffusion/inference/transfer.py
"""

import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cosmos-Transfer1 inference without guardrail")

    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoints")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_video", type=str, required=True, help="Output video path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str,
        default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality.",
        help="Negative prompt")
    parser.add_argument("--control_type", type=str, default="edge", choices=["edge", "depth", "seg"])
    parser.add_argument("--control_weight", type=float, default=1.0)
    parser.add_argument("--use_distilled", action="store_true", help="Use distilled model")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    # Parse arguments FIRST (while still in original directory)
    args = parse_arguments()

    # Convert all paths to absolute before changing directory
    input_video = os.path.abspath(args.input_video)
    output_video = os.path.abspath(args.output_video)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Set up cosmos-transfer1 environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    COSMOS_REPO = os.path.expanduser("~/cosmos-transfer1")

    # Add to path and change directory (required for config file loading)
    sys.path.insert(0, COSMOS_REPO)
    os.chdir(COSMOS_REPO)

    # Now we can import cosmos modules
    import torch
    torch.enable_grad(False)

    import cosmos_transfer1.utils.fix_vllm_registration
    from cosmos_transfer1.checkpoints import (
        BASE_7B_CHECKPOINT_PATH,
        EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
    )
    from cosmos_transfer1.diffusion.inference.world_generation_pipeline import (
        DiffusionControl2WorldGenerationPipeline,
        DistilledControl2WorldGenerationPipeline,
    )
    from cosmos_transfer1.utils import log, misc
    from cosmos_transfer1.utils.io import save_video

    misc.set_random_seed(args.seed)

    # Build control inputs with checkpoint path
    if args.use_distilled:
        # Distilled model checkpoint paths
        distilled_checkpoints = {
            "edge": f"{checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/edge_control_distilled.pt",
        }
        if args.control_type not in distilled_checkpoints:
            log.error(f"Distilled model only supports edge control, got: {args.control_type}")
            return False
        ckpt_path = distilled_checkpoints[args.control_type]
    else:
        # Full model checkpoint paths
        full_checkpoints = {
            "edge": f"{checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/edge_control.pt",
            "depth": f"{checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/depth_control.pt",
            "seg": f"{checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/seg_control.pt",
        }
        ckpt_path = full_checkpoints[args.control_type]

    control_inputs = {
        args.control_type: {
            "control_weight": args.control_weight,
            "ckpt_path": ckpt_path,
        }
    }

    # Create pipeline with guardrail disabled
    log.info(f"Loading model with guardrail disabled...")
    if args.use_distilled:
        checkpoint = EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH
        pipeline = DistilledControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint,
            offload_network=True,  # Offload to make room for T5
            offload_text_encoder_model=True,
            offload_guardrail_models=True,
            disable_guardrail=True,
            guidance=args.guidance,
            num_steps=args.num_steps,
            fps=args.fps,
            seed=args.seed,
            num_input_frames=1,
            control_inputs=control_inputs,
        )
    else:
        checkpoint = BASE_7B_CHECKPOINT_PATH
        pipeline = DiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint,
            offload_network=True,  # Offload to make room for T5
            offload_text_encoder_model=True,
            offload_guardrail_models=True,
            disable_guardrail=True,
            guidance=args.guidance,
            num_steps=args.num_steps,
            fps=args.fps,
            seed=args.seed,
            num_input_frames=1,
            control_inputs=control_inputs,
        )

    log.info(f"Processing: {input_video}")
    log.info(f"Prompt: {args.prompt}")

    # Generate video
    result = pipeline.generate(
        prompt=args.prompt,
        video_path=input_video,
        negative_prompt=args.negative_prompt,
        control_inputs=control_inputs,
    )

    if result is not None:
        # Save output video
        output_dir = os.path.dirname(output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_video(result, output_video, fps=args.fps)
        log.info(f"Saved: {output_video}")
        return True
    else:
        log.error("Generation failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
