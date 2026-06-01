"""
Main entry point for Melanoma SGAN training and image generation.

Supports two modes:
- train: Train the SGAN model on labeled/unlabeled data
- generate: Generate synthetic images using a trained generator

Example:
    Training mode:
        python main.py --mode train --config configs/config.yaml

    Generation mode:
        python main.py --mode generate --checkpoint outputs/checkpoints/best_generator.pt --num_images 2000
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from utils.logging_config import setup_logging, get_logger
from utils.config import SGANConfig

# Enable TF32 for faster training on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True   # auto-tune conv algorithms for fixed input sizes

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments

    Raises:
        SystemExit: If invalid arguments provided
    """
    parser = argparse.ArgumentParser(
        description="Semi-Supervised GAN for Melanoma Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train SGAN:
    python main.py --mode train --config configs/config.yaml

  Generate images:
    python main.py --mode generate --checkpoint outputs/checkpoints/best_generator.pt
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate"],
        help="Execution mode (default: train)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for generation mode"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate in generation mode (default: 1000)"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def prepare_output_dirs(config: SGANConfig) -> None:
    """
    Create required output directories.

    Args:
        config: SGAN configuration object
    """
    dirs = [
        config.output.checkpoint_dir,
        config.output.sample_dir,
        config.output.log_dir
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created/verified directory: {dir_path}")


def print_experiment_info(config: SGANConfig, mode: str) -> None:
    """
    Print experiment configuration summary to logs.

    Args:
        config: SGAN configuration object
        mode: Execution mode ('train' or 'generate')
    """
    logger.info("=" * 50)
    logger.info("SGAN Melanoma Experiment")
    logger.info("=" * 50)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Seed: {config.experiment.seed}")

    logger.info("\nDataset Configuration")
    logger.info("-" * 50)
    logger.info(f"Labeled path: {config.dataset.labeled_path}")
    logger.info(f"Unlabeled path: {config.dataset.unlabeled_path}")
    logger.info(f"Generated path: {config.dataset.generated_path}")
    logger.info(f"Image size: {config.dataset.image_size}x{config.dataset.image_size}")

    if mode == "train":
        logger.info("\nTraining Configuration")
        logger.info("-" * 50)
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Epochs: {config.training.epochs}")
        logger.info(f"LR (Generator): {config.training.lr_generator}")
        logger.info(f"LR (Discriminator): {config.training.lr_discriminator}")
        logger.info(f"LR Scheduler: {config.training.lr_scheduler_type}")

    logger.info("=" * 50 + "\n")


def main() -> None:
    """
    Main entry point.

    Orchestrates training or generation based on CLI arguments.

    Raises:
        FileNotFoundError: If config file or checkpoint not found
        ValueError: If configuration is invalid
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    logger.info(f"Starting Melanoma SGAN - Mode: {args.mode}")

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = SGANConfig.from_yaml(args.config)
        logger.info("Configuration validated successfully")

        # Create output directories
        prepare_output_dirs(config)

        # Print experiment info
        print_experiment_info(config, args.mode)

        if args.mode == "train":
            logger.info("Starting training...")
            from training.train_sgan import train
            train(config)
            logger.info("Training completed successfully")

        elif args.mode == "generate":
            if args.checkpoint is None:
                raise ValueError("--checkpoint required for generation mode")

            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

            logger.info(f"Starting image generation with checkpoint: {args.checkpoint}")
            from augmentation.generate_images import generate_samples
            generate_samples(
                checkpoint_path=args.checkpoint,
                num_images=args.num_images,
                config=config
            )
            logger.info("Image generation completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration or argument: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()