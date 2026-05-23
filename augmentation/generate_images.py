"""
Generate synthetic dermoscopy images using a trained SGAN generator.

Loads a trained generator checkpoint and generates realistic synthetic images
for dataset augmentation.
"""

from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.sgan import SGAN
from utils.config import SGANConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_samples(
    checkpoint_path: str,
    num_images: int = 1000,
    config: Optional[SGANConfig] = None,
    batch_size: int = 64,
    config_path: str = "configs/config.yaml"
) -> int:
    """
    Generate synthetic images using trained generator.

    Args:
        checkpoint_path: Path to model checkpoint
        num_images: Number of images to generate (default: 1000)
        config: SGAN configuration (if None, loads from config_path)
        batch_size: Batch size for generation (default: 64)
        config_path: Path to config file if config not provided

    Returns:
        Number of images generated

    Raises:
        FileNotFoundError: If checkpoint or config not found
        RuntimeError: If checkpoint loading fails
    """
    logger.info(f"Initializing image generation ({num_images} images)...")

    # Load config if not provided
    if config is None:
        logger.info(f"Loading config from: {config_path}")
        config = SGANConfig.from_yaml(config_path)

    # Validate checkpoint exists
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model
        logger.info("Loading SGAN model...")
        model = SGAN(config).to(device)

        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "generator" in checkpoint:
                # Full model checkpoint
                model.load_state_dict(checkpoint)
            elif "generator_state_dict" in checkpoint or "state_dict" in checkpoint:
                # Alternative checkpoint format
                state_dict = checkpoint.get("generator_state_dict", checkpoint.get("state_dict"))
                model.generator.load_state_dict(state_dict)
            else:
                # Try direct load
                model.generator.load_state_dict(checkpoint)
        else:
            # Assume it's a state dict
            model.generator.load_state_dict(checkpoint)

        logger.info("Checkpoint loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}")

    # Prepare generator
    generator = model.generator
    generator.eval()
    logger.info("Generator set to evaluation mode")

    # Prepare output directory
    output_dir = Path(config.dataset.generated_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate images
    latent_dim = config.generator.latent_dim
    generated = 0

    logger.info(f"Generating {num_images} images...")

    with torch.no_grad():
        # Use progress bar
        num_batches = (num_images + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating images"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_images - generated)

            # Generate noise
            z = torch.randn(
                current_batch_size,
                latent_dim,
                1, 1,
                device=device
            )

            # Generate images
            try:
                fake_images = generator(z)
            except Exception as e:
                logger.error(f"Generation failed at batch {batch_idx}: {e}")
                raise

            # Save images
            for img_idx, img in enumerate(fake_images):
                if generated >= num_images:
                    break

                img_filename = f"gen_{generated:06d}.png"
                img_path = output_dir / img_filename

                try:
                    save_image(img, str(img_path), normalize=True)
                    generated += 1
                except Exception as e:
                    logger.error(f"Failed to save image {img_filename}: {e}")
                    raise

    logger.info(f"Successfully generated {generated} images")
    logger.info(f"Images saved to: {output_dir}")

    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic dermoscopy images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation"
    )

    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        config_path=args.config,
        batch_size=args.batch_size
    )
