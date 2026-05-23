"""
Comprehensive sanity checks for the SGAN project.

Validates:
- Dataset availability and integrity
- Model initialization and forward pass
- Configuration loading
- Dependencies installation
"""

import sys
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn

from models.generator import Generator
from models.discriminator import Discriminator
from models.sgan import SGAN
from utils.config import SGANConfig
from utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.

    Returns:
        True if all dependencies available, False otherwise
    """
    logger.info("Checking dependencies...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('pydantic', 'Pydantic'),
    ]

    all_available = True
    for module, name in required_packages:
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} not found")
            all_available = False

    return all_available


def check_config_file(config_path: str = "configs/config.yaml") -> bool:
    """
    Check if configuration file exists and is valid.

    Args:
        config_path: Path to config file

    Returns:
        True if config is valid, False otherwise
    """
    logger.info(f"Checking configuration file: {config_path}")

    try:
        config = SGANConfig.from_yaml(config_path)
        logger.info("  ✓ Configuration loaded successfully")
        logger.info(f"    - Experiment: {config.experiment.name}")
        logger.info(f"    - Image size: {config.dataset.image_size}x{config.dataset.image_size}")
        logger.info(f"    - Batch size: {config.training.batch_size}")
        return True
    except FileNotFoundError:
        logger.error(f"  ✗ Config file not found: {config_path}")
        return False
    except Exception as e:
        logger.error(f"  ✗ Config validation failed: {e}")
        return False


def check_datasets(config: SGANConfig) -> bool:
    """
    Check if dataset directories exist and contain images.

    Args:
        config: SGAN configuration

    Returns:
        True if datasets are valid, False otherwise
    """
    logger.info("Checking datasets...")

    try:
        from datasets.melanoma_dataset import (
            LabeledMelanomaDataset,
            UnlabeledMelanomaDataset
        )

        # Check labeled dataset
        labeled_path = config.dataset.labeled_path
        benign_path = Path(labeled_path) / "benign"
        malignant_path = Path(labeled_path) / "malignant"

        if not benign_path.exists() or not malignant_path.exists():
            logger.warning(f"  ✗ Labeled dataset directories not found")
            logger.warning(f"    Expected: {benign_path}, {malignant_path}")
            logger.info("    Note: Dataset will be downloaded during setup.sh")
            return True  # Not a failure, just not set up yet

        try:
            labeled_dataset = LabeledMelanomaDataset(
                labeled_path,
                config.dataset.image_size
            )
            benign_count = len([s for s in labeled_dataset.samples if s[1] == 0])
            malignant_count = len([s for s in labeled_dataset.samples if s[1] == 1])
            logger.info(f"  ✓ Labeled dataset: {len(labeled_dataset)} samples")
            logger.info(f"    - Benign: {benign_count}")
            logger.info(f"    - Malignant: {malignant_count}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load labeled dataset: {e}")
            return False

        # Check unlabeled dataset
        unlabeled_path = config.dataset.unlabeled_path
        if not Path(unlabeled_path).exists():
            logger.warning(f"  ✗ Unlabeled dataset directory not found: {unlabeled_path}")
            logger.info("    Note: Dataset will be downloaded during setup.sh")
            return True  # Not a failure, just not set up yet

        try:
            unlabeled_dataset = UnlabeledMelanomaDataset(
                unlabeled_path,
                config.dataset.image_size
            )
            logger.info(f"  ✓ Unlabeled dataset: {len(unlabeled_dataset)} samples")
        except Exception as e:
            logger.error(f"  ✗ Failed to load unlabeled dataset: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"  ✗ Dataset check failed: {e}")
        return False


def check_generator(config: SGANConfig) -> bool:
    """
    Check if generator can be initialized and performs forward pass.

    Args:
        config: SGAN configuration

    Returns:
        True if generator works, False otherwise
    """
    logger.info("Checking Generator...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = Generator(
            latent_dim=config.generator.latent_dim,
            feature_maps=config.generator.feature_maps,
            channels=config.dataset.num_channels
        ).to(device)

        batch_size = 8
        z = torch.randn(
            batch_size,
            config.generator.latent_dim,
            1, 1,
            device=device
        )

        with torch.no_grad():
            fake_images = generator(z)

        expected_shape = (batch_size, config.dataset.num_channels,
                          config.dataset.image_size, config.dataset.image_size)

        if fake_images.shape == expected_shape:
            logger.info(f"  ✓ Generator working")
            logger.info(f"    - Input shape: {z.shape}")
            logger.info(f"    - Output shape: {fake_images.shape}")
            return True
        else:
            logger.error(f"  ✗ Generator output shape mismatch")
            logger.error(f"    Expected: {expected_shape}, got: {fake_images.shape}")
            return False

    except Exception as e:
        logger.error(f"  ✗ Generator check failed: {e}")
        return False


def check_discriminator(config: SGANConfig) -> bool:
    """
    Check if discriminator can be initialized and performs forward pass.

    Args:
        config: SGAN configuration

    Returns:
        True if discriminator works, False otherwise
    """
    logger.info("Checking Discriminator...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        discriminator = Discriminator(
            feature_maps=config.discriminator.feature_maps,
            channels=config.dataset.num_channels,
            num_classes=config.dataset.num_classes
        ).to(device)

        batch_size = 8
        images = torch.randn(
            batch_size,
            config.dataset.num_channels,
            config.dataset.image_size,
            config.dataset.image_size,
            device=device
        )

        with torch.no_grad():
            logits = discriminator(images)

        expected_logits_shape = (batch_size, config.dataset.num_classes)

        if logits.shape == expected_logits_shape:
            logger.info(f"  ✓ Discriminator working")
            logger.info(f"    - Input shape: {images.shape}")
            logger.info(f"    - Output shape: {logits.shape}")

            # Check feature extraction
            logits_with_features, features = discriminator(images, return_features=True)
            if features is not None:
                logger.info(f"    - Features shape: {features.shape}")
            return True
        else:
            logger.error(f"  ✗ Discriminator output shape mismatch")
            logger.error(f"    Expected: {expected_logits_shape}, got: {logits.shape}")
            return False

    except Exception as e:
        logger.error(f"  ✗ Discriminator check failed: {e}")
        return False


def check_sgan_model(config: SGANConfig) -> bool:
    """
    Check if SGAN model can be initialized and performs forward pass.

    Args:
        config: SGAN configuration

    Returns:
        True if SGAN works, False otherwise
    """
    logger.info("Checking SGAN Model...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SGAN(config).to(device)

        batch_size = 8

        # Test generator forward pass
        z = torch.randn(
            batch_size,
            config.generator.latent_dim,
            1, 1,
            device=device
        )

        with torch.no_grad():
            fake_images = model.generate(z)

        logger.info(f"  ✓ SGAN model working")
        logger.info(f"    - Generator output: {fake_images.shape}")

        # Test discriminator forward pass
        with torch.no_grad():
            logits = model.discriminate(fake_images)

        logger.info(f"    - Discriminator output: {logits.shape}")
        return True

    except Exception as e:
        logger.error(f"  ✗ SGAN check failed: {e}")
        return False


def check_device() -> bool:
    """
    Check available compute device.

    Returns:
        True always (informational check)
    """
    logger.info("Checking compute device...")

    if torch.cuda.is_available():
        logger.info(f"  ✓ CUDA available")
        logger.info(f"    - Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"    - CUDA version: {torch.version.cuda}")
    else:
        logger.warning(f"  ⚠ CUDA not available, will use CPU (slower)")

    return True


def run_all_checks(config_path: str = "configs/config.yaml") -> bool:
    """
    Run all sanity checks.

    Args:
        config_path: Path to configuration file

    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("SGAN Sanity Checks")
    logger.info("=" * 60 + "\n")

    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", lambda: check_config_file(config_path)),
        ("Compute Device", check_device),
    ]

    all_passed = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_passed = False
        except Exception as e:
            logger.error(f"Unexpected error in {name} check: {e}")
            all_passed = False
        logger.info("")

    # Load config for remaining checks
    try:
        config = SGANConfig.from_yaml(config_path)

        additional_checks = [
            ("Datasets", lambda: check_datasets(config)),
            ("Generator", lambda: check_generator(config)),
            ("Discriminator", lambda: check_discriminator(config)),
            ("SGAN Model", lambda: check_sgan_model(config)),
        ]

        for name, check_fn in additional_checks:
            try:
                if not check_fn():
                    all_passed = False
            except Exception as e:
                logger.error(f"Unexpected error in {name} check: {e}")
                all_passed = False
            logger.info("")

    except Exception as e:
        logger.error(f"Failed to load config for detailed checks: {e}")
        all_passed = False

    # Summary
    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ All sanity checks passed!")
    else:
        logger.warning("✗ Some sanity checks failed")
    logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    setup_logging()

    success = run_all_checks()
    sys.exit(0 if success else 1)
