"""
Main training loop for the Melanoma SGAN.

Orchestrates data loading, model initialization, optimization, and training loop.
"""

from typing import Optional
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch import optim

from datasets.melanoma_dataset import (
    LabeledMelanomaDataset,
    UnlabeledMelanomaDataset
)
from models.sgan import SGAN
from training.trainer import Trainer
from utils.seed import set_seed
from utils.config import SGANConfig
from utils.lr_scheduler import LRScheduler
from utils.logging_config import get_logger

logger = get_logger(__name__)


def train(config: SGANConfig) -> None:
    """
    Train the SGAN model.

    Handles:
    - Seed setting for reproducibility
    - Dataset loading with error handling
    - Model initialization
    - Optimizer and scheduler setup
    - Training loop coordination

    Args:
        config: SGAN configuration object (SGANConfig instance)

    Raises:
        FileNotFoundError: If dataset directories not found
        RuntimeError: If CUDA not available when required
    """
    logger.info("Initializing training...")

    # Set seed for reproducibility
    set_seed(config.experiment.seed)
    cudnn.benchmark = True
    logger.debug(f"Set seed: {config.experiment.seed}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------------------------
    # Load Datasets
    # ---------------------------
    logger.info("Loading datasets...")

    try:
        labeled_dataset = LabeledMelanomaDataset(
            config.dataset.labeled_path,
            config.dataset.image_size
        )
        logger.info(
            f"Loaded labeled dataset: {len(labeled_dataset)} samples "
            f"from {config.dataset.labeled_path}"
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load labeled dataset: {e}")
        raise

    try:
        unlabeled_dataset = UnlabeledMelanomaDataset(
            config.dataset.unlabeled_path,
            config.dataset.image_size
        )
        logger.info(
            f"Loaded unlabeled dataset: {len(unlabeled_dataset)} samples "
            f"from {config.dataset.unlabeled_path}"
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load unlabeled dataset: {e}")
        raise

    # Create data loaders
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
        drop_last=True,
    )

    logger.info(f"Data loaders created with batch size: {config.training.batch_size}")

    # ---------------------------
    # Initialize Model
    # ---------------------------
    logger.info("Initializing SGAN model...")

    model = SGAN(config).to(device)
    num_params_g = sum(p.numel() for p in model.generator.parameters())
    num_params_d = sum(p.numel() for p in model.discriminator.parameters())
    logger.info(
        f"Model initialized - "
        f"Generator params: {num_params_g:,}, "
        f"Discriminator params: {num_params_d:,}"
    )

    # ---------------------------
    # Setup Optimizers
    # ---------------------------
    logger.info("Setting up optimizers...")

    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=config.training.lr_generator,
        betas=(config.training.beta1, config.training.beta2)
    )

    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=config.training.lr_discriminator,
        betas=(config.training.beta1, config.training.beta2)
    )

    logger.debug(
        f"Optimizer config - "
        f"LR_G: {config.training.lr_generator}, "
        f"LR_D: {config.training.lr_discriminator}, "
        f"Beta1: {config.training.beta1}, Beta2: {config.training.beta2}"
    )

    # ---------------------------
    # Setup Learning Rate Scheduler
    # ---------------------------
    lr_scheduler = LRScheduler(
        optimizer_g,
        optimizer_d,
        scheduler_type=config.training.lr_scheduler_type,
        decay_steps=config.training.lr_decay_steps,
        decay_factor=config.training.lr_decay_factor,
        total_epochs=config.training.epochs
    )
    logger.info(f"Learning rate scheduler: {config.training.lr_scheduler_type}")

    # ---------------------------
    # Initialize Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler=lr_scheduler,
        device=device,
        config=config
    )

    # ---------------------------
    # Training Loop
    # ---------------------------
    logger.info(f"Starting training for {config.training.epochs} epochs...")

    try:
        for epoch in range(config.training.epochs):
            avg_loss_d, avg_loss_g = trainer.train_epoch(epoch)

            # Log learning rates
            lr_g, lr_d = lr_scheduler.get_last_lr()
            logger.debug(f"Epoch {epoch} - LR_G: {lr_g:.6f}, LR_D: {lr_d:.6f}")

            # Step scheduler
            lr_scheduler.step()

            if (epoch + 1) % config.output.save_interval == 0:
                logger.info(
                    f"Epoch {epoch}/{config.training.epochs - 1} - "
                    f"Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f}"
                )

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
