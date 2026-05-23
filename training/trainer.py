"""
Training orchestrator for SGAN model.

Handles epoch-level training, loss computation, checkpoint management, and monitoring.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.losses import (
    supervised_loss,
    unlabeled_real_loss,
    fake_loss,
    feature_matching_loss
)
from utils.visualization import save_generated_images
from utils.config import SGANConfig
from utils.lr_scheduler import LRScheduler
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training orchestrator for SGAN.

    Manages:
    - Per-epoch training loops
    - Loss computation (supervised, unlabeled, fake, feature matching)
    - Model checkpointing with best model tracking
    - Metric logging to TensorBoard
    - Sample image generation
    """

    def __init__(
        self,
        model: nn.Module,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        device: torch.device,
        config: SGANConfig
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: SGAN model with generator and discriminator
            labeled_loader: DataLoader for labeled images
            unlabeled_loader: DataLoader for unlabeled images
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            lr_scheduler: Learning rate scheduler (optional)
            device: Compute device (CPU/CUDA)
            config: SGAN configuration object
        """
        self.model = model
        self.G = model.generator
        self.D = model.discriminator

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler = lr_scheduler

        self.device = device
        self.config = config

        self.latent_dim = config.generator.latent_dim

        # Setup output directories
        self.sample_dir = Path(config.output.sample_dir)
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.log_dir = Path(config.output.log_dir)

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Sample dir: {self.sample_dir}")
        logger.debug(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.debug(f"Log dir: {self.log_dir}")

        # Setup TensorBoard
        self.writer = SummaryWriter(str(self.log_dir))
        logger.info(f"TensorBoard logs: {self.log_dir}")

        # Mixed precision training
        self.scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")

        # Track training metrics
        self.best_g_loss = float("inf")
        self.best_d_loss = float("inf")
        self.best_epoch = -1

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Run a single training epoch.

        Args:
            epoch: Epoch number

        Returns:
            Tuple of (average_discriminator_loss, average_generator_loss)

        Raises:
            RuntimeError: If training step fails
        """
        self.G.train()
        self.D.train()

        labeled_iter = iter(self.labeled_loader)

        loop = tqdm(
            self.unlabeled_loader,
            desc=f"Epoch {epoch}",
            leave=True
        )

        total_loss_d = 0.0
        total_loss_g = 0.0
        total_loss_sup = 0.0
        total_loss_unlab = 0.0
        total_loss_fake = 0.0
        total_loss_fm = 0.0

        try:
            for step, unlabeled_batch in enumerate(loop):
                # Get labeled batch (cycle through if needed)
                try:
                    labeled_batch, labels = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.labeled_loader)
                    labeled_batch, labels = next(labeled_iter)

                # Move data to device
                labeled_batch = labeled_batch.to(self.device)
                labels = labels.to(self.device)
                unlabeled_batch = unlabeled_batch.to(self.device)

                batch_size = unlabeled_batch.size(0)

                # Generate fake images
                z = torch.randn(
                    batch_size,
                    self.latent_dim,
                    1, 1,
                    device=self.device
                )
                fake_imgs = self.G(z)

                # ========================================
                # Train Discriminator
                # ========================================
                self.optimizer_d.zero_grad()

                try:
                    with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                        # Supervised loss on labeled images
                        logits_labeled = self.D(labeled_batch)
                        loss_sup = supervised_loss(logits_labeled, labels)

                        # Unlabeled real loss
                        logits_unlabeled = self.D(unlabeled_batch)
                        loss_unlab = unlabeled_real_loss(logits_unlabeled)

                        # Fake loss
                        logits_fake = self.D(fake_imgs.detach())
                        loss_fake_val = fake_loss(logits_fake)

                        # Combined discriminator loss
                        loss_d = loss_sup + loss_unlab + loss_fake_val

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer_d)
                    self.scaler.update()

                except Exception as e:
                    logger.error(f"Discriminator training step failed: {e}")
                    raise RuntimeError(f"Discriminator training failed at step {step}: {e}")

                # ========================================
                # Train Generator
                # ========================================
                self.optimizer_g.zero_grad()

                z = torch.randn(
                    batch_size,
                    self.latent_dim,
                    1, 1,
                    device=self.device
                )
                fake_imgs = self.G(z)

                try:
                    with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                        # Get features for feature matching
                        logits_fake, fake_features = self.D(fake_imgs, return_features=True)
                        _, real_features = self.D(unlabeled_batch, return_features=True)

                        # Feature matching loss
                        loss_g = feature_matching_loss(real_features, fake_features)

                    # Backward pass
                    self.scaler.scale(loss_g).backward()
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()

                except Exception as e:
                    logger.error(f"Generator training step failed: {e}")
                    raise RuntimeError(f"Generator training failed at step {step}: {e}")

                # Accumulate losses
                total_loss_d += loss_d.item()
                total_loss_g += loss_g.item()
                total_loss_sup += loss_sup.item()
                total_loss_unlab += loss_unlab.item()
                total_loss_fake += loss_fake_val.item()
                total_loss_fm += loss_g.item()

                # Update progress bar
                loop.set_postfix({
                    "D": f"{loss_d.item():.3f}",
                    "G": f"{loss_g.item():.3f}"
                })

        except KeyboardInterrupt:
            logger.warning(f"Training interrupted at epoch {epoch}")
            raise
        except Exception as e:
            logger.error(f"Training epoch {epoch} failed: {e}", exc_info=True)
            raise

        # Compute averages
        num_batches = len(self.unlabeled_loader)
        avg_loss_d = total_loss_d / num_batches
        avg_loss_g = total_loss_g / num_batches
        avg_loss_sup = total_loss_sup / num_batches
        avg_loss_unlab = total_loss_unlab / num_batches
        avg_loss_fake = total_loss_fake / num_batches
        avg_loss_fm = total_loss_fm / num_batches

        # Log to TensorBoard
        self.writer.add_scalar("Loss/Discriminator", avg_loss_d, epoch)
        self.writer.add_scalar("Loss/Generator", avg_loss_g, epoch)
        self.writer.add_scalar("Loss/Supervised", avg_loss_sup, epoch)
        self.writer.add_scalar("Loss/UnlabeledReal", avg_loss_unlab, epoch)
        self.writer.add_scalar("Loss/Fake", avg_loss_fake, epoch)
        self.writer.add_scalar("Loss/FeatureMatching", avg_loss_fm, epoch)

        # Save samples and checkpoints
        if (epoch + 1) % self.config.output.sample_interval == 0:
            self.save_samples(epoch)

        if (epoch + 1) % self.config.output.save_interval == 0:
            self.save_checkpoint(epoch, avg_loss_d, avg_loss_g)

        logger.debug(
            f"Epoch {epoch} - Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f}"
        )

        return avg_loss_d, avg_loss_g

    def save_samples(self, epoch: int) -> None:
        """
        Generate and save sample images.

        Args:
            epoch: Epoch number for naming
        """
        self.G.eval()

        try:
            with torch.no_grad():
                z = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
                fake = self.G(z)

                sample_path = self.sample_dir / f"epoch_{epoch:06d}.png"
                save_generated_images(fake, str(sample_path))

            logger.debug(f"Saved sample images at epoch {epoch}")
        except Exception as e:
            logger.error(f"Failed to save samples at epoch {epoch}: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        loss_d: float,
        loss_g: float
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            epoch: Epoch number
            loss_d: Discriminator loss
            loss_g: Generator loss
        """
        try:
            checkpoint = {
                "epoch": epoch,
                "generator_state_dict": self.G.state_dict(),
                "discriminator_state_dict": self.D.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "loss_d": loss_d,
                "loss_g": loss_g,
            }

            # Save latest checkpoint
            latest_path = self.checkpoint_dir / "latest.pt"
            torch.save(checkpoint, str(latest_path))
            logger.debug(f"Saved latest checkpoint: {latest_path}")

            # Save best generator
            if loss_g < self.best_g_loss:
                self.best_g_loss = loss_g
                self.best_epoch = epoch
                best_path = self.checkpoint_dir / "best_generator.pt"
                torch.save(checkpoint, str(best_path))
                logger.info(
                    f"New best generator at epoch {epoch} "
                    f"(Loss_G: {loss_g:.4f})"
                )

            # Periodic checkpoint
            if (epoch + 1) % 50 == 0:
                periodic_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:06d}.pt"
                torch.save(checkpoint, str(periodic_path))
                logger.debug(f"Saved periodic checkpoint: {periodic_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Resume epoch number

        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)

            self.G.load_state_dict(checkpoint["generator_state_dict"])
            self.D.load_state_dict(checkpoint["discriminator_state_dict"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            loss_g = checkpoint.get("loss_g", float("inf"))
            self.best_g_loss = loss_g

            logger.info(f"Loaded checkpoint from epoch {epoch}")
            return epoch + 1

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")