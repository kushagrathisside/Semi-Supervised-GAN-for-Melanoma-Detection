"""
Training orchestrator for SGAN model.

Handles epoch-level training, loss computation, checkpoint management, and monitoring.
"""

import contextlib
from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.losses import (
    supervised_loss,
    unlabeled_real_loss,
    fake_loss,
    feature_matching_loss,
    dino_feature_matching_loss,
    confidence_weighted_dino_loss,
)
from evaluation.metrics import compute_auc_roc
from models.projection_head import DINOProjectionHead
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
        self.best_d_acc = 0.0      # labeled-set classification accuracy
        self.best_epoch = -1

        # DINO feature matching (optional — only active when config.dino is set)
        self.dino = None
        self.proj_head = None
        self.class_anchors: Dict[int, torch.Tensor] = {}
        self._use_dino = config.dino is not None

        if self._use_dino:
            self._init_dino()

    # ──────────────────────────────────────────────────────────────────
    # DINO helpers (supports DINOv2 and DINOv3)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _dino_hub_repo(model_name: str) -> str:
        """Resolve the correct facebookresearch hub repo for a model name."""
        if model_name.startswith("dinov3"):
            return "facebookresearch/dinov3"
        return "facebookresearch/dinov2"

    def _init_dino(self) -> None:
        """Load frozen DINO model and the pre-trained projection head."""
        cfg = self.config.dino
        hub_repo = self._dino_hub_repo(cfg.model)
        logger.info(f"Loading frozen {cfg.model} from {hub_repo} ...")
        self.dino = torch.hub.load(hub_repo, cfg.model)
        self.dino.to(self.device).eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

        proj_path = Path(cfg.projection_head_path)
        if not proj_path.exists():
            raise FileNotFoundError(
                f"Projection head not found: {proj_path}. "
                "Run `python training/pretrain_projhead.py` first."
            )
        self.proj_head = DINOProjectionHead(input_dim=768, hidden_dim=512, output_dim=256)
        self.proj_head.load_state_dict(
            torch.load(str(proj_path), map_location=self.device)
        )
        self.proj_head.to(self.device).eval()
        for p in self.proj_head.parameters():
            p.requires_grad_(False)

        logger.info("Pre-computing fixed class anchors from all labeled images ...")
        self.class_anchors = self._precompute_anchors()
        logger.info(f"DINO feature matching ready ({cfg.model}).")

    def _get_dino_proj(self, images: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        """
        Map images to the 256-dim SupCon-adapted DINO projection space.

        Works with both DINOv2 (patch_size=14) and DINOv3 (patch_size=16).
        DINOv3 register tokens are in 'x_storage_tokens' and are automatically
        excluded from 'x_norm_patchtokens' — no special handling needed here.

        Args:
            images:  [B, 3, H, W] in [-1, 1] (training resolution, typically 64x64)
            no_grad: if True, wrap DINO+proj_head in torch.no_grad().
                     Use True for real images (anchor computation, mean reference).
                     Use False for fake images so gradient reaches G.
        Returns:
            [B, 256] L2-normalized projections.
            DINOv3-B/16 at input_size=112: patch_tokens shape [B, 49, 768].
            DINOv2-B/14 at input_size=112: patch_tokens shape [B, 64, 768].
        """
        cfg = self.config.dino
        x = (images + 1.0) / 2.0
        x = F.interpolate(x, size=cfg.input_size, mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            out = self.dino.forward_features(x)
            patch_tokens = out['x_norm_patchtokens']  # [B, num_patches, 768]
            patch_mean = patch_tokens.mean(dim=1)      # [B, 768]

        return self.proj_head(patch_mean)              # [B, 256]

    def _precompute_anchors(self) -> Dict[int, torch.Tensor]:
        """
        Compute fixed per-class mean projections from ALL labeled images.

        These are stored as plain tensors (not nn.Parameter) and never updated
        after initialization. Using the full labeled set (not per-batch samples)
        gives a stable, low-variance anchor for the class-conditional matching term.
        """
        per_class: Dict[int, list] = {}
        with torch.no_grad():
            for imgs, lbls in self.labeled_loader:
                imgs = imgs.to(self.device)
                proj = self._get_dino_proj(imgs, no_grad=True)
                for c in range(self.config.dataset.num_classes):
                    mask = lbls == c
                    if mask.sum() > 0:
                        per_class.setdefault(c, []).append(proj[mask].cpu())

        anchors = {}
        for c, vecs in per_class.items():
            anchors[c] = torch.cat(vecs, dim=0).mean(0).to(self.device)
            logger.debug(f"  Anchor class {c}: norm={anchors[c].norm().item():.3f}")
        return anchors

    def _effective_lambda_cc(self, epoch: int) -> float:
        """
        Linear warmup for the class-conditional loss weight.

        The discriminator needs several epochs to learn class probabilities
        before its softmax outputs can serve as meaningful assignment weights.
        Activating the class-conditional term too early causes the generator
        to be pulled toward the midpoint of both class anchors simultaneously.
        """
        cfg = self.config.dino
        if epoch < cfg.lambda_cc_warmup_start:
            return 0.0
        ramp = min(1.0, (epoch - cfg.lambda_cc_warmup_start) / cfg.lambda_cc_warmup_epochs)
        return cfg.lambda_cc * ramp

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
        total_correct = 0
        total_labeled = 0
        all_logits_labeled: list = []
        all_labels: list = []

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

                        # Track classification accuracy and collect logits for AUC-ROC
                        with torch.no_grad():
                            preds = logits_labeled[:, :self.config.dataset.num_classes].argmax(dim=1)
                            total_correct += (preds == labels).sum().item()
                            total_labeled += labels.size(0)
                            all_logits_labeled.append(logits_labeled.detach().cpu())
                            all_labels.append(labels.detach().cpu())

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
                    self.scaler.unscale_(self.optimizer_d)
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
                        if self._use_dino:
                            # ── DINO-guided feature matching ────────────────────
                            # Real: no grad needed (reference distribution)
                            proj_real_mean = self._get_dino_proj(
                                unlabeled_batch, no_grad=True
                            ).mean(0)

                            # Fake: grad must flow through DINO → proj_head → fake → G
                            proj_fake = self._get_dino_proj(fake_imgs, no_grad=False)

                            loss_g = dino_feature_matching_loss(
                                proj_real_mean,
                                proj_fake,
                                lambda_var=self.config.dino.lambda_var,
                            )

                            # Class-conditional term (warmed up, zero in early epochs)
                            lcc = self._effective_lambda_cc(epoch)
                            if lcc > 0.0:
                                # Discriminator confidence as soft class weights (detached)
                                fake_probs = F.softmax(
                                    self.D(fake_imgs.detach())[:, :self.config.dataset.num_classes],
                                    dim=-1,
                                )
                                loss_g = loss_g + lcc * confidence_weighted_dino_loss(
                                    proj_fake, self.class_anchors, fake_probs
                                )

                            # NaN guard: DINO backprop can produce NaN under heavy AMP scaling
                            if torch.isnan(loss_g):
                                logger.warning(f"NaN in DINO generator loss at step {step}; skipping update")
                                self.optimizer_g.zero_grad()
                                continue

                        else:
                            # ── Original discriminator feature matching ──────────
                            logits_fake, fake_features = self.D(fake_imgs, return_features=True)
                            _, real_features = self.D(unlabeled_batch, return_features=True)
                            loss_g = feature_matching_loss(real_features, fake_features)

                    # Backward + gradient clip (both paths — consistent with D clipping)
                    self.scaler.scale(loss_g).backward()
                    self.scaler.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=5.0)
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
        d_acc = total_correct / max(total_labeled, 1)

        # Log to TensorBoard
        self.writer.add_scalar("Loss/Discriminator", avg_loss_d, epoch)
        self.writer.add_scalar("Loss/Generator", avg_loss_g, epoch)
        self.writer.add_scalar("Loss/Supervised", avg_loss_sup, epoch)
        self.writer.add_scalar("Loss/UnlabeledReal", avg_loss_unlab, epoch)
        self.writer.add_scalar("Loss/Fake", avg_loss_fake, epoch)
        self.writer.add_scalar("Loss/FeatureMatching", avg_loss_fm, epoch)
        # AUC-ROC over all labeled batches seen this epoch
        if all_logits_labeled:
            all_logits_cat = torch.cat(all_logits_labeled, dim=0)
            all_labels_cat = torch.cat(all_labels, dim=0)
            auc = compute_auc_roc(all_logits_cat, all_labels_cat)
        else:
            auc = 0.5

        self.writer.add_scalar("Training/D_LabeledAcc", d_acc, epoch)
        self.writer.add_scalar("Training/D_AUC_ROC", auc, epoch)
        if self._use_dino:
            self.writer.add_scalar(
                "Training/DINO_lambda_cc", self._effective_lambda_cc(epoch), epoch
            )

        # Save samples and checkpoints
        if (epoch + 1) % self.config.output.sample_interval == 0:
            self.save_samples(epoch)

        if (epoch + 1) % self.config.output.save_interval == 0:
            self.save_checkpoint(epoch, avg_loss_d, avg_loss_g, d_acc)

        logger.debug(
            f"Epoch {epoch} - Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f}, "
            f"D_LabeledAcc: {d_acc:.4f}"
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
        loss_g: float,
        d_acc: float = 0.0,
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            epoch:  Epoch number
            loss_d: Discriminator loss
            loss_g: Generator loss
            d_acc:  Discriminator accuracy on labeled images (used for best-model selection)
        """
        try:
            checkpoint = {
                "epoch": epoch,
                "generator_state_dict": self.G.state_dict(),
                "discriminator_state_dict": self.D.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None else None,
                "loss_d": loss_d,
                "loss_g": loss_g,
                "best_g_loss": self.best_g_loss,
                "best_d_acc": self.best_d_acc,
            }

            # Save latest checkpoint
            latest_path = self.checkpoint_dir / "latest.pt"
            torch.save(checkpoint, str(latest_path))
            logger.debug(f"Saved latest checkpoint: {latest_path}")

            # Save best generator — selected by labeled-set classification accuracy,
            # not by G loss. G loss is a training signal and is not monotonically
            # correlated with generation quality or downstream classifier utility.
            if d_acc > self.best_d_acc:
                self.best_d_acc = d_acc
                self.best_epoch = epoch
                best_path = self.checkpoint_dir / "best_generator.pt"
                torch.save(checkpoint, str(best_path))
                logger.info(
                    f"New best checkpoint at epoch {epoch} "
                    f"(D_LabeledAcc: {d_acc:.4f})"
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

            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if "lr_scheduler_state_dict" in checkpoint and checkpoint["lr_scheduler_state_dict"] is not None:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            self.best_g_loss = checkpoint.get("best_g_loss", float("inf"))
            self.best_d_acc = checkpoint.get("best_d_acc", 0.0)

            logger.info(f"Loaded checkpoint from epoch {epoch}")
            return epoch + 1

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")