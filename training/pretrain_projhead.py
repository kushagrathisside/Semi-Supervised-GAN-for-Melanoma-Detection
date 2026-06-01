"""
Standalone script to pre-train the DINOProjectionHead on labeled dermoscopy images
using Supervised Contrastive loss.

Run BEFORE main GAN training:
    python training/pretrain_projhead.py

What this does:
  1. Loads all 200 labeled images (100 benign + 100 malignant)
  2. Extracts frozen DINOv3 mean patch features for each image
  3. Trains a 2-layer MLP (DINOProjectionHead) with SupCon loss to maximize
     the angular margin between benign and malignant clusters in 256-dim space
  4. Verifies within-class variance is non-trivial (avoids representation collapse)
  5. Saves weights to outputs/projection_head.pt

Memory note: DINO forward only (no backward), batch=200 images at 112x112.
DINOv3-B/16 at 112x112 → 49 patch tokens, ~0.73 GB fp16. Safe alongside other processes.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.melanoma_dataset import LabeledMelanomaDataset
from models.projection_head import DINOProjectionHead
from utils.config import SGANConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
# ─────────────────────────────────────────────

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    """
    Args:
        features: [N, D] L2-normalized embeddings
        labels:   [N] integer class labels
        temperature: scaling factor; higher = softer distribution, less collapse risk
    Returns:
        Scalar loss.

    Numerical notes:
        - The diagonal (self-similarity) is excluded from the denominator by
          masking with -1e9 (not -inf) before logsumexp. Using -inf causes
          0 * (-inf) = NaN in the positive-pair product since IEEE 754.
        - logsumexp is numerically stable across the full temperature range.
    """
    N = features.size(0)
    device = features.device

    sim = torch.matmul(features, features.T) / temperature  # [N, N]

    # Exclude self from denominator: use -1e9 (finite) so log_prob diagonal
    # is large-negative but finite — avoids 0 * (-inf) = NaN in the loss.
    self_mask = torch.eye(N, device=device, dtype=torch.bool)
    log_denom = torch.logsumexp(sim.masked_fill(self_mask, -1e9), dim=1, keepdim=True)
    log_prob = sim - log_denom  # [N, N]; diagonal is finite but will be zeroed by pos_mask

    # Positive mask: same class, not self
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float().masked_fill(self_mask, 0)  # [N, N]

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
    return loss.mean()


# ─────────────────────────────────────────────
# DINO feature extractor (no grad, patch mean)
# ─────────────────────────────────────────────

def extract_dino_features(
    images: torch.Tensor,
    dino: nn.Module,
    input_size: int = 112,
) -> torch.Tensor:
    """
    Args:
        images:     [B, 3, 64, 64] in [-1, 1]
        dino:       frozen DINO model (DINOv2 or DINOv3)
        input_size: resize target; must be multiple of patch_size
                    (16 for DINOv3, 14 for DINOv2). 112 works for both.
    Returns:
        [B, 768] mean of patch tokens.
        DINOv3 register tokens are in 'x_storage_tokens' (separate key) and
        are NOT included in 'x_norm_patchtokens' — no special handling needed.
    """
    x = (images + 1.0) / 2.0                                       # → [0, 1]
    x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    with torch.no_grad():
        out = dino.forward_features(x)
        # forward_features returns a dict with 'x_norm_patchtokens' in both
        # DINOv2 and DINOv3. DINOv3 also adds 'x_storage_tokens' (register
        # tokens) but those are separate and excluded from patchtokens.
        patch_tokens = out['x_norm_patchtokens']  # [B, num_patches, 768]

    return patch_tokens.mean(dim=1)  # [B, 768]


# ─────────────────────────────────────────────
# Main pre-training routine
# ─────────────────────────────────────────────

def pretrain(config_path: str = "configs/config.yaml") -> None:
    config = SGANConfig.from_yaml(config_path)

    dino_cfg = config.dino
    if dino_cfg is None:
        raise ValueError("No [dino] section in config.yaml. Add it before pre-training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load all 200 labeled images as a single batch ──────────────────────
    dataset = LabeledMelanomaDataset(
        root_dir=config.dataset.labeled_path,
        image_size=config.dataset.image_size,
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=4)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    logger.info(f"Loaded {len(dataset)} labeled images. "
                f"Class distribution: {labels.bincount().tolist()}")

    # ── Load frozen DINO (supports DINOv2 and DINOv3) ─────────────────────
    hub_repo = "facebookresearch/dinov3" if dino_cfg.model.startswith("dinov3") \
               else "facebookresearch/dinov2"
    logger.info(f"Loading {dino_cfg.model} from {hub_repo} ...")
    dino = torch.hub.load(hub_repo, dino_cfg.model)
    dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    # ── Extract DINO features once (frozen, so this is constant) ──────────
    logger.info("Extracting DINO patch features ...")
    with torch.no_grad():
        dino_features = extract_dino_features(images, dino, input_size=dino_cfg.input_size)
    logger.info(f"DINO features shape: {dino_features.shape}")  # [200, 768]

    # ── Initialize projection head ─────────────────────────────────────────
    proj_head = DINOProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    optimizer = torch.optim.Adam(proj_head.parameters(), lr=dino_cfg.pretrain_lr)

    # ── SupCon training loop ───────────────────────────────────────────────
    logger.info(f"Pre-training projection head for {dino_cfg.pretrain_epochs} epochs ...")
    for epoch in range(dino_cfg.pretrain_epochs):
        proj_head.train()
        optimizer.zero_grad()

        embeddings = proj_head(dino_features)          # [200, 256], L2-normalized
        loss = supcon_loss(embeddings, labels, temperature=dino_cfg.supcon_temperature)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch + 1}/{dino_cfg.pretrain_epochs}  loss={loss.item():.4f}")

    # ── Verify within-class variance (collapse check) ─────────────────────
    proj_head.eval()
    with torch.no_grad():
        final_embeddings = proj_head(dino_features)

    for c in range(config.dataset.num_classes):
        mask = labels == c
        class_emb = final_embeddings[mask]
        within_std = class_emb.std(dim=0).mean().item()
        logger.info(f"  Class {c} within-class std: {within_std:.4f} "
                    f"({'OK' if within_std > 0.05 else 'WARNING: possible collapse'})")

    # ── Save weights ───────────────────────────────────────────────────────
    out_path = Path(dino_cfg.projection_head_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj_head.state_dict(), str(out_path))
    logger.info(f"Saved projection head weights → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    from utils.logging_config import setup_logging
    setup_logging(log_dir="outputs/logs", log_file="pretrain_projhead.log")

    pretrain(args.config)
