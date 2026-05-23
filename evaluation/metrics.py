"""
Enhanced evaluation metrics for generative models.

Provides FID score, Inception Score, and discriminator classification metrics.
"""

import subprocess
import re
from typing import Optional, Tuple
import numpy as np
import torch
from pathlib import Path


def compute_fid(real_dir: str, fake_dir: str) -> Optional[float]:
    """
    Compute Fréchet Inception Distance (FID) between real and fake images.

    Lower FID indicates better quality and diversity of generated images.

    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing fake/generated images

    Returns:
        FID score or None if computation fails

    Raises:
        RuntimeError: If pytorch_fid is not installed
    """
    try:
        cmd = [
            "python",
            "-m",
            "pytorch_fid",
            real_dir,
            fake_dir
        ]

        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        output = result.decode()

        match = re.search(r"FID:\s*([0-9\.]+)", output)
        if match:
            fid = float(match.group(1))
            return fid
        else:
            raise RuntimeError("FID value not found in output")

    except subprocess.CalledProcessError as e:
        print(f"FID computation failed: {e.output.decode()}")
        return None
    except FileNotFoundError:
        raise RuntimeError("pytorch_fid not found. Install with: pip install pytorch-fid")


def compute_inception_score(
    images: torch.Tensor,
    batch_size: int = 32,
    resize: bool = True
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS) of generated images.

    Higher IS indicates better quality and diversity.

    Args:
        images: Tensor of images (N, C, H, W) with values in [0, 1]
        batch_size: Batch size for processing
        resize: Whether to resize images to 299x299 for Inception network

    Returns:
        Tuple of (IS_mean, IS_std)

    Note:
        Requires torchvision's Inception v3 model.
        Inception score alone doesn't guarantee good images, combine with FID.
    """
    try:
        from torchvision.models import inception_v3
        from torch.nn.functional import softmax
    except ImportError:
        raise RuntimeError("torchvision required for Inception Score computation")

    device = images.device
    inception_model = inception_v3(pretrained=True, transform_input=True).to(device)
    inception_model.eval()

    # Resize if needed
    if resize and images.shape[-1] != 299:
        from torch.nn.functional import interpolate
        images = interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

    # Ensure images are in [0, 1]
    images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
    images = torch.clamp(images, 0, 1)

    scores = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Get predictions
            logits = inception_model(batch)
            probs = softmax(logits, dim=1)

            # Compute score
            py = probs.mean(dim=0)
            pyx = probs
            scores_batch = (pyx * (torch.log(pyx + 1e-8) - torch.log(py + 1e-8))).sum(dim=1)
            scores.append(scores_batch.cpu().numpy())

    scores = np.concatenate(scores)
    is_mean = np.exp(scores.mean())
    is_std = np.exp(scores.std())

    return float(is_mean), float(is_std)


def compute_precision_recall(
    discriminator_logits: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for binary classification.

    Args:
        discriminator_logits: Model output logits (N, num_classes)
        labels: Ground truth labels (N,)

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    predictions = torch.argmax(discriminator_logits, dim=1)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # For binary classification (0: benign, 1: malignant)
    if predictions.max() == 1:
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return float(precision), float(recall), float(f1)
    else:
        raise ValueError("Expected binary classification task")


def compute_class_wise_accuracy(
    discriminator_logits: torch.Tensor,
    labels: torch.Tensor
) -> dict:
    """
    Compute per-class accuracy metrics.

    Args:
        discriminator_logits: Model output logits (N, num_classes)
        labels: Ground truth labels (N,)

    Returns:
        Dictionary with per-class accuracy metrics
    """
    predictions = torch.argmax(discriminator_logits, dim=1)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    num_classes = discriminator_logits.shape[1]
    metrics = {}

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.sum() == 0:
            metrics[f"class_{class_idx}_accuracy"] = 0.0
        else:
            accuracy = (predictions[class_mask] == class_idx).mean()
            metrics[f"class_{class_idx}_accuracy"] = float(accuracy)

    # Overall accuracy
    overall_accuracy = (predictions == labels).mean()
    metrics["overall_accuracy"] = float(overall_accuracy)

    return metrics
