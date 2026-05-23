"""
Melanoma dataset loaders for labeled and unlabeled images.

Provides PyTorch Dataset classes for loading dermoscopy images
with automatic preprocessing and error handling.
"""

import os
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.transforms import get_transforms


class LabeledMelanomaDataset(Dataset):
    """
    Dataset for labeled melanoma images (benign/malignant).

    Loads images from separate benign/ and malignant/ directories
    and applies standard preprocessing transforms.

    Attributes:
        samples: List of (image_path, label) tuples
        transform: Image preprocessing pipeline

    Raises:
        FileNotFoundError: If benign/ or malignant/ directories don't exist
        ValueError: If no images found in directories
    """

    def __init__(self, root_dir: str, image_size: int = 64) -> None:
        """
        Initialize labeled dataset.

        Args:
            root_dir: Root directory containing 'benign' and 'malignant' subdirs
            image_size: Target image size for preprocessing (default: 64)

        Raises:
            FileNotFoundError: If required directories don't exist
            ValueError: If no images found
        """
        self.samples: List[Tuple[str, int]] = []
        self.image_size = image_size

        benign_dir = Path(root_dir) / "benign"
        malignant_dir = Path(root_dir) / "malignant"

        # Validate directories exist
        if not benign_dir.exists():
            raise FileNotFoundError(f"Benign directory not found: {benign_dir}")
        if not malignant_dir.exists():
            raise FileNotFoundError(f"Malignant directory not found: {malignant_dir}")

        # Load benign images (label=0)
        benign_images = list(benign_dir.glob("*"))
        for img_path in benign_images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.samples.append((str(img_path), 0))

        # Load malignant images (label=1)
        malignant_images = list(malignant_dir.glob("*"))
        for img_path in malignant_images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.samples.append((str(img_path), 1))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {root_dir}. "
                "Expected subdirectories: benign/, malignant/"
            )

        self.transform = get_transforms(image_size)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (preprocessed_image, label)

        Raises:
            RuntimeError: If image fails to load or process
        """
        try:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at index {idx} ({img_path}): {e}"
            )


class UnlabeledMelanomaDataset(Dataset):
    """
    Dataset for unlabeled melanoma images.

    Loads all images from a directory without labels.
    Useful for semi-supervised training with GAN models.

    Attributes:
        images: List of image paths
        transform: Image preprocessing pipeline

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no images found in directory
    """

    def __init__(self, root_dir: str, image_size: int = 64) -> None:
        """
        Initialize unlabeled dataset.

        Args:
            root_dir: Directory containing unlabeled images
            image_size: Target image size for preprocessing (default: 64)

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no images found
        """
        self.image_size = image_size
        root_path = Path(root_dir)

        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # Find all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.images: List[str] = [
            str(img_path)
            for img_path in root_path.iterdir()
            if img_path.suffix.lower() in valid_extensions
        ]

        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")

        self.transform = get_transforms(image_size)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Preprocessed image tensor

        Raises:
            RuntimeError: If image fails to load or process
        """
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at index {idx} ({img_path}): {e}"
            )
