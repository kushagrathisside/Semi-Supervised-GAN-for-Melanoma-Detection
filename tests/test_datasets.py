"""
Unit tests for dataset loading and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import torch

from datasets.melanoma_dataset import (
    LabeledMelanomaDataset,
    UnlabeledMelanomaDataset
)


class TestLabeledMelanomaDataset:
    """Test labeled dataset with error handling."""

    @pytest.fixture
    def dataset_dir(self):
        """Create temporary dataset directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create subdirectories
            (tmppath / "benign").mkdir()
            (tmppath / "malignant").mkdir()

            # Create dummy images
            for i in range(3):
                img = Image.new('RGB', (64, 64), color='red')
                img.save(tmppath / "benign" / f"benign_{i}.jpg")

            for i in range(2):
                img = Image.new('RGB', (64, 64), color='blue')
                img.save(tmppath / "malignant" / f"malignant_{i}.jpg")

            yield str(tmppath)

    def test_load_valid_dataset(self, dataset_dir):
        """Test loading valid dataset."""
        dataset = LabeledMelanomaDataset(dataset_dir)
        assert len(dataset) == 5
        assert len([s for s in dataset.samples if s[1] == 0]) == 3
        assert len([s for s in dataset.samples if s[1] == 1]) == 2

    def test_get_item_returns_tuple(self, dataset_dir):
        """Test __getitem__ returns (image, label) tuple."""
        dataset = LabeledMelanomaDataset(dataset_dir)
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert label in [0, 1]

    def test_missing_benign_dir_raises(self):
        """Test missing benign directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "malignant").mkdir()
            with pytest.raises(FileNotFoundError):
                LabeledMelanomaDataset(str(tmppath))

    def test_missing_malignant_dir_raises(self):
        """Test missing malignant directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "benign").mkdir()
            with pytest.raises(FileNotFoundError):
                LabeledMelanomaDataset(str(tmppath))

    def test_empty_dataset_raises(self):
        """Test empty dataset raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "benign").mkdir()
            (tmppath / "malignant").mkdir()
            with pytest.raises(ValueError):
                LabeledMelanomaDataset(str(tmppath))

    def test_invalid_image_in_batch(self, dataset_dir):
        """Test corrupted image raises error on load."""
        dataset = LabeledMelanomaDataset(dataset_dir)

        # Corrupt an image
        benign_files = list((Path(dataset_dir) / "benign").glob("*.jpg"))
        with open(benign_files[0], 'w') as f:
            f.write("corrupted data")

        with pytest.raises(RuntimeError):
            dataset[0]


class TestUnlabeledMelanomaDataset:
    """Test unlabeled dataset with error handling."""

    @pytest.fixture
    def unlabeled_dir(self):
        """Create temporary unlabeled dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create dummy images
            for i in range(5):
                img = Image.new('RGB', (64, 64), color='green')
                img.save(tmppath / f"image_{i}.jpg")

            yield str(tmppath)

    def test_load_valid_unlabeled_dataset(self, unlabeled_dir):
        """Test loading valid unlabeled dataset."""
        dataset = UnlabeledMelanomaDataset(unlabeled_dir)
        assert len(dataset) == 5

    def test_get_item_returns_tensor(self, unlabeled_dir):
        """Test __getitem__ returns image tensor."""
        dataset = UnlabeledMelanomaDataset(unlabeled_dir)
        img = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)

    def test_missing_directory_raises(self):
        """Test missing directory raises error."""
        with pytest.raises(FileNotFoundError):
            UnlabeledMelanomaDataset("nonexistent/directory")

    def test_empty_directory_raises(self):
        """Test empty directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                UnlabeledMelanomaDataset(str(tmpdir))

    def test_filters_non_image_files(self):
        """Test non-image files are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create images and non-image files
            img = Image.new('RGB', (64, 64))
            img.save(tmppath / "image_1.jpg")
            img.save(tmppath / "image_2.png")

            # Create non-image files
            (tmppath / "readme.txt").write_text("some text")
            (tmppath / "data.csv").write_text("col1,col2")

            dataset = UnlabeledMelanomaDataset(str(tmppath))
            assert len(dataset) == 2

    def test_various_image_formats(self):
        """Test support for various image formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            formats = [('jpg', 'JPEG'), ('png', 'PNG'), ('bmp', 'BMP')]
            for ext, pil_format in formats:
                img = Image.new('RGB', (64, 64))
                img.save(tmppath / f"image.{ext}", format=pil_format)

            dataset = UnlabeledMelanomaDataset(str(tmppath))
            assert len(dataset) == 3
