"""
Unit tests for SGAN training losses.
"""

import pytest
import torch
from training.losses import (
    supervised_loss,
    unlabeled_real_loss,
    fake_loss,
    feature_matching_loss
)


class TestLosses:
    """Test suite for loss functions."""

    @pytest.fixture
    def logits_3class(self) -> torch.Tensor:
        """Create dummy 3-class logits."""
        return torch.randn(32, 3)

    @pytest.fixture
    def labels_binary(self) -> torch.Tensor:
        """Create dummy binary labels."""
        return torch.randint(0, 2, (32,))

    @pytest.fixture
    def features(self) -> torch.Tensor:
        """Create dummy feature maps."""
        return torch.randn(32, 256)

    def test_supervised_loss_shape(self, logits_3class, labels_binary):
        """Test supervised loss returns scalar."""
        loss = supervised_loss(logits_3class, labels_binary)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_supervised_loss_value_range(self, logits_3class, labels_binary):
        """Test supervised loss is reasonable value."""
        loss = supervised_loss(logits_3class, labels_binary)
        assert 0 <= loss.item() < 10  # Reasonable range for cross-entropy

    def test_unlabeled_real_loss_shape(self, logits_3class):
        """Test unlabeled real loss returns scalar."""
        loss = unlabeled_real_loss(logits_3class)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_fake_loss_shape(self, logits_3class):
        """Test fake loss returns scalar."""
        loss = fake_loss(logits_3class)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_feature_matching_loss_shape(self, features):
        """Test feature matching loss returns scalar."""
        real_features = torch.randn(32, 256)
        fake_features = torch.randn(32, 256)
        loss = feature_matching_loss(real_features, fake_features)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_feature_matching_loss_zero_when_equal(self):
        """Test feature matching loss is zero when features are equal."""
        features = torch.randn(32, 256)
        loss = feature_matching_loss(features, features)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_supervised_loss_gradients(self, logits_3class, labels_binary):
        """Test supervised loss computes gradients."""
        logits_3class.requires_grad = True
        loss = supervised_loss(logits_3class, labels_binary)
        loss.backward()
        assert logits_3class.grad is not None
        assert logits_3class.grad.shape == logits_3class.shape
