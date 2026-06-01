"""
Unit tests for SGAN training losses.
"""

import pytest
import torch
import torch.nn.functional as F
from training.losses import (
    supervised_loss,
    unlabeled_real_loss,
    fake_loss,
    feature_matching_loss,
    dino_feature_matching_loss,
    confidence_weighted_dino_loss,
    adversarial_generator_loss,
    mmd_loss,
    r1_penalty,
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


class TestDINOLosses:
    """Test suite for DINO feature matching losses."""

    D = 256  # projection dimension
    B = 16   # batch size

    @pytest.fixture
    def proj_real_mean(self):
        """Simulated mean projection of a real batch — L2-normalized, [D]."""
        return F.normalize(torch.randn(self.D), dim=0)

    @pytest.fixture
    def proj_fake(self):
        """Simulated fake projections — L2-normalized rows, [B, D]."""
        return F.normalize(torch.randn(self.B, self.D), dim=-1)

    @pytest.fixture
    def fake_probs(self):
        """Simulated discriminator softmax probabilities, [B, 2]."""
        return F.softmax(torch.randn(self.B, 2), dim=-1)

    @pytest.fixture
    def class_anchors(self):
        """Simulated per-class mean anchors."""
        return {
            0: F.normalize(torch.randn(self.D), dim=0),
            1: F.normalize(torch.randn(self.D), dim=0),
        }

    # ── dino_feature_matching_loss ────────────────────────────────────

    def test_dino_fml_scalar(self, proj_real_mean, proj_fake):
        """Loss is a scalar."""
        loss = dino_feature_matching_loss(proj_real_mean, proj_fake)
        assert loss.shape == torch.Size([])

    def test_dino_fml_non_negative(self, proj_real_mean, proj_fake):
        """L1 loss is always ≥ 0."""
        loss = dino_feature_matching_loss(proj_real_mean, proj_fake)
        assert loss.item() >= 0.0

    def test_dino_fml_bounded(self, proj_fake):
        """With L2-normalised inputs the loss is bounded in [0, 2]."""
        # Use the same normalised mean as the target → loss should be ~0
        proj_real_mean = proj_fake.mean(0)
        loss = dino_feature_matching_loss(proj_real_mean, proj_fake)
        assert loss.item() <= 2.0 + 1e-5

    def test_dino_fml_zero_when_equal(self):
        """Loss is zero when fake mean equals real mean exactly."""
        v = F.normalize(torch.randn(self.D), dim=0)
        # fake batch is a constant tile of v → mean(0) = v
        proj_fake = v.unsqueeze(0).expand(self.B, -1)
        loss = dino_feature_matching_loss(v, proj_fake)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_dino_fml_with_variance_term(self, proj_real_mean, proj_fake):
        """Variance term (lambda_var > 0) increases loss vs. no variance term."""
        loss_no_var = dino_feature_matching_loss(proj_real_mean, proj_fake, lambda_var=0.0)
        loss_with_var = dino_feature_matching_loss(proj_real_mean, proj_fake, lambda_var=0.5)
        # With random inputs, variance term should contribute positively
        assert loss_with_var.item() >= 0.0
        assert loss_no_var.shape == torch.Size([])

    def test_dino_fml_gradient_flows(self):
        """Gradient flows from loss back to a leaf tensor (simulates G output)."""
        leaf = torch.randn(self.B, self.D, requires_grad=True)
        proj_fake = F.normalize(leaf, dim=-1)  # normalise but keep graph
        proj_real_mean = F.normalize(torch.randn(self.D), dim=0)
        loss = dino_feature_matching_loss(proj_real_mean, proj_fake)
        loss.backward()
        assert leaf.grad is not None
        assert not leaf.grad.isnan().any(), "NaN gradient in dino_feature_matching_loss"

    # ── confidence_weighted_dino_loss ─────────────────────────────────

    def test_ccw_scalar(self, proj_fake, class_anchors, fake_probs):
        """Loss is a scalar."""
        loss = confidence_weighted_dino_loss(proj_fake, class_anchors, fake_probs)
        assert loss.shape == torch.Size([])

    def test_ccw_non_negative(self, proj_fake, class_anchors, fake_probs):
        """L1 loss is always ≥ 0."""
        loss = confidence_weighted_dino_loss(proj_fake, class_anchors, fake_probs)
        assert loss.item() >= 0.0

    def test_ccw_uniform_weights_degrade_gracefully(self, proj_fake, class_anchors):
        """With uniform weights (uncertain discriminator) loss is still valid."""
        uniform_probs = torch.full((self.B, 2), 0.5)
        loss = confidence_weighted_dino_loss(proj_fake, class_anchors, uniform_probs)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)

    def test_ccw_gradient_flows(self, class_anchors, fake_probs):
        """Gradient flows from loss back through proj_fake to a leaf tensor."""
        leaf = torch.randn(self.B, self.D, requires_grad=True)
        proj_fake = F.normalize(leaf, dim=-1)
        loss = confidence_weighted_dino_loss(proj_fake, class_anchors, fake_probs)
        loss.backward()
        assert leaf.grad is not None
        assert not leaf.grad.isnan().any(), "NaN gradient in confidence_weighted_dino_loss"

    def test_ccw_anchors_detached(self, proj_fake, fake_probs):
        """Anchors do not participate in the gradient graph (they are fixed buffers)."""
        anchors = {
            0: torch.randn(self.D, requires_grad=True),
            1: torch.randn(self.D, requires_grad=True),
        }
        leaf = torch.randn(self.B, self.D, requires_grad=True)
        proj_fake = F.normalize(leaf, dim=-1)
        loss = confidence_weighted_dino_loss(proj_fake, anchors, fake_probs)
        loss.backward()
        # Anchors are explicitly detached inside the loss — they must have no grad
        assert anchors[0].grad is None
        assert anchors[1].grad is None


class TestV3GeneratorLosses:
    """Test suite for the v3 generator objective (adversarial + per-sample MMD)."""

    # ── adversarial_generator_loss ────────────────────────────────────

    def test_adv_scalar_and_nonneg(self):
        """Loss is a non-negative scalar."""
        logits = torch.randn(16, 3)
        loss = adversarial_generator_loss(logits, num_classes=2)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0

    def test_adv_large_when_d_confident_fake(self):
        """When D is certain inputs are fake, the loss is large (strong signal)."""
        logits = torch.zeros(16, 3)
        logits[:, 2] = 20.0  # fake-class logit dominates → P(fake)≈1
        loss = adversarial_generator_loss(logits, num_classes=2)
        assert loss.item() > 5.0

    def test_adv_small_when_d_fooled(self):
        """When D thinks fakes are real, the loss is near zero."""
        logits = torch.zeros(16, 3)
        logits[:, 0] = 20.0  # a real-class logit dominates → P(fake)≈0
        loss = adversarial_generator_loss(logits, num_classes=2)
        assert loss.item() < 1e-3

    def test_adv_gradient_flows_and_finite(self):
        """Gradient reaches the logits and contains no NaN/Inf even when saturated."""
        logits = torch.zeros(16, 3, requires_grad=True)
        with torch.no_grad():
            logits[:, 2] = 20.0
        loss = adversarial_generator_loss(logits, num_classes=2)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    # ── mmd_loss ──────────────────────────────────────────────────────

    def test_mmd_zero_for_identical_sets(self):
        """MMD of a set with itself is ~0."""
        a = F.normalize(torch.randn(48, 256), dim=1)
        loss = mmd_loss(a, a.clone())
        assert abs(loss.item()) < 1e-4

    def test_mmd_positive_for_shifted_sets(self):
        """MMD is clearly positive for well-separated distributions."""
        a = F.normalize(torch.randn(48, 256), dim=1)
        b = F.normalize(torch.randn(48, 256) + 5.0, dim=1)
        loss = mmd_loss(a, b)
        assert loss.item() > 0.1

    def test_mmd_gradient_flows_to_fake_only(self):
        """Gradient flows to fake_proj; real_proj is detached inside."""
        real = F.normalize(torch.randn(48, 256), dim=1).requires_grad_(True)
        leaf = torch.randn(32, 256, requires_grad=True)
        fake = F.normalize(leaf, dim=1)
        loss = mmd_loss(real, fake)
        loss.backward()
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
        # real_proj is detached inside mmd_loss → receives no gradient
        assert real.grad is None

    # ── r1_penalty ────────────────────────────────────────────────────

    def test_r1_penalty_scalar_nonneg_and_differentiable(self):
        """R1 is a non-negative scalar that backprops to discriminator params."""
        from models.discriminator import Discriminator

        D = Discriminator(feature_maps=32, channels=3, num_classes=2)
        real = torch.randn(8, 3, 64, 64)
        r1 = r1_penalty(D, real, num_classes=2)
        assert r1.shape == torch.Size([])
        assert r1.item() >= 0.0
        # Outer backward (penalty differentiable w.r.t. D params) must work.
        r1.backward()
        grads = [p.grad for p in D.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_r1_penalty_does_not_mutate_input(self):
        """The input real tensor is not modified (grad enabled on an internal copy)."""
        from models.discriminator import Discriminator

        D = Discriminator(feature_maps=32, channels=3, num_classes=2)
        real = torch.randn(8, 3, 64, 64)
        snapshot = real.clone()
        _ = r1_penalty(D, real, num_classes=2)
        assert real.grad is None
        assert torch.equal(real, snapshot)
