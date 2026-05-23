"""
Unit tests for training utilities.
"""

import pytest
import torch
import torch.optim as optim
from utils.lr_scheduler import LRScheduler


class TestLRScheduler:
    """Test learning rate scheduler."""

    @pytest.fixture
    def optimizer_pair(self):
        """Create generator and discriminator optimizers."""
        model_g = torch.nn.Linear(10, 10)
        model_d = torch.nn.Linear(10, 10)
        opt_g = optim.Adam(model_g.parameters(), lr=0.0002)
        opt_d = optim.Adam(model_d.parameters(), lr=0.0001)
        return opt_g, opt_d

    def test_no_scheduler(self, optimizer_pair):
        """Test no scheduler mode."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(opt_g, opt_d, scheduler_type="none")

        initial_lr_g, initial_lr_d = scheduler.get_last_lr()
        scheduler.step()
        after_step_lr_g, after_step_lr_d = scheduler.get_last_lr()

        assert initial_lr_g == after_step_lr_g
        assert initial_lr_d == after_step_lr_d

    def test_step_scheduler(self, optimizer_pair):
        """Test step decay scheduler."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(
            opt_g, opt_d,
            scheduler_type="step",
            decay_steps=2,
            decay_factor=0.5
        )

        initial_lr_g, _ = scheduler.get_last_lr()

        # After 2 steps, LR should decay
        for _ in range(2):
            scheduler.step()

        after_decay_lr_g, _ = scheduler.get_last_lr()
        assert after_decay_lr_g < initial_lr_g

    def test_exponential_scheduler(self, optimizer_pair):
        """Test exponential decay scheduler."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(
            opt_g, opt_d,
            scheduler_type="exponential",
            decay_factor=0.9
        )

        initial_lr_g, _ = scheduler.get_last_lr()

        scheduler.step()
        after_step_lr_g, _ = scheduler.get_last_lr()

        assert after_step_lr_g < initial_lr_g

    def test_cosine_scheduler(self, optimizer_pair):
        """Test cosine annealing scheduler."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(
            opt_g, opt_d,
            scheduler_type="cosine",
            total_epochs=100
        )

        initial_lr_g, _ = scheduler.get_last_lr()

        # Step multiple times
        for _ in range(50):
            scheduler.step()

        mid_lr_g, _ = scheduler.get_last_lr()

        # LR should decrease with cosine annealing
        assert mid_lr_g <= initial_lr_g

    def test_invalid_scheduler_raises(self, optimizer_pair):
        """Test invalid scheduler type raises error."""
        opt_g, opt_d = optimizer_pair
        with pytest.raises(ValueError):
            LRScheduler(opt_g, opt_d, scheduler_type="invalid")

    def test_get_last_lr_returns_tuple(self, optimizer_pair):
        """Test get_last_lr returns tuple of two floats."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(opt_g, opt_d)
        lr_g, lr_d = scheduler.get_last_lr()

        assert isinstance(lr_g, float)
        assert isinstance(lr_d, float)
        assert lr_g > 0
        assert lr_d > 0

    def test_state_dict_operations(self, optimizer_pair):
        """Test state dict save and load."""
        opt_g, opt_d = optimizer_pair
        scheduler = LRScheduler(
            opt_g, opt_d,
            scheduler_type="step",
            decay_steps=2,
            decay_factor=0.5
        )

        # Step a few times
        for _ in range(3):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert "scheduler_type" in state

        # Create new scheduler and load state
        opt_g2, opt_d2 = optimizer_pair
        scheduler2 = LRScheduler(
            opt_g2, opt_d2,
            scheduler_type="step",
            decay_steps=2,
            decay_factor=0.5
        )

        # Should have different LRs initially
        lr1_g, _ = scheduler.get_last_lr()
        lr2_g, _ = scheduler2.get_last_lr()

        # After loading state, should match
        scheduler2.load_state_dict(state)
        lr2_g_after, _ = scheduler2.get_last_lr()

        assert abs(lr1_g - lr2_g_after) < 1e-6
