"""
Learning rate scheduling utilities.

Provides multiple learning rate scheduling strategies for stable training.
"""

from typing import Tuple, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, LambdaLR
)


class LRScheduler:
    """
    Wrapper for learning rate scheduling.

    Supports multiple scheduling strategies: step decay, exponential decay, cosine annealing.
    """

    def __init__(
        self,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        scheduler_type: str = "none",
        decay_steps: int = 100,
        decay_factor: float = 0.95,
        total_epochs: int = 600
    ) -> None:
        """
        Initialize learning rate schedulers.

        Args:
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            scheduler_type: Type of scheduler ('none', 'step', 'exponential', 'cosine')
            decay_steps: Steps between decay applications (for step scheduler)
            decay_factor: Multiplicative decay factor (for step/exponential)
            total_epochs: Total training epochs (for cosine scheduler)

        Raises:
            ValueError: If scheduler_type is not supported
        """
        self.scheduler_type = scheduler_type
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        if scheduler_type == "none":
            self.scheduler_g = None
            self.scheduler_d = None
        elif scheduler_type == "step":
            self.scheduler_g = StepLR(optimizer_g, step_size=decay_steps, gamma=decay_factor)
            self.scheduler_d = StepLR(optimizer_d, step_size=decay_steps, gamma=decay_factor)
        elif scheduler_type == "exponential":
            self.scheduler_g = ExponentialLR(optimizer_g, gamma=decay_factor)
            self.scheduler_d = ExponentialLR(optimizer_d, gamma=decay_factor)
        elif scheduler_type == "cosine":
            self.scheduler_g = CosineAnnealingLR(optimizer_g, T_max=total_epochs)
            self.scheduler_d = CosineAnnealingLR(optimizer_d, T_max=total_epochs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self) -> None:
        """Step the learning rate schedulers."""
        if self.scheduler_type != "none":
            if self.scheduler_g is not None:
                self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

    def get_last_lr(self) -> Tuple[float, float]:
        """
        Get current learning rates.

        Returns:
            Tuple of (generator_lr, discriminator_lr)
        """
        lr_g = self.optimizer_g.param_groups[0]["lr"]
        lr_d = self.optimizer_d.param_groups[0]["lr"]
        return lr_g, lr_d

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        state = {"scheduler_type": self.scheduler_type}
        if self.scheduler_g is not None:
            state["scheduler_g"] = self.scheduler_g.state_dict()
        if self.scheduler_d is not None:
            state["scheduler_d"] = self.scheduler_d.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        if "scheduler_g" in state_dict and self.scheduler_g is not None:
            self.scheduler_g.load_state_dict(state_dict["scheduler_g"])
        if "scheduler_d" in state_dict and self.scheduler_d is not None:
            self.scheduler_d.load_state_dict(state_dict["scheduler_d"])
