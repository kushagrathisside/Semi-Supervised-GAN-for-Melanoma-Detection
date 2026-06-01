"""
Configuration validation module using Pydantic.

Provides type-safe configuration parsing and validation with defaults.
"""

from typing import Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    name: str = Field(..., description="Experiment name")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    class Config:
        """Pydantic config."""
        extra = "forbid"


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    labeled_path: str = Field(default="data/labeled")
    unlabeled_path: str = Field(default="data/unlabeled/images")
    generated_path: str = Field(default="data/generated")
    image_size: int = Field(default=64, ge=32, le=512)
    num_channels: int = Field(default=3, ge=1, le=4)
    num_classes: int = Field(default=2, ge=2, le=10)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class GeneratorConfig(BaseModel):
    """Generator network configuration."""

    latent_dim: int = Field(default=256, ge=64, le=1024)
    feature_maps: int = Field(default=160, ge=32, le=1024)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class DiscriminatorConfig(BaseModel):
    """Discriminator network configuration."""

    feature_maps: int = Field(default=160, ge=32, le=1024)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=256, ge=1, le=2048)
    epochs: int = Field(default=600, ge=1, le=10000)
    lr_generator: float = Field(default=0.0002, gt=0, le=0.1)
    lr_discriminator: float = Field(default=0.0001, gt=0, le=0.1)
    beta1: float = Field(default=0.5, ge=0, le=1)
    beta2: float = Field(default=0.999, ge=0, le=1)
    num_workers: int = Field(default=8, ge=0, le=64)
    lr_scheduler_type: str = Field(default="none", description="Learning rate scheduler type")
    lr_decay_steps: int = Field(default=100, ge=1)
    lr_decay_factor: float = Field(default=0.95, gt=0, le=1)

    @validator("lr_scheduler_type")
    def validate_scheduler(cls, v: str) -> str:
        """Validate scheduler type."""
        valid_schedulers = ["none", "step", "exponential", "cosine"]
        if v not in valid_schedulers:
            raise ValueError(f"Scheduler must be one of {valid_schedulers}, got {v}")
        return v

    class Config:
        """Pydantic config."""
        extra = "forbid"


class GeneratorLossConfig(BaseModel):
    """
    Generator objective configuration (v3).

    The generator loss is a weighted sum of:
      - adversarial_weight * non-saturating adversarial loss (per-sample realism
        pressure via the discriminator's real/fake verdict), and
      - dino_mmd_weight * per-sample RBF-MMD between DINO projections of real and
        fake batches (distribution matching in foundation-model feature space).

    Setting dino_mmd_weight=0 gives a pure adversarial SGAN baseline (v3 Run 1).
    Setting adversarial_weight=0 reproduces a DINO-only generator (ablation).
    """

    adversarial_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    dino_mmd_weight: float = Field(default=0.0, ge=0.0, le=100.0)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class OutputConfig(BaseModel):
    """Output configuration."""

    checkpoint_dir: str = Field(default="outputs/checkpoints")
    sample_dir: str = Field(default="outputs/samples")
    log_dir: str = Field(default="outputs/logs")
    save_interval: int = Field(default=5, ge=1)
    sample_interval: int = Field(default=1, ge=1)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class DINOConfig(BaseModel):
    """DINOv2 feature matching configuration."""

    model: str = Field(default="dinov2_vitb14")
    input_size: int = Field(default=112, ge=56, le=224)
    projection_head_path: str = Field(default="outputs/projection_head.pt")
    lambda_cc: float = Field(default=0.1, ge=0.0, le=10.0)
    lambda_var: float = Field(default=0.0, ge=0.0, le=10.0)
    pretrain_epochs: int = Field(default=100, ge=1, le=10000)
    pretrain_lr: float = Field(default=0.001, gt=0.0, le=0.1)
    supcon_temperature: float = Field(default=0.15, gt=0.0, le=1.0)
    lambda_cc_warmup_start: int = Field(default=50, ge=0)
    lambda_cc_warmup_epochs: int = Field(default=50, ge=1)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class SGANConfig(BaseModel):
    """Complete SGAN configuration."""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    training: TrainingConfig
    output: OutputConfig
    generator_loss: GeneratorLossConfig = Field(default_factory=GeneratorLossConfig)
    dino: Optional[DINOConfig] = Field(default=None)

    class Config:
        """Pydantic config."""
        extra = "forbid"

    @classmethod
    def from_yaml(cls, config_path: str) -> "SGANConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated SGANConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

        if config_dict is None:
            raise ValueError(f"Config file is empty: {config_path}")

        try:
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")

    def save_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            output_path: Path to save YAML config
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
