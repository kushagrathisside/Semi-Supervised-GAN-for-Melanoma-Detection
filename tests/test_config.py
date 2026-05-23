"""
Unit tests for configuration validation.
"""

import pytest
import tempfile
from pathlib import Path
from utils.config import (
    SGANConfig, ExperimentConfig, DatasetConfig,
    GeneratorConfig, DiscriminatorConfig, TrainingConfig, OutputConfig
)


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_valid_config(self):
        """Test valid experiment config."""
        config = ExperimentConfig(name="test", seed=42)
        assert config.name == "test"
        assert config.seed == 42

    def test_missing_name_raises(self):
        """Test missing required name field."""
        with pytest.raises(ValueError):
            ExperimentConfig(seed=42)


class TestDatasetConfig:
    """Test dataset configuration."""

    def test_valid_config(self):
        """Test valid dataset config."""
        config = DatasetConfig(
            labeled_path="data/labeled",
            image_size=64
        )
        assert config.image_size == 64

    def test_image_size_bounds(self):
        """Test image size bounds validation."""
        with pytest.raises(ValueError):
            DatasetConfig(image_size=16)  # Too small

        with pytest.raises(ValueError):
            DatasetConfig(image_size=1024)  # Too large


class TestTrainingConfig:
    """Test training configuration."""

    def test_valid_config(self):
        """Test valid training config."""
        config = TrainingConfig(batch_size=256, epochs=600)
        assert config.batch_size == 256

    def test_lr_scheduler_validation(self):
        """Test learning rate scheduler type validation."""
        with pytest.raises(ValueError):
            TrainingConfig(lr_scheduler_type="invalid_scheduler")

    def test_valid_schedulers(self):
        """Test all valid scheduler types."""
        valid_schedulers = ["none", "step", "exponential", "cosine"]
        for scheduler in valid_schedulers:
            config = TrainingConfig(lr_scheduler_type=scheduler)
            assert config.lr_scheduler_type == scheduler


class TestSGANConfig:
    """Test complete SGAN configuration."""

    def test_from_yaml_valid_config(self):
        """Test loading valid YAML configuration."""
        config_yaml = """
experiment:
  name: test_experiment
  seed: 42

dataset:
  labeled_path: data/labeled
  unlabeled_path: data/unlabeled/images
  generated_path: data/generated
  image_size: 64
  num_channels: 3
  num_classes: 2

generator:
  latent_dim: 256
  feature_maps: 160

discriminator:
  feature_maps: 160

training:
  batch_size: 256
  epochs: 100
  lr_generator: 0.0002
  lr_discriminator: 0.0001
  beta1: 0.5
  beta2: 0.999
  num_workers: 8

output:
  checkpoint_dir: outputs/checkpoints
  sample_dir: outputs/samples
  log_dir: outputs/logs
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            config = SGANConfig.from_yaml(config_path)
            assert config.experiment.name == "test_experiment"
            assert config.training.batch_size == 256
        finally:
            Path(config_path).unlink()

    def test_from_yaml_missing_file(self):
        """Test loading non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            SGANConfig.from_yaml("nonexistent/config.yaml")

    def test_from_yaml_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(ValueError):
                SGANConfig.from_yaml(config_path)
        finally:
            Path(config_path).unlink()

    def test_from_yaml_missing_required_fields(self):
        """Test loading YAML with missing required fields."""
        config_yaml = """
experiment:
  seed: 42
dataset:
  labeled_path: data/labeled
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            with pytest.raises(ValueError):
                SGANConfig.from_yaml(config_path)
        finally:
            Path(config_path).unlink()

    def test_save_yaml(self):
        """Test saving config to YAML file."""
        config_yaml = """
experiment:
  name: test_experiment
  seed: 42

dataset:
  labeled_path: data/labeled
  unlabeled_path: data/unlabeled/images
  generated_path: data/generated
  image_size: 64
  num_channels: 3
  num_classes: 2

generator:
  latent_dim: 256
  feature_maps: 160

discriminator:
  feature_maps: 160

training:
  batch_size: 256
  epochs: 100
  lr_generator: 0.0002
  lr_discriminator: 0.0001
  beta1: 0.5
  beta2: 0.999
  num_workers: 8

output:
  checkpoint_dir: outputs/checkpoints
  sample_dir: outputs/samples
  log_dir: outputs/logs
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            input_path = f.name

        output_path = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False).name

        try:
            config = SGANConfig.from_yaml(input_path)
            config.save_yaml(output_path)
            assert Path(output_path).exists()

            # Reload and verify
            reloaded = SGANConfig.from_yaml(output_path)
            assert reloaded.experiment.name == config.experiment.name
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()
