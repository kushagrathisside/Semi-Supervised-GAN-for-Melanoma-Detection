"""
Experiment tracking and monitoring utilities.

Supports local TensorBoard logging and optional Weights & Biases integration.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import json


class ExperimentTracker:
    """
    Track experiment metrics and metadata.

    Logs to both local JSON files and TensorBoard summaries.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "outputs/experiments",
        use_wandb: bool = False
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
            use_wandb: Whether to use Weights & Biases (optional)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.config_file = self.experiment_dir / "config.json"

        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(project="melanoma-sgan", name=experiment_name)
            except ImportError:
                print("Weights & Biases not installed. Disabling wandb logging.")
                self.use_wandb = False

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        if self.use_wandb:
            self.wandb.config.update(config)

    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log training metrics.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names and values
            step: Optional step number (for detailed logging)
        """
        log_entry = {"epoch": epoch, **metrics}
        if step is not None:
            log_entry["step"] = step

        # Write to JSONL file
        with open(self.metrics_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

        # Log to Weights & Biases
        if self.use_wandb:
            self.wandb.log(log_entry, step=step or epoch)

    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """
        Log an artifact (model checkpoint, sample image, etc.).

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact ('model', 'image', etc.)
        """
        if self.use_wandb:
            artifact = self.wandb.Artifact(
                f"{self.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.metrics_file.exists():
            return {}

        metrics_list = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                metrics_list.append(json.loads(line))

        if not metrics_list:
            return {}

        # Aggregate metrics
        summary = {}
        for key in metrics_list[-1].keys():
            if key in ["epoch", "step"]:
                continue
            values = [m[key] for m in metrics_list if key in m]
            if values:
                summary[f"{key}_latest"] = values[-1]
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)

        return summary

    def close(self) -> None:
        """Clean up and close experiment tracking."""
        if self.use_wandb:
            self.wandb.finish()
