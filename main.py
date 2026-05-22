import os
import argparse
import yaml

from training.train_sgan import train

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """
    Parse command line arguments for experiment control.
    """

    parser = argparse.ArgumentParser(
        description="Semi-Supervised GAN for Melanoma Detection"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate"],
        help="Experiment mode"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for generation"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate"
    )

    return parser.parse_args()


def load_config(config_path):
    """
    Load YAML configuration file.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def prepare_output_dirs(config):
    """
    Ensure experiment output directories exist.
    """

    checkpoint_dir = config["output"]["checkpoint_dir"]
    sample_dir = config["output"]["sample_dir"]
    log_dir = config["output"]["log_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def print_experiment_info(config, args):

    print("\n==============================")
    print("SGAN Melanoma Experiment")
    print("==============================")

    print(f"Mode: {args.mode}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {config['experiment']['seed']}")

    print("\nDataset")
    print("-------")
    print("Labeled:", config["dataset"]["labeled_path"])
    print("Unlabeled:", config["dataset"]["unlabeled_path"])

    print("\nTraining")
    print("--------")
    print("Batch size:", config["training"]["batch_size"])
    print("Epochs:", config["training"]["epochs"])

    print("==============================\n")


def main():

    args = parse_args()

    config = load_config(args.config)

    prepare_output_dirs(config)

    print_experiment_info(config, args)

    if args.mode == "train":

        train(config)

    elif args.mode == "generate":

        if args.checkpoint is None:

            raise ValueError(
                "Checkpoint required for generation mode."
            )

        from augmentation.generate_images import generate_samples

        generate_samples(
            checkpoint_path=args.checkpoint,
            num_images=args.num_images
        )


if __name__ == "__main__":

    main()