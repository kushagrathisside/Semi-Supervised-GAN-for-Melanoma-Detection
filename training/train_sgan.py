import torch
import yaml

from torch.utils.data import DataLoader
from torch import optim

from datasets.melanoma_dataset import (
    LabeledMelanomaDataset,
    UnlabeledMelanomaDataset
)

from models.sgan import SGAN

from training.trainer import Trainer

from utils.seed import set_seed

import torch.backends.cudnn as cudnn


def train(config):

    set_seed(config["experiment"]["seed"])
    cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------
    # datasets
    # ---------------------------

    labeled_dataset = LabeledMelanomaDataset(
        config["dataset"]["labeled_path"],
        config["dataset"]["image_size"]
    )

    unlabeled_dataset = UnlabeledMelanomaDataset(
        config["dataset"]["unlabeled_path"],
        config["dataset"]["image_size"]
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # ---------------------------
    # model
    # ---------------------------

    model = SGAN(config).to(device)

    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=config["training"]["lr_generator"],
        betas=(
            config["training"]["beta1"],
            config["training"]["beta2"]
        )
    )

    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=config["training"]["lr_discriminator"],
        betas=(
            config["training"]["beta1"],
            config["training"]["beta2"]
        )
    )

    trainer = Trainer(
        model,
        labeled_loader,
        unlabeled_loader,
        optimizer_g,
        optimizer_d,
        device,
        config
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        trainer.train_epoch(epoch)
