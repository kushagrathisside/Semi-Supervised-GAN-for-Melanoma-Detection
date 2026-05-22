import os
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from training.losses import (
    supervised_loss,
    unlabeled_real_loss,
    fake_loss,
    feature_matching_loss
)

from utils.visualization import save_generated_images


class Trainer:

    def __init__(
        self,
        model,
        labeled_loader,
        unlabeled_loader,
        optimizer_g,
        optimizer_d,
        device,
        config
    ):

        self.model = model
        self.G = model.generator
        self.D = model.discriminator

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.device = device
        self.config = config

        self.latent_dim = config["generator"]["latent_dim"]

        self.sample_dir = config["output"]["sample_dir"]
        self.checkpoint_dir = config["output"]["checkpoint_dir"]
        self.log_dir = config["output"]["log_dir"]

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        self.scaler = GradScaler("cuda")

        # Track best generator
        self.best_g_loss = float("inf")

    # -------------------------------------------------

    def train(self):

        epochs = self.config["training"]["epochs"]

        for epoch in range(epochs):

            loss_d, loss_g = self.train_epoch(epoch)

            self.writer.add_scalar("Loss/Discriminator", loss_d, epoch)
            self.writer.add_scalar("Loss/Generator", loss_g, epoch)

            self.save_samples(epoch)

            # Save best generator
            self.save_best_checkpoint(epoch, loss_g)

    # -------------------------------------------------

    def train_epoch(self, epoch):

        self.G.train()
        self.D.train()

        labeled_iter = iter(self.labeled_loader)

        loop = tqdm(self.unlabeled_loader, desc=f"Epoch {epoch}")

        total_loss_d = 0
        total_loss_g = 0

        for step, unlabeled_imgs in enumerate(loop):

            try:
                labeled_imgs, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_imgs, labels = next(labeled_iter)

            labeled_imgs = labeled_imgs.to(self.device)
            labels = labels.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)

            batch_size = unlabeled_imgs.size(0)

            # -------------------------
            # Fake images
            # -------------------------

            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_imgs = self.G(z)

            # -------------------------
            # Train Discriminator
            # -------------------------

            self.optimizer_d.zero_grad()

            with autocast("cuda"):

                logits_labeled = self.D(labeled_imgs)
                loss_sup = supervised_loss(logits_labeled, labels)

                logits_unlabeled = self.D(unlabeled_imgs)
                loss_unlab = unlabeled_real_loss(logits_unlabeled)

                logits_fake = self.D(fake_imgs.detach())
                loss_fake_val = fake_loss(logits_fake)

                loss_d = loss_sup + loss_unlab + loss_fake_val

            self.scaler.scale(loss_d).backward()
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5)
            self.scaler.step(self.optimizer_d)
            self.scaler.update()

            # -------------------------
            # Train Generator
            # -------------------------

            self.optimizer_g.zero_grad()

            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_imgs = self.G(z)

            with autocast("cuda"):

                logits_fake, fake_features = self.D(fake_imgs, return_features=True)
                _, real_features = self.D(unlabeled_imgs, return_features=True)

                loss_g = feature_matching_loss(real_features, fake_features)

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()

            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()

            loop.set_postfix({
                "D": f"{loss_d.item():.3f}",
                "G": f"{loss_g.item():.3f}"
            })

        avg_loss_d = total_loss_d / len(self.unlabeled_loader)
        avg_loss_g = total_loss_g / len(self.unlabeled_loader)

        return avg_loss_d, avg_loss_g

    # -------------------------------------------------

    def save_samples(self, epoch):

        self.G.eval()

        with torch.no_grad():

            z = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
            fake = self.G(z)

            save_generated_images(
                fake,
                f"{self.sample_dir}/epoch_{epoch}.png"
            )

    # -------------------------------------------------

    def save_best_checkpoint(self, epoch, g_loss):

        if g_loss < self.best_g_loss:

            self.best_g_loss = g_loss

            path = f"{self.checkpoint_dir}/best_generator.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "generator": self.G.state_dict(),
                    "discriminator": self.D.state_dict(),
                    "optimizer_g": self.optimizer_g.state_dict(),
                    "optimizer_d": self.optimizer_d.state_dict()
                },
                path
            )

            print(f"\nNew best generator saved at epoch {epoch}")