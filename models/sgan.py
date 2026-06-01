import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator


class SGAN(nn.Module):

    def __init__(self, config):

        super().__init__()

        latent_dim = config.generator.latent_dim
        g_maps = config.generator.feature_maps

        d_maps = config.discriminator.feature_maps

        num_classes = config.dataset.num_classes
        channels = config.dataset.num_channels

        self.generator = Generator(
            latent_dim,
            g_maps,
            channels
        )

        self.discriminator = Discriminator(
            d_maps,
            channels,
            num_classes
        )

    def generate(self, z):

        return self.generator(z)

    def discriminate(self, x):

        return self.discriminator(x)

