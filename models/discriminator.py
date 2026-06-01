import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from models.generator import _dcgan_init


class Discriminator(nn.Module):

    def __init__(self, feature_maps=64, channels=3, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(

            spectral_norm(nn.Conv2d(channels, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1,)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Spectral norm applied to the classifier too — keeps singular values bounded
        # across the entire network, not just the feature extractor.
        self.classifier = spectral_norm(nn.Conv2d(
            feature_maps * 8,
            num_classes + 1,
            4,
            1,
            0,
            bias=False,
        ))

        self.apply(_dcgan_init)

    def forward(self, x, return_features=False):

        features = self.features(x)

        logits = self.classifier(features)

        logits = logits.view(x.size(0), -1)

        if return_features:
            return logits, features

        return logits
