import torch
import torch.nn as nn


def _dcgan_init(module: nn.Module) -> None:
    """DCGAN-style weight initialisation: Conv/ConvTranspose ~ N(0, 0.02), BN ~ N(1, 0.02)."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias.data)


class Generator(nn.Module):

    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super().__init__()

        self.model = nn.Sequential(

            # latent → feature map
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(_dcgan_init)

    def forward(self, z):

        return self.model(z)
