import torch.nn as nn
import torch.nn.functional as F


class DINOProjectionHead(nn.Module):
    """
    2-layer MLP that maps DINO mean patch features (768-dim) to a
    L2-normalized 256-dim embedding, pre-trained with Supervised Contrastive
    loss to adapt DINO's manifold to the benign/malignant margin in dermoscopy.

    Compatible with both DINOv2 (patch_size=14) and DINOv3 (patch_size=16)
    since both ViT-B variants produce 768-dim patch embeddings.

    Frozen during GAN training — only the SupCon pre-training step updates
    these weights.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
