import torch
import torch.nn.functional as F
from typing import Dict


def feature_matching_loss(real_features, fake_features):

    loss = torch.mean(
            torch.abs(
                real_features.mean(0) - fake_features.mean(0)
            )
    )

    return loss


def supervised_loss(logits, labels):

    smooth = 0.1
    num_classes = logits.size(1)

    one_hot = F.one_hot(labels, num_classes).float()

    one_hot = one_hot * (1 - smooth) + smooth / num_classes

    log_prob = F.log_softmax(logits, dim=1)

    loss = -(one_hot * log_prob).sum(dim=1).mean()

    return loss


def unlabeled_real_loss(logits):
    """
    Encourage discriminator to classify
    real unlabeled images as NOT fake.
    """

    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[:, -1]

    loss = -torch.mean(torch.log(1 - fake_prob + 1e-8))

    return loss


def fake_loss(logits):
    """
    Encourage discriminator to detect fake images.
    """

    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[:, -1]

    loss = -torch.mean(torch.log(fake_prob + 1e-8))

    return loss


def generator_loss(logits):
    """
    Generator tries to fool discriminator.
    """

    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[:, -1]

    loss = -torch.mean(torch.log(1 - fake_prob + 1e-8))

    return loss


def dino_feature_matching_loss(
    proj_real_mean: torch.Tensor,
    proj_fake: torch.Tensor,
    lambda_var: float = 0.0,
) -> torch.Tensor:
    """
    L1 distance between mean DINO projections of real and fake batches.

    Both inputs must be L2-normalized vectors (output of DINOProjectionHead),
    so the loss is bounded in [0, 2].

    Args:
        proj_real_mean: [D] pre-computed batch mean of real projections (no grad).
        proj_fake:      [B, D] projected fake images (grad flows through to G).
        lambda_var:     weight for optional variance-matching term. 0 disables it.
    """
    mean_diff = torch.mean(torch.abs(proj_real_mean - proj_fake.mean(0)))

    if lambda_var > 0.0:
        var_real = proj_real_mean.var()
        var_fake = proj_fake.var(dim=0).mean()
        return mean_diff + lambda_var * torch.abs(var_real - var_fake)

    return mean_diff


def confidence_weighted_dino_loss(
    proj_fake: torch.Tensor,
    class_anchors: Dict[int, torch.Tensor],
    fake_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Class-conditional DINO matching weighted by discriminator confidence.

    For each class c, weights the fake projections by the discriminator's
    softmax probability that each fake image belongs to class c, then
    matches the weighted centroid to the pre-computed real class anchor.

    When the discriminator is uncertain (all probs ~0.5), both class terms
    pull toward their respective anchors equally, so the loss degrades
    gracefully toward the unconditional case.

    Args:
        proj_fake:     [B, D] projected fake images (grad flows through to G).
        class_anchors: {class_id: [D]} pre-computed fixed real class means (no grad).
        fake_probs:    [B, K] discriminator softmax probabilities, detached — used
                       as importance weights only, not differentiated.
    Returns:
        Scalar loss.
    """
    loss = proj_fake.new_zeros(1).squeeze()

    for c, anchor in class_anchors.items():
        w = fake_probs[:, c]                                          # [B]
        denom = w.sum() + 1e-8
        weighted_fake = (proj_fake * w.unsqueeze(1)).sum(0) / denom  # [D]
        loss = loss + torch.mean(torch.abs(anchor.detach() - weighted_fake))

    return loss
