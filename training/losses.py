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


def adversarial_generator_loss(
    logits_fake: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Non-saturating generator loss for the K+1 semi-supervised GAN.

    The generator wants the discriminator to assign fakes to ONE of the
    `num_classes` real classes rather than the fake class, i.e. to minimize
        -log P(real) = -log( sum_{c < K} softmax(logits)[c] ).

    Computed in log-space via logsumexp for numerical stability. Unlike the
    saturating form -log(P(fake)) negated, this provides strong gradients even
    when the discriminator is confident the input is fake (P(fake) -> 1), which
    is exactly the regime where the previous DINO-only generator stalled.

    Args:
        logits_fake: [B, K+1] discriminator logits on generated images
                     (grad must flow back to the generator — do NOT detach).
        num_classes: K, the number of real classes (fake class is the last index).
    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits_fake, dim=1)                     # [B, K+1]
    log_p_real = torch.logsumexp(log_probs[:, :num_classes], dim=1)   # log P(real)
    return -log_p_real.mean()


def mmd_loss(
    real_proj: torch.Tensor,
    fake_proj: torch.Tensor,
    scales: tuple = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """
    Multi-bandwidth RBF-kernel Maximum Mean Discrepancy between two sets of
    embeddings.

    Unlike mean-matching (which is MMD with a linear kernel and collapses each
    batch to a single vector), the RBF kernel matches the FULL distribution of
    individual embeddings. Every fake sample is therefore pressured toward the
    real manifold, supplying the per-sample realism signal that batch-mean
    matching lacked.

    Bandwidths are set per call from the median pairwise distance (median
    heuristic), so the loss adapts to the embedding scale automatically.

    Args:
        real_proj: [N, D] reference projections (detached internally — no grad).
        fake_proj: [B, D] fake projections (grad flows back to the generator).
        scales:    multipliers applied to the median bandwidth for a mixture
                   of kernels at several resolutions.
    Returns:
        Scalar MMD^2 estimate (>= 0 up to estimator noise).
    """
    real = real_proj.detach()

    d_xx = torch.cdist(real, real).pow(2)        # [N, N]
    d_yy = torch.cdist(fake_proj, fake_proj).pow(2)  # [B, B]
    d_xy = torch.cdist(real, fake_proj).pow(2)   # [N, B]

    # Median heuristic for the base bandwidth (from cross distances, detached).
    with torch.no_grad():
        base = torch.median(d_xy).clamp(min=1e-8)

    k_xx = torch.zeros_like(d_xx)
    k_yy = torch.zeros_like(d_yy)
    k_xy = torch.zeros_like(d_xy)
    for s in scales:
        gamma = 1.0 / (2.0 * base * s)
        k_xx = k_xx + torch.exp(-gamma * d_xx)
        k_yy = k_yy + torch.exp(-gamma * d_yy)
        k_xy = k_xy + torch.exp(-gamma * d_xy)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


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
