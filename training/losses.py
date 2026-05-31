import torch
import torch.nn.functional as F

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
