import torchvision.transforms as transforms


def get_train_transforms(image_size: int = 64):
    """
    Augmented transforms for training images.

    Dermoscopy images have no canonical orientation, so vertical flips and
    90-degree rotations are valid. ColorJitter captures skin-tone variation.
    All augmentations are applied stochastically per sample.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(image_size: int = 64):
    """
    Deterministic transforms for evaluation / FID preparation.

    No random augmentation — every call returns the same tensor for the same
    input, so metrics are reproducible across runs.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_transforms(image_size: int = 64):
    """Backwards-compatible alias — returns training transforms."""
    return get_train_transforms(image_size)
