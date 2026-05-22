import torchvision.transforms as transforms


def get_transforms(image_size: int = 64):
    """
    Returns image preprocessing transforms used for melanoma images.
    """

    transform = transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    ])

    return transform

#Why this matters: GANs train better when images are normalized to: [-1 , 1] because the generator output uses Tanh activation.
