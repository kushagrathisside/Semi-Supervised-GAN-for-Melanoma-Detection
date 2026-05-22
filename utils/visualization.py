import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def save_generated_images(images, path, nrow=8):
    """
    Save generated image grid to disk.
    """

    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=True
    )

    vutils.save_image(grid, path)


def show_images(images, nrow=8):
    """
    Display image grid using matplotlib.
    """

    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=True
    )

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.show()
