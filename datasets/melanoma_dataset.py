import os
from PIL import Image
from torch.utils.data import Dataset
from utils.transforms import get_transforms


class LabeledMelanomaDataset(Dataset):
    """
    Dataset for labeled melanoma images.
    """

    def __init__(self, root_dir, image_size=64):

        self.samples = []

        benign_dir = os.path.join(root_dir, "benign")
        malignant_dir = os.path.join(root_dir, "malignant")

        for img in os.listdir(benign_dir):
            path = os.path.join(benign_dir, img)
            self.samples.append((path, 0))

        for img in os.listdir(malignant_dir):
            path = os.path.join(malignant_dir, img)
            self.samples.append((path, 1))

        self.transform = get_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


class UnlabeledMelanomaDataset(Dataset):
    """
    Dataset for unlabeled melanoma images.
    """

    def __init__(self, root_dir, image_size=64):

        self.images = [
            os.path.join(root_dir, img)
            for img in os.listdir(root_dir)
        ]

        self.transform = get_transforms(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image
