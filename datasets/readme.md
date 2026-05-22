# Dataset Loaders

This module contains PyTorch dataset classes for loading melanoma
dermoscopy images.

The project uses a **semi-supervised learning setup**, therefore
two different datasets are required:

1. LabeledMelanomaDataset  
   Contains images with class labels.

2. UnlabeledMelanomaDataset  
   Contains images without labels used for semi-supervised GAN training.

## Dataset Structure

The expected directory layout is:

data/

    labeled/
        benign/
        malignant/

    unlabeled/
        images/

## Data Flow in Training

During training the model receives:

- labeled images → used for supervised classification loss
- unlabeled images → used for unsupervised discriminator training
- generated images → produced by the generator

This enables learning from large unlabeled datasets.
