# Model Architectures

This module contains neural network architectures used in the project.

The project implements a **Semi-Supervised Generative Adversarial Network (SGAN)**.

Unlike standard GANs, the discriminator in SGAN performs two tasks:

1. Classifies real images into medical classes
2. Detects generated images as fake

## Components

### Generator

The generator learns to produce realistic dermoscopy images from random
latent noise.

Input:
    random latent vector

Output:
    synthetic dermoscopy image

### Discriminator

The discriminator outputs **three classes**:

0 → benign  
1 → malignant  
2 → fake

This enables semi-supervised learning using:

- labeled images
- unlabeled images
- generated images

### SGAN Wrapper

The SGAN module combines generator and discriminator for unified training.
