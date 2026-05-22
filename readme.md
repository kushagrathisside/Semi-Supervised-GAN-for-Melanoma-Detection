# Semi-Supervised GAN for Melanoma Detection

This project implements a Semi-Supervised Generative Adversarial Network (SGAN)
for melanoma detection using dermoscopy images.

The system leverages a large pool of unlabeled images (~7000) and a small set
of labeled images to improve classification performance.

The SGAN architecture allows the discriminator to function both as:

1. A classifier for labeled images
2. A real/fake discriminator for GAN training

## Project Goals

- Generate synthetic dermoscopy images
- Augment limited labeled melanoma datasets
- Improve classification accuracy

## Architecture

Generator → produces synthetic dermoscopy images  
Discriminator → predicts

- Benign
- Malignant
- Fake

## Pipeline

1. Load labeled + unlabeled images
2. Train SGAN
3. Generate synthetic images
4. Augment dataset
5. Train melanoma classifier

## Dataset Structure

data/
    labeled/
        benign/
        malignant/

    unlabeled/
        images/

## Training

Run:

