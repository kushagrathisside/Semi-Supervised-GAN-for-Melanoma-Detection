# Training Pipeline

This module implements the training logic for the Semi-Supervised GAN.

Training a Semi-Supervised GAN differs from standard GAN training
because the discriminator performs two tasks:

1. classify real images into medical classes
2. detect generated images as fake

The discriminator therefore outputs:

0 → benign  
1 → malignant  
2 → fake

## Training Components

losses.py
    Implements loss functions used in SGAN training.

trainer.py
    Defines the training loop and optimization logic.

train_sgan.py
    Entry point that initializes datasets, models and training.
