# Synthetic Image Generation

This module generates synthetic dermoscopy images using the trained
generator network.

After the SGAN model is trained, the generator can produce new images
that follow the learned distribution of melanoma datasets.

These synthetic images can be used for:

- dataset augmentation
- class balancing
- visualization of GAN learning

## Output

Generated images are saved in:

data/generated/

These samples can later be merged with the labeled dataset to improve
classifier training.

## Usage

Run the generation script after training a model checkpoint.

Example:

python augmentation/generate_images.py
