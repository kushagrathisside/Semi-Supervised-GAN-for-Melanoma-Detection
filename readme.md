# Semi-Supervised GAN for Melanoma Detection

A PyTorch implementation of a Semi-Supervised Generative Adversarial Network (SGAN) for melanoma detection and synthetic image generation using dermoscopy images.

This project leverages a large pool of **~7000 unlabeled images** and a **small set of labeled images** (benign/malignant) to train a GAN that can both classify melanomas and generate realistic synthetic dermoscopy images for dataset augmentation.

## Overview

The SGAN architecture enables the discriminator to serve a **dual purpose**:
- **Classification**: Predict benign vs. malignant for labeled images
- **Adversarial Training**: Distinguish real vs. fake images for GAN training

This semi-supervised approach maximizes the value of limited labeled data by incorporating the full unlabeled dataset during training.

## Key Features

- Semi-supervised learning with labeled + unlabeled data
- 3-class discriminator (Benign, Malignant, Fake)
- Feature matching loss for stable generator training
- Mixed precision training (AMP) for faster convergence
- TensorBoard logging for real-time monitoring
- Configurable architecture via YAML
- Image augmentation pipeline
- FID score evaluation

## Architecture

### Generator
- **Input**: 256-dimensional latent vector
- **Output**: 64×64 RGB dermoscopy images
- **Feature Maps**: 160
- **Activation**: Tanh (outputs in [-1, 1])

### Discriminator
- **Input**: 64×64 RGB images
- **Output**: 3-class logits (Benign, Malignant, Fake)
- **Feature Maps**: 160
- **Purpose**: Classification + Real/Fake discrimination

### Loss Functions

| Loss | Purpose |
|------|---------|
| **Supervised Loss** | Train D to classify labeled images correctly |
| **Unlabeled Real Loss** | Train D to mark unlabeled images as "not fake" |
| **Fake Loss** | Train D to detect generated images |
| **Feature Matching Loss** | Train G to match real image features |

## Dataset Structure

```
data/
├── labeled/
│   ├── benign/          # Labeled benign samples
│   └── malignant/       # Labeled malignant samples
├── unlabeled/
│   └── images/          # Unlabeled dermoscopy images
└── generated/           # Synthetic images (created during generation)
```

## Installation

### Requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Dependencies
```
torch
torchvision
numpy
opencv-python
matplotlib
tqdm
pyyaml
scikit-learn
pillow
pytorch-fid
tensorboard
```

### Quick Setup

```bash
# Clone/navigate to project directory
cd melanoma-sgan

# Full setup (download dataset + install dependencies)
make setup

# Or install dependencies only
make install
```

The `setup.sh` script will:
1. Create required directories
2. Download the melanoma dataset
3. Organize labeled/unlabeled images
4. Install Python dependencies
5. Run dataset sanity checks

## Usage

### Training Mode

Train the SGAN on labeled and unlabeled data:

```bash
# Using Makefile
make train

# Or directly via CLI
python main.py --mode train --config configs/config.yaml
```

**Training details:**
- Loads labeled dataset from `data/labeled/`
- Loads unlabeled dataset from `data/unlabeled/images/`
- Trains for 600 epochs (configurable)
- Saves checkpoints to `outputs/checkpoints/`
- Generates sample images every epoch to `outputs/samples/`
- Logs metrics to TensorBoard: `outputs/logs/`

**Monitor training:**
```bash
make tensorboard
# Opens TensorBoard at http://localhost:6006
```

### Generation Mode

Generate synthetic dermoscopy images from a trained model:

```bash
# Using Makefile (generates 2000 images)
make generate

# Or directly via CLI
python main.py --mode generate --checkpoint outputs/checkpoints/best_generator.pt --num_images 2000
```

**Generation details:**
- Loads pre-trained generator checkpoint
- Generates specified number of images
- Saves to `data/generated/` as PNG files
- Each image named: `gen_0.png`, `gen_1.png`, etc.

### Evaluation

Compute Fréchet Inception Distance (FID) to evaluate generated image quality:

```python
from evaluation.fid_score import compute_fid

fid = compute_fid(
    real_dir="data/unlabeled/images",
    fake_dir="data/generated"
)
print(f"FID Score: {fid}")
```

## Configuration

Edit `configs/config.yaml` to customize training parameters:

```yaml
experiment:
  name: melanoma_sgan_high_capacity_128
  seed: 42                    # Reproducibility

dataset:
  labeled_path: data/labeled
  unlabeled_path: data/unlabeled/images
  generated_path: data/generated
  image_size: 64              # Image resolution
  num_channels: 3             # RGB
  num_classes: 2              # Benign/Malignant

generator:
  latent_dim: 256             # Latent vector dimension
  feature_maps: 160           # Network capacity

discriminator:
  feature_maps: 160           # Network capacity

training:
  batch_size: 256
  epochs: 600
  lr_generator: 0.0002
  lr_discriminator: 0.0001
  beta1: 0.5                  # Adam optimizer parameter
  beta2: 0.999                # Adam optimizer parameter
  num_workers: 8              # Data loading threads

output:
  checkpoint_dir: outputs/checkpoints
  sample_dir: outputs/samples
  log_dir: outputs/logs
```

### Key Hyperparameters

- **batch_size**: Batch size for training (256)
- **epochs**: Number of training epochs (600)
- **latent_dim**: Dimension of noise vector fed to generator (256)
- **feature_maps**: Network capacity for both G and D (160)
- **Learning rates**: G=0.0002, D=0.0001 (from GAN best practices)
- **image_size**: Input image resolution (64×64)

## Training Pipeline

The training process alternates between:

### Discriminator Step
1. **Supervised Loss**: Pass labeled images, compute classification loss
2. **Unlabeled Real Loss**: Pass unlabeled images, encourage "not fake" prediction
3. **Fake Loss**: Pass generated images, encourage "fake" prediction
4. All three losses are combined: `loss_d = supervised + unlabeled + fake`

### Generator Step
1. Generate batch of fake images
2. Extract features from discriminator
3. Compute feature matching loss against real unlabeled images
4. Backpropagate to update generator

**Stability features:**
- Mixed precision training (Automatic Mixed Precision - AMP)
- Gradient clipping (max norm: 5)
- Adaptive learning rates (Adam optimizer)
- Feature matching loss (prevents mode collapse)

## Project Structure

```
melanoma-sgan/
├── main.py                          # Entry point (train/generate modes)
├── Makefile                         # Common commands
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Dataset setup script
├── sanitytest.py                    # Dataset validation
│
├── configs/
│   ├── config.yaml                  # Training configuration
│   └── readme.md
│
├── data/
│   ├── labeled/                     # Labeled images (benign/malignant)
│   ├── unlabeled/images/            # Unlabeled images
│   ├── generated/                   # Generated synthetic images
│   └── readme.md
│
├── datasets/
│   ├── melanoma_dataset.py          # Dataset loaders
│   └── readme.md
│
├── models/
│   ├── sgan.py                      # SGAN model wrapper
│   ├── generator.py                 # Generator network
│   ├── discriminator.py             # Discriminator network
│   └── readme.md
│
├── training/
│   ├── train_sgan.py                # Training entry point
│   ├── trainer.py                   # Trainer class (epoch loop)
│   ├── losses.py                    # Loss functions
│   └── readme.md
│
├── augmentation/
│   ├── generate_images.py           # Image generation pipeline
│   └── readme.md
│
├── evaluation/
│   ├── fid_score.py                 # FID metric computation
│   ├── prepare_real_fid.py          # Prepare real images for FID
│   ├── fake_samples/                # Generated images for FID
│   ├── real_samples/                # Real images for FID
│   └── readme.md
│
├── utils/
│   ├── seed.py                      # Reproducibility utilities
│   ├── transforms.py                # Image preprocessing
│   ├── visualization.py             # Image visualization
│   └── readme.md
│
├── outputs/
│   ├── checkpoints/                 # Model checkpoints
│   ├── samples/                     # Generated samples during training
│   └── logs/                        # TensorBoard event files
│
└── readme.md                         # This file
```

## Common Commands

```bash
# Setup and installation
make setup                  # Full setup (dataset + dependencies)
make install                # Install Python dependencies only

# Training and generation
make train                  # Train SGAN model
make generate               # Generate synthetic images

# Monitoring and cleanup
make tensorboard            # Launch TensorBoard dashboard
make clean                  # Remove generated outputs

# Get help
make help                   # Show available commands
```

## Workflow

### Complete Training & Generation Pipeline

```bash
# 1. Setup (one-time)
make setup

# 2. Train SGAN
make train
# Monitor training in another terminal:
make tensorboard

# 3. Generate synthetic images
make generate

# 4. Evaluate quality
python -c "from evaluation.fid_score import compute_fid; print(compute_fid('data/unlabeled/images', 'data/generated'))"

# 5. Use augmented dataset for downstream classification
# (Generated images can now be used to augment labeled dataset)
```

## Utilities

### Image Preprocessing
- Resize to 64×64
- Random horizontal flip (50% probability)
- Normalize to [-1, 1] (required for GAN training with Tanh activation)

### Reproducibility
- Seed control for Python, NumPy, PyTorch, CUDA
- Configure in `config.yaml`: `experiment.seed`

### Visualization
- Save generated image grids during training
- Display and save utilities in `utils/visualization.py`

## Performance Metrics

### Fréchet Inception Distance (FID)
- Measures similarity between generated and real distributions
- **Lower is better**
- Computed using: `evaluation/fid_score.py`

### Training Metrics (TensorBoard)
- Discriminator loss
- Generator loss
- Loss breakdowns (supervised, unlabeled, fake, feature matching)

## Requirements

See `requirements.txt` for complete dependencies. Key packages:

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torchvision | Image utilities and pretrained models |
| pytorch-fid | FID score computation |
| tensorboard | Training visualization |
| pyyaml | Config file parsing |
| pillow | Image I/O |
| opencv-python | Advanced image processing |

## License

[Add license information here]

## References

- Goodfellow et al., 2014: "Generative Adversarial Networks"
- Salimans et al., 2016: "Improved Techniques for Training GANs"
- Springenberg, 2015: "Unsupervised and Semi-Supervised Learning with Categorical Generative Adversarial Networks"

