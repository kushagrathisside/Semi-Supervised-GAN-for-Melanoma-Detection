# Architecture

Technical deep-dive into the SGAN architecture and design decisions.

---

## Overview

The model follows the Semi-Supervised GAN formulation from Salimans et al. (2016). A single discriminator serves two purposes simultaneously:

1. **Classifier** — distinguish benign from malignant lesions on labeled data
2. **Adversary** — detect fake images produced by the generator

This is achieved by extending the discriminator's output from `K` real classes to `K + 1`, where the extra class represents "fake."

---

## Generator

**File:** [`models/generator.py`](../models/generator.py)

A standard DCGAN-style transposed convolution network.

```
Latent vector z ~ N(0,1)
Shape: [B, 256, 1, 1]

ConvTranspose2d(256 → 1280, k=4, s=1, p=0)  →  [B, 1280, 4, 4]
BatchNorm2d + ReLU

ConvTranspose2d(1280 → 640, k=4, s=2, p=1)  →  [B, 640, 8, 8]
BatchNorm2d + ReLU

ConvTranspose2d(640 → 320, k=4, s=2, p=1)   →  [B, 320, 16, 16]
BatchNorm2d + ReLU

ConvTranspose2d(320 → 160, k=4, s=2, p=1)   →  [B, 160, 32, 32]
BatchNorm2d + ReLU

ConvTranspose2d(160 → 3, k=4, s=2, p=1)     →  [B, 3, 64, 64]
Tanh  →  output in [-1, 1]
```

Channel counts above use `feature_maps=160` (configured in `config.yaml`). The formula per layer is `feature_maps × [8, 4, 2, 1, -]`.

**Why Tanh output?** The discriminator's input images are normalized to `[-1, 1]` via `Normalize(mean=0.5, std=0.5)`. Matching the generator's output range ensures real and fake images are on the same scale.

---

## Discriminator

**File:** [`models/discriminator.py`](../models/discriminator.py)

A convolutional feature extractor followed by a multi-class classifier.

```
Input image
Shape: [B, 3, 64, 64]

SpectralNorm(Conv2d(3 → 160, k=4, s=2, p=1))    →  [B, 160, 32, 32]
LeakyReLU(0.2)

SpectralNorm(Conv2d(160 → 320, k=4, s=2, p=1))  →  [B, 320, 16, 16]
LeakyReLU(0.2)

SpectralNorm(Conv2d(320 → 640, k=4, s=2, p=1))  →  [B, 640, 8, 8]
LeakyReLU(0.2)

SpectralNorm(Conv2d(640 → 1280, k=4, s=2, p=1)) →  [B, 1280, 4, 4]
LeakyReLU(0.2)
                                    ↑
                          features (returned when return_features=True)

Conv2d(1280 → 3, k=4, s=1, p=0)                 →  [B, 3, 1, 1]
view(B, -1)                                       →  [B, 3]
```

Output logits layout:
| Index | Meaning |
|-------|---------|
| 0 | Benign |
| 1 | Malignant |
| 2 | Fake (generated) |

**Why Spectral Normalization?** Constrains the Lipschitz constant of each layer, which stabilizes discriminator training and prevents gradient explosion — a common failure mode in GANs with limited data.

**Why no BatchNorm on the discriminator?** BatchNorm statistics depend on the batch composition. When a batch contains both real and fake images at different stages of training, it can interfere with the adversarial signal. Spectral norm is the preferred stabilizer here.

---

## SGAN Wrapper

**File:** [`models/sgan.py`](../models/sgan.py)

A thin wrapper that wires `Generator` and `Discriminator` together and reads their parameters from `SGANConfig`. Use `model.generate(z)` and `model.discriminate(x)` to invoke each component independently.

---

## Loss Functions

**File:** [`training/losses.py`](../training/losses.py)

### `supervised_loss(logits, labels)`

Label-smoothed cross-entropy on the labeled batch. Smoothing factor `ε = 0.1` redistributes confidence to prevent over-fitting on the small labeled set.

```
smooth_target = one_hot(labels) × (1 - ε) + ε / K
loss = -mean( smooth_target · log_softmax(logits) )
```

Note: the full `K+1` logits are passed through softmax, so the "fake" class competes with real classes. The model must learn that real labeled images should not score high on the fake index.

### `unlabeled_real_loss(logits)`

Pushes the discriminator to assign low fake-class probability to unlabeled real images.

```
fake_prob = softmax(logits)[:, -1]
loss = -mean( log(1 - fake_prob) )
```

### `fake_loss(logits)`

Pushes the discriminator to assign high fake-class probability to generated images.

```
fake_prob = softmax(logits)[:, -1]
loss = -mean( log(fake_prob) )
```

### `feature_matching_loss(real_features, fake_features)`

Trains the generator to produce images whose discriminator-internal activations match those of real images on average. This replaces the standard adversarial generator loss and is significantly more stable.

```
loss = mean( |mean_batch(real_features) - mean_batch(fake_features)| )
```

The features used are the output of the last convolutional block (before the classifier head), shape `[B, 1280, 4, 4]`.

### Combined discriminator loss

```
loss_D = supervised_loss + unlabeled_real_loss + fake_loss
```

All three terms are weighted equally. The generator loss is only `feature_matching_loss`.

---

## Design Trade-offs

| Choice | Alternative | Reason |
|--------|-------------|--------|
| Feature matching for G | Adversarial G loss | Less prone to mode collapse; gradients don't vanish when D is strong |
| Spectral norm on D | BatchNorm on D | More stable across varied batch compositions |
| K+1 output classes | Separate real/fake head | Unified softmax keeps all objectives in one probability simplex |
| Latent dim 256 | 100 (DCGAN default) | Larger latent space gives more expressive generation capacity |
| LR_G = 2× LR_D | Equal LR | Compensates for the discriminator having three objectives vs. generator's one |
