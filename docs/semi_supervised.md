# Semi-Supervised Learning

How and why the SGAN uses unlabeled data, and what each loss term contributes.

---

## The Problem

Labeled dermoscopy datasets are expensive to produce — each image requires a clinical diagnosis. This project has only **200 labeled images** (100 benign, 100 malignant), which is far too few to train a reliable classifier from scratch.

However, **7,018 unlabeled dermoscopy images** are available. These carry no class label, but they do carry information about what dermoscopy images look like in general. Semi-supervised learning exploits this.

---

## The Key Insight: K+1 Classes

A standard supervised classifier outputs `K` class probabilities. The SGAN discriminator outputs `K + 1`:

```
[P(benign), P(malignant), P(fake)]
```

The `K` real classes are trained with labeled data in the usual supervised way. The `fake` class is trained adversarially using unlabeled real images and GAN-generated images.

This unified formulation means:
- The discriminator learns what real dermoscopy images look like (from unlabeled data) — it must push P(fake) low for all real images.
- This learned prior about "realness" regularizes the supervised classification signal and makes it generalize better from few labeled examples.

---

## What Each Loss Term Does

### `supervised_loss` — labeled data only

```
inputs:  labeled images (benign or malignant)
targets: {0, 1} (with label smoothing)
purpose: teach the discriminator to classify
```

This is the only loss that uses class labels. With only 200 images, it would overfit quickly if used alone. The other two D losses act as regularizers.

### `unlabeled_real_loss` — unlabeled data only

```
inputs:  unlabeled real images
targets: "not fake" (i.e. P(fake) → 0)
purpose: expose the discriminator to the full data distribution
```

This forces the discriminator to learn a boundary between the entire real image manifold and fake images. Without this, the discriminator would only know what 200 real images look like, and its "realness" detector would be fragile.

### `fake_loss` — GAN-generated images only

```
inputs:  images from G(z)
targets: "fake" (P(fake) → 1)
purpose: adversarial training — make D detect G's outputs
```

This creates the adversarial pressure that forces the generator to improve.

### `feature_matching_loss` — generator only

```
inputs:  discriminator features of real (unlabeled) vs. fake images
purpose: train G to produce images whose feature statistics match real ones
```

The generator never directly trains to fool the discriminator. Instead, it trains to match the discriminator's internal representation of real images. This indirect objective is more stable and does not suffer from vanishing gradients when the discriminator is strong.

---

## Why Feature Matching Instead of Adversarial G Loss?

The standard adversarial generator loss is:

```
loss_G = -log(1 - P(fake | G(z)))
```

When the discriminator is well-trained, `P(fake | G(z)) ≈ 1` and the gradient nearly vanishes. Early in training, the generator receives almost no learning signal.

Feature matching bypasses this by giving the generator a stable target: match the average intermediate activations of real images. The gradient flows regardless of how well the discriminator can identify fakes.

---

## The Semi-Supervised Learning Signal

The learning mechanism works as follows:

1. The `unlabeled_real_loss` forces the discriminator to build a good internal representation of what dermoscopy images look like — using 7,018 examples.
2. The `supervised_loss` then maps that rich representation onto the benign/malignant distinction — using only 200 examples, but on top of already-useful features.
3. The GAN training loop (fake_loss + feature_matching_loss) keeps the generator improving, which in turn keeps the discriminator's "realness" boundary from collapsing.

This is why the semi-supervised setup outperforms a purely supervised classifier trained on 200 images: the unlabeled data provides a strong inductive bias about the structure of the input space.
