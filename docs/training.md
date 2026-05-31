# Training

How the training loop works and guidance for tuning hyperparameters.

---

## Training Loop

**Files:** [`training/train_sgan.py`](../training/train_sgan.py), [`training/trainer.py`](../training/trainer.py)

Each epoch iterates over `unlabeled_loader`. The labeled loader is cycled in parallel — when it runs out, it resets. This means the unlabeled set defines the epoch length (~28 batches at batch size 256 with 7,018 images).

### Per-step sequence

```
For each unlabeled batch:

  1. Sample labeled batch (cycle if needed)
  2. Generate fake images: z ~ N(0,1),  fake = G(z)

  ── Discriminator step ──────────────────────────
  3. Logits on labeled batch  → supervised_loss
  4. Logits on unlabeled batch → unlabeled_real_loss
  5. Logits on fake.detach()  → fake_loss
  6. loss_D = sum of above
  7. Backward + clip_grad_norm(D, max=5.0) + optimizer_D.step()

  ── Generator step ───────────────────────────────
  8. New z ~ N(0,1),  fake = G(z)
  9. D(fake, return_features=True)  → fake_features
  10. D(unlabeled, return_features=True) → real_features
  11. loss_G = feature_matching_loss(real_features, fake_features)
  12. Backward + optimizer_G.step()
```

All forward passes use Automatic Mixed Precision (`torch.amp.autocast`). Gradient scaling (`GradScaler`) prevents underflow in fp16.

Note: fake images are re-generated for the generator step (step 8) so the generator receives fresh gradients uncontaminated by the discriminator step.

---

## Checkpointing

Checkpoints are saved to `outputs/checkpoints/`:

| File | When saved |
|------|-----------|
| `latest.pt` | Every `save_interval` epochs (default: 5) |
| `best_generator.pt` | When generator loss improves |
| `checkpoint_epoch_XXXXXX.pt` | Every 50 epochs |

Each checkpoint contains generator state dict, discriminator state dict, both optimizer state dicts, epoch number, and losses.

To resume training, call `trainer.load_checkpoint("outputs/checkpoints/latest.pt")` before the epoch loop.

---

## TensorBoard Metrics

Launch with `make tensorboard`.

| Scalar | Meaning |
|--------|---------|
| `Loss/Discriminator` | Combined D loss per epoch |
| `Loss/Generator` | Feature matching loss per epoch |
| `Loss/Supervised` | Classification loss on labeled images |
| `Loss/UnlabeledReal` | Loss pushing unlabeled → not-fake |
| `Loss/Fake` | Loss pushing generated → fake |
| `Loss/FeatureMatching` | Same as Generator (redundant alias) |

---

## Hyperparameter Guidance

Current defaults from `configs/config.yaml`:

```yaml
training:
  batch_size: 256
  epochs: 600
  lr_generator: 0.0002
  lr_discriminator: 0.0001
  beta1: 0.5
  beta2: 0.999
  lr_scheduler_type: none
```

### Learning rates

The 2:1 ratio (G:D) is intentional. The discriminator has a harder joint objective (three losses), so a lower LR prevents it from overpowering the generator early. If `Loss/Discriminator` collapses to near zero within the first 20 epochs, lower `lr_discriminator` further or raise `lr_generator`.

### Batch size

256 fills the GPU well with 64×64 images. If you increase resolution to 128×128, halve the batch size or expect OOM errors.

### Epochs

600 epochs is the target. The discriminator loss in the current run was still trending downward at epoch 280 (D≈0.7), suggesting training had not converged. The full 600 epochs are needed for the generator to produce high-quality images.

### LR Scheduler

Four options in `lr_scheduler_type`:

| Value | Behavior |
|-------|---------|
| `none` | Constant LR (current default) |
| `step` | Multiply by `lr_decay_factor` every `lr_decay_steps` epochs |
| `exponential` | Multiply by `lr_decay_factor` every epoch |
| `cosine` | Cosine annealing over `epochs` total |

Cosine annealing is a reasonable first experiment if the run plateaus in the 400–500 epoch range.

### Feature maps

`feature_maps: 160` gives ~16M generator parameters and ~16M discriminator parameters. This is a high-capacity configuration. If training on CPU or low-VRAM GPU, drop to 64 for a ~3× reduction in compute.

---

## Expected Loss Behavior

| Phase | D loss | G loss |
|-------|--------|--------|
| Epochs 0–20 | 2.0–2.5 (high, learning) | 0.005–0.02 |
| Epochs 20–150 | Steady decline toward 1.0 | Stable, small |
| Epochs 150–300 | 0.7–1.0, oscillating | Stable, small |
| Epochs 300–600 | Should stabilize ~0.6–0.8 | May slowly decrease |

Very low D loss (< 0.3) suggests the discriminator is dominating — lower `lr_discriminator`. Very high D loss that doesn't decrease suggests the generator is not improving — check feature matching loss and verify unlabeled data is loading correctly.

---

## Running

```bash
# Full training run
make train

# Custom config
python main.py --mode train --config configs/config.yaml --log_level DEBUG

# Monitor
make tensorboard
```
