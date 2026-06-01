# Model Versions — Generator Training Signal

This document tracks the three iterations of the SGAN generator objective for the
melanoma augmentation project. The discriminator is a K+1 semi-supervised classifier
throughout (`[P(benign), P(malignant), P(fake)]`); what changes across versions is
**how the generator is trained** and **how the discriminator is balanced against it**.

All image data is 32×32 native, upsampled to 64×64 for training. 200 labeled
(100 benign / 100 malignant) + 7018 unlabeled images.

---

## v1 — Vanilla SGAN (Salimans-style feature matching)

**Generator loss:** discriminator feature matching.
```
loss_G = || mean(D_features(real)) - mean(D_features(fake)) ||_1
```
where `D_features` are the discriminator's penultimate activations
([B, 1280, 4, 4] = 20,480-dim).

**Discriminator:** `feature_maps=160` (~17.3M params), `lr_d=1e-4`.

**Rationale:** Standard Salimans et al. (2016) recipe. The generator matches the
*discriminator's own* intermediate feature statistics rather than directly maximizing
the fooling objective, which is more stable than naive adversarial G loss.

**Status:** Baseline. The feature-matching target co-adapts with the discriminator,
so it is a moving (but informative) signal. Never the research contribution — it's
the reference point the project aimed to beat.

---

## v2 — Frozen DINOv2 batch-mean matching  ❌ FAILED

**Idea (intended contribution):** Replace the discriminator's *learned* features with
a *frozen foundation-model* feature space. Project DINOv2 patch tokens through a small
head pre-trained with Supervised Contrastive loss on the 200 labeled images, then match
the mean projection of real vs. fake batches.

**Generator loss:**
```
proj = DINOProjectionHead( mean_patch_tokens( DINOv2(x @ 112px) ) )   # [B, 256], L2-normed
loss_G = || mean(proj(real)) - mean(proj(fake)) ||_1                  # batch-mean L1
       + λ_cc · Σ_c || anchor_c - confidence_weighted_mean(proj(fake), w_c) ||_1
```
`λ_cc` warmed up 0→0.1 over epochs 50–100. No adversarial term in G's objective.

**Discriminator:** unchanged from v1 (`feature_maps=160`, ~17.3M params, `lr_d=1e-4`).
Trained every step (supervised + unlabeled-real + fake losses) but its verdict
**never fed back to the generator** except as detached softmax weights on the anchors.

**Result — full 600-epoch run (archived in `outputs/archive_run1_dino_meanmatch/`):**

| Metric | Value | Reading |
|---|---|---|
| FID (best / final) | **393 (ep249) / 423 (ep599)** | catastrophic — good GANs hit 20–80 |
| D labeled accuracy | 99.96% | memorized the 200 labeled images |
| D AUC-ROC (train) | 1.000 | overfit |
| `Loss/Fake` | **0.000 from epoch 50** | D detects every fake with P(fake)=1.0 |
| G loss | 0.011, stable | **misleading** — see below |
| Generated images | colored checkerboard noise, no lesion structure | |

**Root cause (diagnosed, not guessed):**
1. **Adversarial signal died at epoch 50.** The 17M-param discriminator memorized the
   labeled set and flagged every fake as fake with P(fake)=1.0 (margin 0.99 vs. real).
   But this never mattered for G, because —
2. **G's only objective was batch-mean DINO matching, which is underdetermined.**
   Matching the *mean of patch-mean* projections collapses `256 × B` numbers into a
   single 256-d vector. A checkerboard pattern can satisfy "make your batch average
   embedding ≈ this target." There was **zero per-sample realism pressure** anywhere.
   G=0.011 only meant "your average embedding matches" — it said nothing about whether
   any individual image looked real.
3. **Checkerboard is signal-driven, not architectural.** The generator uses
   `ConvTranspose2d(k=4, s=2, p=1)` — the recommended low-checkerboard DCGAN config.
   Nothing in the loss penalized local structure, so artifacts were never corrected.

**Verdict:** Distributional *mean*-matching alone is insufficient to drive realistic
generation. This is a genuine negative result and is kept for the paper.

---

## v3 — Adversarial + per-sample DINO MMD, rebalanced D  ✅ CURRENT

Fixes both failure modes from v2: restores live adversarial pressure **and** makes the
DINO term operate per-sample instead of on the batch mean. Discriminator is rebalanced
so it cannot trivially win.

**Generator loss:**
```
loss_G = adversarial_weight · L_adv
       + dino_mmd_weight     · MMD_rbf( proj(real_batch), proj(fake_batch) )
```

- **`L_adv` — non-saturating adversarial loss** (`adversarial_generator_loss`):
  ```
  L_adv = -mean( log Σ_{c<K} softmax(D(fake))[c] ) = -mean( log P(real) )
  ```
  Computed in log-space via `logsumexp`. Gives strong gradient *even when D is confident
  the input is fake* — exactly the regime where v2 stalled. This is per-sample realism
  pressure routed through the discriminator.

- **`MMD_rbf` — per-sample distribution matching** (`mmd_loss`): multi-bandwidth RBF-kernel
  Maximum Mean Discrepancy between the *sets* of individual DINO projections (not their
  means). Unlike v2's linear-kernel-equivalent mean matching, the RBF kernel matches the
  full distribution, pressuring every fake toward the real manifold. Bandwidths set per
  call via the median heuristic so the loss self-scales to the embedding geometry.

**Discriminator rebalanced:**
| Knob | v2 | v3 | Why |
|---|---|---|---|
| `feature_maps` | 160 | **64** | 17.3M → 2.78M params; stop instant memorization |
| `lr_discriminator` | 1e-4 | **5e-5** | let G keep pace |

**Sequenced rollout (config `generator_loss`):**
- **Run 1 (current default):** `adversarial_weight=1.0, dino_mmd_weight=0.0`
  → pure adversarial SGAN baseline. Goal: confirm FID drops from ~400 into a sane range
  with the rebalanced discriminator. DINO is not even loaded when `dino_mmd_weight=0`.
- **Run 2 (contribution):** set `dino_mmd_weight=0.3` (or sweep) to layer the per-sample
  DINO MMD term on top of the working baseline, and measure the improvement over Run 1.

**Smoke test (3 epochs) vs. v2:**

| | v2 (failed) | v3 Run 1 |
|---|---|---|
| D params | 17.3M | 2.78M |
| G adversarial loss | 0.000 (dead) | 0.31–0.45 (alive, oscillating) |
| D loss | 0.32 (D dominating) | ~2.6 (competitive) |
| Throughput | 1.15 s/it | 3.0 it/s (~9 s/epoch) |

The adversarial loss oscillating at ~0.4 (⇒ D assigns P(real)≈0.67 to fakes) confirms
the generator is competitively fooling the discriminator — the dynamic v2 never had.

---

## Code map

| Component | File |
|---|---|
| `adversarial_generator_loss`, `mmd_loss` (v3) | [training/losses.py](../training/losses.py) |
| `dino_feature_matching_loss`, `confidence_weighted_dino_loss` (v2) | [training/losses.py](../training/losses.py) |
| `feature_matching_loss` (v1) | [training/losses.py](../training/losses.py) |
| Generator step / loss composition | [training/trainer.py](../training/trainer.py) |
| `GeneratorLossConfig`, `DINOConfig` | [utils/config.py](../utils/config.py) |
| Weights, D rebalance | [configs/config.yaml](../configs/config.yaml) |
| Failed v2 run artifacts | `outputs/archive_run1_dino_meanmatch/` |
