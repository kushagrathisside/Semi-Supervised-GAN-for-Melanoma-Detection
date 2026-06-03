# Model Variants — Generator Training Signal

This document tracks the method variants explored for the melanoma SGAN augmentation
project. The discriminator is a K+1 semi-supervised classifier throughout
(`[P(benign), P(malignant), P(fake)]`); what changes across variants is **how the
generator is trained** and **how the discriminator is regularized against it**.

Data: 32×32 native, upsampled to 64×64. 200 labeled (100 benign / 100 malignant)
+ 7018 unlabeled images.

## Naming (paper ↔ dev)

| Paper name | Method | Dev tag | Role |
|---|---|---|---|
| **FM-SGAN** | Salimans discriminator feature-matching SGAN | v1 | baseline |
| **DINO-Mean** | frozen-DINO batch-mean matching as the sole generator signal | v2 | negative result |
| **Adv-R1** | adversarial SGAN with R1-regularized, rebalanced discriminator | v3.1 | backbone / ablation baseline |
| **DINO-MMD** *(Ours)* | Adv-R1 + per-sample DINO Maximum Mean Discrepancy | Run 2 | **proposed method** |

Intended paper usage: *"We evaluate **DINO-MMD** against the **FM-SGAN** baseline and the
**Adv-R1** ablation (DINO-MMD with the MMD term removed). **DINO-Mean** is reported as a
negative result establishing that naïve distribution-mean matching is insufficient."*

The discriminator configuration `Adv-R1` is the shared backbone for both Adv-R1 and
DINO-MMD; they differ only by the generator's `dino_mmd_weight` (0 vs. >0).

---

## FM-SGAN (baseline) — discriminator feature matching

**Generator loss:**
```
loss_G = || mean(D_features(real)) - mean(D_features(fake)) ||_1
```
`D_features` = discriminator penultimate activations ([B, 1280, 4, 4] = 20,480-dim).

**Discriminator:** `feature_maps=160` (~17.3M params), `lr_d=1e-4`.

**Rationale:** Standard Salimans et al. (2016) recipe — the generator matches the
*discriminator's own* feature statistics rather than directly maximizing the fooling
objective. Reference point the project aims to beat. (Note: in this project FM-SGAN was
never run to a verified-good FID; an earlier attempt was interrupted with no checkpoints.)

---

## DINO-Mean (negative result) — frozen-DINO batch-mean matching  ❌

**Idea (original intended contribution):** Replace the discriminator's *learned* features
with a *frozen foundation-model* feature space. Project DINOv2 patch tokens through a small
head pre-trained with Supervised Contrastive loss on the 200 labeled images, then match the
mean projection of real vs. fake batches.

**Generator loss:**
```
proj   = DINOProjectionHead( mean_patch_tokens( DINOv2(x @ 112px) ) )   # [B, 256], L2-normed
loss_G = || mean(proj(real)) - mean(proj(fake)) ||_1                    # batch-mean L1
       + λ_cc · Σ_c || anchor_c - confidence_weighted_mean(proj(fake), w_c) ||_1
```
`λ_cc` warmed 0→0.1 over epochs 50–100. **No adversarial term in G's objective.**

**Discriminator:** unchanged from FM-SGAN (`feature_maps=160`, `lr_d=1e-4`). Trained every
step but its verdict **never fed back to G** except as detached softmax weights on anchors.

**Result — full 600-epoch run** (archived in `outputs/archive_run1_dino_meanmatch/`):

| Metric | Value | Reading |
|---|---|---|
| FID (best / final) | **393 (ep249) / 423 (ep599)** | catastrophic — good GANs hit 20–80 |
| D labeled accuracy | 99.96% | memorized the 200 labeled images |
| D AUC-ROC (train) | 1.000 | overfit |
| `Loss/Fake` | **0.000 from epoch 50** | D detects every fake with P(fake)=1.0 |
| G loss | 0.011, stable | **misleading** (see below) |
| Generated images | colored checkerboard noise | no lesion structure |

**Root cause (diagnosed):**
1. **Adversarial signal died at epoch 50** — the 17M-param D memorized the labeled set,
   flagging every fake as fake (P(fake)=1.0, margin 0.99 vs. real). Irrelevant to G, because—
2. **G's only objective was batch-mean DINO matching, which is underdetermined.** Matching
   the *mean of patch-mean* projections collapses `256 × B` numbers into one 256-d vector;
   a checkerboard satisfies it. **Zero per-sample realism pressure** anywhere. G=0.011 only
   meant "your average embedding matches," not that any image looked real.
3. **Checkerboard is signal-driven, not architectural** — the generator uses
   `ConvTranspose2d(k=4, s=2, p=1)`, the recommended low-checkerboard config. Nothing in the
   loss penalized local structure.

**Verdict / paper takeaway:** distribution-*mean* matching alone cannot drive realistic
generation. Kept as a negative result.

---

## Adv-R1 (backbone / ablation baseline) — adversarial SGAN, R1-stabilized D  ✅

Fixes DINO-Mean's two failures: restores live per-sample adversarial pressure, and makes the
discriminator a **stable, informative** critic via R1 regularization instead of brittle
capacity tuning.

**Generator loss (MMD term off):**
```
loss_G = adversarial_weight · L_adv
L_adv  = -mean( log Σ_{c<K} softmax(D(fake))[c] ) = -mean( log P(real) )     # non-saturating
```
Computed in log-space (`logsumexp`); gives strong gradient *even when D is confident the
input is fake* — exactly the regime where DINO-Mean stalled.

**Discriminator stabilization — the path to a healthy critic:**

| Knob | DINO-Mean | (v3.0 attempt) | **Adv-R1 (v3.1)** | Why |
|---|---|---|---|---|
| `feature_maps` | 160 | 64 | **128** | 160 memorized; 64 collapsed; 128 is the middle ground |
| `lr_d / lr_g` | 1e-4 / 2e-4 | 5e-5 / 2e-4 | **3e-4 / 1e-4 (TTUR)** | D fast enough to stay informative |
| `beta1` | 0.5 | 0.5 | **0.0** | standard for R1-regularized GANs |
| R1 penalty | — | — | **γ=10, every 16 steps** | smooths D's real/fake boundary |

**R1 gradient penalty** (`r1_penalty`, Mescheder et al. 2018): penalizes
`E_x[ ||∇_x s(x)||² ]` where `s(x) = logsumexp(real-class logits) − fake-class logit` is the
discriminator's realness score on real images. fp32 double-backward, applied lazily every 16
steps with StyleGAN2 `(γ/2 × interval)` compensation. This is what prevents *both* failure
modes — over-sharpening (memorization, margin→0.99) and collapse (margin→0.01).

**Why a tuning detour (v3.0 → v3.1) happened:** the first rebalance over-corrected — cutting D
to 64 maps + halved LR made D too weak to separate real from fake (margin 0.013), so the
adversarial gradient became noise and the generator drifted back to checkerboard (FID 464).
R1 + a 128-map D + TTUR is the stable fix. (v3.0 is a dev tuning step, not a paper variant.)

**Validation (25-epoch smoke):**

| | DINO-Mean | v3.0 (too weak) | **Adv-R1** |
|---|---|---|---|
| D real/fake margin | 0.99 (memorized) | 0.013 (collapsed) | **0.448 (healthy)** |
| D P(fake): fake / real | 1.0 / 0.01 | 0.34 / 0.33 | **0.91 / 0.46** |
| D labeled acc | 99.96% | 0.61 | 0.76 |
| Generated images | checkerboard | checkerboard | **lesion-like pigmented blobs** |
| FID @ ep24 | (≈449 @ ep49) | 464 | 448 (smooth/low-texture; structure now correct) |

The checkerboard is eliminated for the first time and the discriminator is healthy. FID at 25
epochs is still high because early samples are smooth and lack fine texture; unlike DINO-Mean
and v3.0 (stuck on checkerboard), the structure is now correct, so FID has a real path to drop
over a full run. The full Adv-R1 run is the test of whether it does.

---

## DINO-MMD *(proposed method)* — Adv-R1 + per-sample DINO MMD

Adds the project's actual contribution on top of the working Adv-R1 backbone:

```
loss_G = adversarial_weight · L_adv
       + dino_mmd_weight     · MMD_rbf( proj(real_batch), proj(fake_batch) )
```

- **`MMD_rbf`** (`mmd_loss`): multi-bandwidth RBF-kernel Maximum Mean Discrepancy between the
  *sets* of individual DINO projections (not their means). Unlike DINO-Mean's linear-kernel-
  equivalent mean matching, the RBF kernel matches the **full distribution**, pressuring every
  fake toward the real manifold. Bandwidths set per call via the median heuristic.

**Rollout:** Adv-R1 is `dino_mmd_weight=0.0`; DINO-MMD sets `dino_mmd_weight=0.3` (or sweep).
DINO is only loaded when the weight is > 0. The improvement of DINO-MMD over Adv-R1 is the
headline ablation.

**Status of the headline ablation:** On the 32×32 liveProject data (held-out, n=40),
DINO-MMD did **not** beat Adv-R1 on classification (within noise; it only improved generator
FID 545→527). That data is also non-citeable (educational liveProject set). The decisive
test now runs on **HAM10000 @ 64×64** with the official ISIC-2018 test set (n=1511). Full
results, the held-out methodology, and the dataset-provenance story are in
[EXPERIMENTS.md](EXPERIMENTS.md).

---

## Code map

| Component | File |
|---|---|
| `adversarial_generator_loss`, `mmd_loss`, `r1_penalty` (Adv-R1 / DINO-MMD) | [training/losses.py](../training/losses.py) |
| `dino_feature_matching_loss`, `confidence_weighted_dino_loss` (DINO-Mean) | [training/losses.py](../training/losses.py) |
| `feature_matching_loss` (FM-SGAN) | [training/losses.py](../training/losses.py) |
| Generator step / loss composition | [training/trainer.py](../training/trainer.py) |
| `GeneratorLossConfig`, `DINOConfig`, R1 knobs in `TrainingConfig` | [utils/config.py](../utils/config.py) |
| Loss weights, D rebalance, TTUR, R1 | [configs/config.yaml](../configs/config.yaml) |
| Held-out classifier eval / supervised baseline / split | [evaluation/evaluate_classifier.py](../evaluation/evaluate_classifier.py), [evaluation/train_supervised_baseline.py](../evaluation/train_supervised_baseline.py), [evaluation/make_split.py](../evaluation/make_split.py) |
| HAM10000 → 64px binary SSL prep | [scripts/prep_ham10000.py](../scripts/prep_ham10000.py) |
| HAM10000 run configs | [configs/ham64_advr1.yaml](../configs/ham64_advr1.yaml), [configs/ham64_dinomMMD.yaml](../configs/ham64_dinomMMD.yaml) |
| Evaluation results, methodology, dataset provenance | [docs/EXPERIMENTS.md](EXPERIMENTS.md) |
| DINO-Mean (negative result) run artifacts | `outputs/archive_run1_dino_meanmatch/` |
