# Experiments & Evaluation

Companion to [MODEL_VERSIONS.md](MODEL_VERSIONS.md) (which defines the FM-SGAN /
DINO-Mean / Adv-R1 / DINO-MMD variants). This file tracks **how they are evaluated**
and **what the held-out numbers actually say**.

Target venue: **ISIC Skin Image Analysis Workshop @ MICCAI 2026** (deadline **July 1
2026**, LNCS/Springer, citeable). Goal: an honest, reproducible study on a proper
dataset — a *positive* method paper if DINO-MMD transfers, an honest *cautionary*
study if it doesn't.

---

## Evaluation methodology

The K+1 discriminator **is** the classifier (benign vs malignant from the first two
logits). Two non-negotiable rules:

1. **Held-out only.** The discriminator trains on the labeled set, so metrics on those
   images measure *memorization*, not generalization. Evaluation uses a test set the
   model never saw. Scripts: `evaluation/evaluate_classifier.py` (a prominent in-sample
   warning fires if no `--test-dir` is given), `evaluation/train_supervised_baseline.py`
   (supervised-only reference), `evaluation/make_split.py` (fixed stratified split).
2. **No test peeking.** Final-epoch checkpoint (`latest.pt`) is evaluated, not the
   train-accuracy-selected `best_generator.pt` (that biases toward the most overfit model).

### The in-sample → held-out collapse (a key honest finding)

On the original 32×32 data, the epoch-584 discriminator scored **0.97 acc / 0.998 AUC
in-sample** (all 200 training images). On a held-out split the *same architecture*
scored **0.625 acc / 0.683 AUC**. The 0.97 number was almost entirely memorization —
exactly the trap that in-sample reporting hides in small-data dermatology work.

---

## Dataset provenance — why we left the original data

The original `data/labeled` (200, 32×32) + `data/unlabeled` (7018) is the **Manning
liveProject "Semi-Supervised GANs for Melanoma Detection"** (Olga Petrova) dataset — an
*educational* set, downsampled to 32×32, with **no documented upstream source and no
citation/license**. It also has an official 600-image test split that was never provided.
It is **not publishable** (medical-image provenance + license are mandatory).

→ We pivoted to **HAM10000** (the canonical, citeable ISIC-2018 dataset) at 64×64. This
also fixes comparability and, crucially, gives **real image detail** (native 600×450 →
64px) instead of 32px upsampled to 64.

---

## Result 1 — Gate on the 32×32 liveProject data (held-out, n=40)

A hand-made stratified 160/40 split (`data/splits/split_seed42.json`). Purpose: decide
fast whether DINO-MMD adds anything before investing in a proper dataset.

| Model | Acc | ROC-AUC | Sensitivity | Specificity | Confusion (TN/FP/FN/TP) |
|---|---|---|---|---|---|
| Supervised | 0.625 | 0.683 | 0.55 | 0.70 | 14/6/9/11 |
| **Adv-R1** | **0.675** | **0.723** | 0.70 | 0.65 | 13/7/6/14 |
| DINO-MMD | 0.650 | 0.713 | 0.70 | 0.60 | 12/8/6/14 |

Generator FID (held-out generators): Adv-R1 **545.5**, DINO-MMD **527.1** (−18).

**Verdict.** Adv-R1 (the SGAN) directionally beats supervised, biggest gain on
**sensitivity** (0.55→0.70). But **DINO-MMD does NOT beat Adv-R1** on classification
(1 image worse, within noise); it only improves FID slightly, and both generators are
poor (FID ~530). At n=40 every difference is within noise. So: no novel *positive*
classification result on this data, and the contribution does not transfer downstream
here. This gated the move to a better dataset/resolution.

---

## Result 2 — HAM10000 @ 64×64 (held-out = official ISIC-2018 test, n=1511)

Prepared by `scripts/prep_ham10000.py` → `data/ham64/`:
- **labeled**: 100 malignant + 100 benign (few-label SSL regime)
- **unlabeled**: 9,815
- **test**: official ISIC-2018 Task-3 set, **171 malignant + 1,340 benign = 1,511**
  (real held-out, large enough that **k-fold is unnecessary** — tight CIs from a single run)

Binary mapping: `dx == 'mel'` → malignant, else benign. Configs: `configs/ham64_advr1.yaml`,
`configs/ham64_dinomMMD.yaml` (both `num_workers=16`). Eval: `--test-dir data/ham64/test`.

| Model | Acc | ROC-AUC | Sensitivity | Specificity | Confusion (TN/FP/FN/TP) | Status |
|---|---|---|---|---|---|---|
| Supervised | 0.633 | **0.795** | 0.825 | 0.608 | 815/525/30/141 | ✅ done |
| Adv-R1 | — | — | — | — | — | 🟢 training |
| DINO-MMD | — | — | — | — | — | ⏳ queued |

Notes:
- Real 64px data lifts the supervised baseline from AUC 0.683 (32px) → **0.795** — a
  respectable, publishable reference, confirming the resolution/quality pivot was right.
- The test set is imbalanced (~11% malignant), so **ROC-AUC, sensitivity, specificity**
  are the fair metrics; raw accuracy/precision are threshold-sensitive (the balanced-trained
  model over-calls malignant → 525 FP, low precision but high recall).
- Open question this run answers: does semi-supervised GAN training beat AUC 0.795, and
  does the **DINO-MMD term add anything on real data** (the question n=40 couldn't settle)?

---

## Decision tree (where this goes)

- **DINO-MMD > Adv-R1 on HAM10000** → positive method paper: "per-sample foundation-feature
  distribution matching improves semi-supervised skin-lesion GAN augmentation."
- **DINO-MMD ≈/< Adv-R1** → honest cautionary study on a citeable dataset: the
  foundation-feature term improves generation but does not transfer to classification,
  plus the in-sample→held-out memorization finding and the mean-matching→MMD+R1 diagnosis.

Either outcome is ISIC-W-appropriate and unblocked on provenance.

---

## Reproduce

```bash
source venv/bin/activate
# data (HAM10000 + ISIC2018 test → 64px binary SSL split)
python -m scripts.prep_ham10000 --labeled-per-class 100 --size 64 --seed 42
# DINO projection head (SupCon on the 200 labeled)
python training/pretrain_projhead.py --config configs/ham64_dinomMMD.yaml
# supervised baseline → AUC 0.795
python -m evaluation.train_supervised_baseline --config configs/ham64_advr1.yaml \
  --train-dir data/ham64/labeled --test-dir data/ham64/test --epochs 300 \
  --out outputs/ham64/supervised_baseline.json
# Adv-R1 and DINO-MMD (run sequentially — single 8GB GPU)
python main.py --mode train --config configs/ham64_advr1.yaml
python -m evaluation.evaluate_classifier --checkpoint outputs/ham64_advr1/checkpoints/latest.pt \
  --test-dir data/ham64/test --out outputs/ham64/advr1_test.json
python main.py --mode train --config configs/ham64_dinomMMD.yaml
python -m evaluation.evaluate_classifier --checkpoint outputs/ham64_dinomMMD/checkpoints/latest.pt \
  --test-dir data/ham64/test --out outputs/ham64/dinomMMD_test.json
```
