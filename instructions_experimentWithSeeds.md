# Multi-Seed Experiment — Handoff Instructions

**For:** the contributor running this on the lab GPU (24 GB).
**Goal:** finish the 3-seed evaluation so the main result has `mean ± std` and a
significance test, instead of single-run numbers.

Read [docs/MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the full background. This file is the standalone how-to.

---

## 1. What we're testing

A semi-supervised melanoma GAN where the K+1 **discriminator is the classifier**
(benign vs malignant). Three conditions, all trained on **HAM10000 @ 64×64** (100
labeled/class + 9,815 unlabeled) and evaluated on the **official ISIC-2018 Task-3 test
set (1,511 images)**:

| Condition | What it is | Generator loss |
|---|---|---|
| **Supervised** | same discriminator architecture, supervised-only (no GAN) | — |
| **Adv-R1** | semi-supervised SGAN backbone | adversarial only |
| **DINO-MMD** | Adv-R1 + per-sample DINO feature MMD (**the contribution**) | adversarial + `0.3·MMD` |

Single-run AUC differences are ~0.01 and the per-checkpoint AUC oscillates ±0.01, so we
need **3 seeds per condition** to know whether DINO-MMD's edge is real.

## 2. What's already done (seed 42 + supervised) — DO NOT re-run

| Condition | seed 42 | seed 123 | seed 2024 |
|---|---|---|---|
| Supervised | AUC 0.795 ✅ | 0.791 ✅ | 0.781 ✅ (mean **0.789 ± 0.007**) |
| Adv-R1 | AUC 0.797 ✅ | **TODO** | **TODO** |
| DINO-MMD | AUC 0.807 ✅ | **TODO** | **TODO** |

**Your job: the 4 TODO runs** — Adv-R1 and DINO-MMD, each at seed 123 and seed 2024.

---

## 3. ⚠️ Consistency rules — do not change these (they break the comparison)

1. **Data-prep seed is ALWAYS 42** (`prep_ham10000.py --seed 42`). This fixes *which*
   200 images are labeled. It is **different** from the training seed.
2. **Training seed** is the only thing that varies: `experiment.seed` ∈ {123, 2024}
   (42 is done). It's already set in the provided configs.
3. **Keep `batch_size: 256`** — do NOT raise it even though the 24 GB GPU could. Seed-42
   runs used 256; changing it makes results incomparable.
4. **Keep `epochs: 600`**, the augmentation, learning rates, and architecture exactly as
   in the configs. Only the seed changes.

---

## 4. Environment setup

Needs Python 3.12 and a CUDA 12.x GPU.

```bash
git clone <repo-url> melanoma-sgan && cd melanoma-sgan
python3.12 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 1) PyTorch first, matched to the lab's CUDA (this pins CUDA 12.8 — adjust if different)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 2) everything else (torch is already satisfied, so this won't touch it)
pip install -r requirements.txt

# sanity
python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

(`requirements.txt` is pinned to the working environment. `torchmetrics`/`termcolor` are
needed by the DINOv2 hub loader, `pandas` by the data-prep, `scipy` by the significance
analysis.)

---

## 5. Data prep (REQUIRED — data is gitignored, not in the repo)

Downloads HAM10000 + the official ISIC-2018 test set (~3.2 GB) from Harvard Dataverse and
builds the 64×64 binary split. **The `--seed 42` here is mandatory.**

```bash
mkdir -p data/ham10000/raw && cd data/ham10000/raw
# label files
curl -sL "https://dataverse.harvard.edu/api/access/datafile/4338392" -o HAM10000_metadata.tab
curl -sL "https://dataverse.harvard.edu/api/access/datafile/6924466" -o ISIC2018_Test_GroundTruth.tab
# images (parallel)
curl -sL "https://dataverse.harvard.edu/api/access/datafile/3172585" -o HAM10000_images_part_1.zip &
curl -sL "https://dataverse.harvard.edu/api/access/datafile/3172584" -o HAM10000_images_part_2.zip &
curl -sL "https://dataverse.harvard.edu/api/access/datafile/3855824" -o ISIC2018_Test_Images.zip &
wait
cd ../../..

python -m scripts.prep_ham10000 --labeled-per-class 100 --size 64 --seed 42
```

Expected output:
```
labeled : 100 malignant + 100 benign
unlabeled: 9815
test    : 171 malignant + 1340 benign
```
If you don't see exactly these counts, stop — something is wrong with the download.

---

## 6. Projection head for DINO-MMD (REQUIRED — also gitignored)

```bash
python training/pretrain_projhead.py --config configs/ham64_dinomMMD.yaml
# -> writes outputs/ham64/projection_head.pt  (used by all DINO-MMD seeds)
```

---

## 7. Generate the seed configs (skip if already in the repo)

The four configs may already be committed. If `ls configs/ham64_advr1_s123.yaml` exists,
skip this. Otherwise generate them (deterministic):

```bash
python - <<'EOF'
import yaml
def mk(base, name, seed):
    c = yaml.safe_load(open(f"configs/{base}.yaml"))
    c["experiment"]["name"] = name
    c["experiment"]["seed"] = seed
    c["output"]["checkpoint_dir"] = f"outputs/{name}/checkpoints"
    c["output"]["sample_dir"]     = f"outputs/{name}/samples"
    c["output"]["log_dir"]        = f"outputs/{name}/logs"
    yaml.dump(c, open(f"configs/{name}.yaml","w"), sort_keys=False)
for seed in (123, 2024):
    mk("ham64_advr1",    f"ham64_advr1_s{seed}",    seed)
    mk("ham64_dinomMMD", f"ham64_dinomMMD_s{seed}", seed)
print("configs generated")
EOF
```

---

## 8. Run the 4 experiments (the main task)

Each command trains then auto-evaluates on the official test set (writes a JSON).

```bash
source venv/bin/activate
for cfg in ham64_advr1_s123 ham64_advr1_s2024 ham64_dinomMMD_s123 ham64_dinomMMD_s2024; do
  echo "===== $cfg ====="
  python main.py --mode train --config configs/$cfg.yaml
  python -m evaluation.evaluate_classifier \
    --checkpoint outputs/$cfg/checkpoints/latest.pt \
    --config configs/$cfg.yaml \
    --test-dir data/ham64/test \
    --out outputs/ham64/${cfg}_test.json
done
echo "ALL SEEDS DONE"
```

**Timing / VRAM (per run):** Adv-R1 ≈ 4 GB, ~2–3 h. DINO-MMD ≈ 7 GB, ~5–7 h. Sequential
total ≈ 17 h on an 8 GB card; the 24 GB lab GPU should be faster.

**Optional speedup (24 GB only):** these runs are data-loading-bound (GPU ~50–60% util),
so 2–3 can run **in parallel** — each config has its own output dir, so they won't collide.
E.g. launch Adv-R1 s123 and DINO-MMD s123 together, then the s2024 pair. Watch memory with
`nvidia-smi`; keep total well under 24 GB. Run the matching `evaluate_classifier` after each
training finishes. (Sequential is simpler and foolproof — only parallelize if comfortable.)

To survive a disconnected SSH session, wrap the loop:
```bash
nohup bash -c 'for cfg in ...; do ...; done' > seeds.log 2>&1 &
tail -f seeds.log
```

---

## 9. What to send back

The four result JSONs (small text files):
```
outputs/ham64/ham64_advr1_s123_test.json
outputs/ham64/ham64_advr1_s2024_test.json
outputs/ham64/ham64_dinomMMD_s123_test.json
outputs/ham64/ham64_dinomMMD_s2024_test.json
```
Also keep the trained checkpoints (`outputs/<cfg>/checkpoints/latest.pt`) in case we need
per-sample scores for the DeLong/bootstrap significance test.

Each JSON has `roc_auc`, `sensitivity`, `specificity`, `accuracy`, and the confusion
matrix. We'll aggregate to `mean ± std` per condition and test whether DINO-MMD's edge
over Adv-R1 holds across seeds.

---

## 10. Quick sanity checklist before the long runs

- [ ] `nvidia-smi` shows the 24 GB GPU and `torch.cuda.is_available()` is `True`
- [ ] data prep printed `100/100 labeled, 9815 unlabeled, 171/1340 test`
- [ ] `outputs/ham64/projection_head.pt` exists
- [ ] `configs/ham64_dinomMMD_s123.yaml` shows `seed: 123` and `dino_mmd_weight: 0.3`
- [ ] a 2-epoch smoke test runs without error:
      `python main.py --mode train --config configs/ham64_advr1_s123.yaml` (Ctrl+C after epoch 1–2)
