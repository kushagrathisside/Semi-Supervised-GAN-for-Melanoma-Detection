# Data Pipeline

Dataset structure, preprocessing, and how to prepare images for FID evaluation.

---

## Dataset Structure

```
data/
├── labeled/
│   ├── benign/        # 100 labeled benign dermoscopy images
│   └── malignant/     # 100 labeled malignant dermoscopy images
├── unlabeled/
│   └── images/        # 7,018 unlabeled dermoscopy images
└── generated/         # Synthetic images produced by the generator
```

Labels are encoded by directory name: `benign → 0`, `malignant → 1`.

Supported file extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`.

---

## Dataset Classes

**File:** [`datasets/melanoma_dataset.py`](../datasets/melanoma_dataset.py)

### `LabeledMelanomaDataset`

Loads from `data/labeled/`. Returns `(image_tensor, label)` tuples.

```python
from datasets.melanoma_dataset import LabeledMelanomaDataset

dataset = LabeledMelanomaDataset(root_dir="data/labeled", image_size=64)
image, label = dataset[0]
# image: torch.Tensor [3, 64, 64], values in [-1, 1]
# label: int, 0=benign or 1=malignant
```

### `UnlabeledMelanomaDataset`

Loads from `data/unlabeled/images/`. Returns image tensors only.

```python
from datasets.melanoma_dataset import UnlabeledMelanomaDataset

dataset = UnlabeledMelanomaDataset(root_dir="data/unlabeled/images", image_size=64)
image = dataset[0]
# image: torch.Tensor [3, 64, 64]
```

---

## Preprocessing Pipeline

**File:** [`utils/transforms.py`](../utils/transforms.py)

Applied to all images (labeled and unlabeled) during loading:

```
1. Resize to (image_size × image_size)      — default 64×64
2. RandomHorizontalFlip(p=0.5)              — data augmentation
3. ToTensor()                               — converts to [0, 1] float
4. Normalize(mean=[0.5,0.5,0.5],
             std=[0.5,0.5,0.5])             — rescales to [-1, 1]
```

The `[-1, 1]` normalization is required because the Generator outputs through a `Tanh` activation. Real and fake images must be on the same scale for the discriminator to make meaningful comparisons.

To denormalize for visualization: `img = img * 0.5 + 0.5` (converts back to `[0, 1]`).

---

## Class Imbalance

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| Labeled | 100 | 100 | 200 |
| Unlabeled | — | — | 7,018 |

The labeled set is balanced (100/100). However, the unlabeled pool likely reflects the natural class distribution of dermoscopy datasets (benign images are more common). The semi-supervised training does not require unlabeled class balance since unlabeled images are only used for the "not-fake" objective.

---

## Data Loader Configuration

From `train_sgan.py`:

```python
DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

`pin_memory=True` speeds up CPU→GPU transfers. Reduce `num_workers` if you see dataloader-related crashes on Windows.

The **epoch length is determined by the unlabeled loader** (~28 batches at batch_size=256). The labeled loader is cycled — when exhausted mid-epoch it resets and continues, so every unlabeled batch is paired with a labeled batch.

---

## Preparing Real Images for FID

**File:** [`evaluation/prepare_real_fid.py`](../evaluation/prepare_real_fid.py)

Randomly samples up to 1,000 images from the full dataset (labeled benign + malignant + unlabeled) and copies them to `evaluation/real_samples/` for use as the FID reference distribution.

```bash
python evaluation/prepare_real_fid.py
```

This only needs to be run once. The resulting `evaluation/real_samples/` directory is already populated.

---

## Generated Images

After training, run generation to populate `data/generated/`:

```bash
make generate
# equivalent to:
python main.py --mode generate \
  --checkpoint outputs/checkpoints/best_generator.pt \
  --num_images 2000
```

Images are saved as `gen_000000.png`, `gen_000001.png`, … in `data/generated/`.

These can then be used to augment the labeled training set for a downstream classifier, or evaluated against real images via FID.
