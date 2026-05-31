# Evaluation

How to measure image quality and model performance after training.

---

## Fréchet Inception Distance (FID)

**Lower is better.** FID measures the distance between the distribution of real images and generated images in Inception-v3 feature space. It captures both quality (are images realistic?) and diversity (do they cover the full distribution?).

### Quick evaluation

```python
from evaluation.metrics import compute_fid

fid = compute_fid(
    real_dir="evaluation/real_samples",   # prepared by prepare_real_fid.py
    fake_dir="data/generated"             # produced by generation mode
)
print(f"FID: {fid:.2f}")
```

Internally this calls `python -m pytorch_fid <real_dir> <fake_dir>` and parses the output. Both directories must contain at least ~50 images for stable statistics; 1,000+ is recommended.

### FID reference ranges for dermoscopy GANs

| FID | Interpretation |
|-----|----------------|
| < 50 | Good — images are realistic and diverse |
| 50–100 | Moderate — recognizable structure, some artifacts |
| 100–200 | Poor — blurry or mode-collapsed |
| > 200 | Training failed or checkpoint too early |

These are rough benchmarks for 64×64 medical images. Exact numbers depend on the reference dataset size and preprocessing.

### Preparing real samples

```bash
python evaluation/prepare_real_fid.py
```

This samples 1,000 images from the full dataset (labeled + unlabeled) into `evaluation/real_samples/`. Only needs to be run once.

---

## Inception Score (IS)

**Higher is better.** IS measures sharpness (does each image look like something?) and diversity (do different images look like different things?). Less reliable than FID for domain-specific images like dermoscopy.

```python
import torch
from evaluation.metrics import compute_inception_score

# Load generated images as a tensor [N, 3, H, W] in [-1, 1]
# ... load your images here ...

is_mean, is_std = compute_inception_score(images, batch_size=32)
print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
```

---

## Discriminator Classification Metrics

The discriminator doubles as a classifier. After training, evaluate its classification performance on the labeled test set.

```python
import torch
from evaluation.metrics import compute_precision_recall, compute_class_wise_accuracy

# Run discriminator on labeled images
model.eval()
with torch.no_grad():
    logits = model.discriminate(labeled_images)  # [N, 3]
    # Only use the real-class logits (drop the fake class)
    class_logits = logits[:, :2]

precision, recall, f1 = compute_precision_recall(class_logits, labels)
accuracy = compute_class_wise_accuracy(class_logits, labels)

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print(f"Benign accuracy: {accuracy['class_0_accuracy']:.3f}")
print(f"Malignant accuracy: {accuracy['class_1_accuracy']:.3f}")
```

**Important:** Malignant recall (sensitivity) is the most clinically meaningful metric. A missed malignant lesion is more dangerous than a false positive.

---

## Evaluation Workflow

```bash
# 1. Finish training (or load best checkpoint)
# 2. Generate synthetic images
make generate

# 3. Prepare real reference images (one-time)
python evaluation/prepare_real_fid.py

# 4. Compute FID
python -c "
from evaluation.metrics import compute_fid
fid = compute_fid('evaluation/real_samples', 'data/generated')
print(f'FID: {fid}')
"
```

---

## Checkpoints for Evaluation

The trainer saves `outputs/checkpoints/best_generator.pt` whenever generator loss improves. This checkpoint minimizes the feature matching loss, which does not directly measure visual quality. FID is a more reliable criterion for selecting the best model — consider evaluating FID on checkpoints from multiple epochs and selecting the one with the lowest FID.

Periodic checkpoints at every 50 epochs are saved as `checkpoint_epoch_XXXXXX.pt`.
