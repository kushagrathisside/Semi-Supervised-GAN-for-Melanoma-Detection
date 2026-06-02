"""
Supervised-only baseline classifier for the held-out augmentation evaluation.

Trains the SAME discriminator architecture as the SGAN, but purely supervised
(cross-entropy on benign/malignant, no fake class, no unlabeled data, no generator)
on the train split, then evaluates on the held-out test split. This isolates the
benefit of the semi-supervised GAN training: any gap between this baseline and an
SGAN variant evaluated the same way is attributable to the GAN / unlabeled data /
generator, not to the network or the data split.

Methodology notes (kept honest):
  - Trains on data/labeled_train only; evaluates on data/labeled_test only.
  - Reports the FINAL-epoch test metrics (no test-set model selection / peeking).
  - Uses the same augmentation transforms as SGAN training for fairness.

Usage:
    python -m evaluation.train_supervised_baseline \
        --train-dir data/labeled_train --test-dir data/labeled_test \
        --epochs 300 --out outputs/heldout/supervised_baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.discriminator import Discriminator
from utils.config import SGANConfig
from datasets.melanoma_dataset import LabeledMelanomaDataset


def _evaluate(D, loader, num_classes, device):
    D.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits_all.append(D(imgs.to(device))[:, :num_classes].cpu())
            y_all.append(labels)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision_melanoma": float(precision_score(y, preds, pos_label=1, zero_division=0)),
        "recall_melanoma": float(recall_score(y, preds, pos_label=1, zero_division=0)),
        "f1_melanoma": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) > 1 else None,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train-dir", default="data/labeled_train")
    ap.add_argument("--test-dir", default="data/labeled_test")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/heldout/supervised_baseline.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = SGANConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = cfg.dataset.num_classes

    train_ds = LabeledMelanomaDataset(args.train_dir, train=True)
    test_ds = LabeledMelanomaDataset(args.test_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)),
                              shuffle=True, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"train={len(train_ds)}  test={len(test_ds)}  (same Discriminator, supervised-only)")

    D = Discriminator(cfg.discriminator.feature_maps, cfg.dataset.num_channels, nc).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        D.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = D(imgs)[:, :nc]
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            tr_acc = _evaluate(D, train_loader, nc, device)["accuracy"]
            te_acc = _evaluate(D, test_loader, nc, device)["accuracy"]
            print(f"  epoch {epoch+1:>4}: train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

    metrics = _evaluate(D, test_loader, nc, device)
    metrics.update({"model": "supervised_baseline", "epochs": args.epochs,
                    "train_dir": args.train_dir, "test_dir": args.test_dir,
                    "n_test": len(test_ds), "in_sample": False})

    print("\n=== Supervised baseline — HELD-OUT test metrics ===")
    for k in ["accuracy", "precision_melanoma", "recall_melanoma", "f1_melanoma",
              "roc_auc", "sensitivity", "specificity"]:
        v = metrics[k]
        print(f"  {k:<22}{v:.4f}" if v is not None else f"  {k:<22}N/A")
    c = metrics["confusion_matrix"]
    print(f"  confusion: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
