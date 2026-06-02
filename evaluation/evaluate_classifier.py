"""
Evaluate the SGAN discriminator as a benign/malignant classifier.

The K+1 discriminator outputs [P(benign), P(malignant), P(fake)]; this script uses
the first two logits for binary melanoma classification and reports the standard
clinical metrics.

IMPORTANT — held-out vs in-sample:
    The 200 labeled images in data/labeled/ were ALL used during SGAN training.
    Evaluating on them measures memorization, not generalization, and is NOT valid
    for a paper's results table. Pass --test-dir pointing at a directory of UNSEEN
    labeled images (same benign/ + malignant/ layout) for a valid estimate. Without
    it, the script evaluates on the training images and prints an explicit warning.

Usage:
    # In-sample sanity check (training images):
    python -m evaluation.evaluate_classifier --checkpoint outputs/checkpoints/best_generator.pt

    # Valid held-out evaluation:
    python -m evaluation.evaluate_classifier \
        --checkpoint outputs/checkpoints/best_generator.pt \
        --test-dir data/test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.sgan import SGAN
from utils.config import SGANConfig
from datasets.melanoma_dataset import LabeledMelanomaDataset


def evaluate(checkpoint: str, config_path: str, test_dir: str | None) -> dict:
    cfg = SGANConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SGAN(cfg).to(device)
    ck = torch.load(checkpoint, map_location=device)
    model.discriminator.load_state_dict(ck["discriminator_state_dict"])
    D = model.discriminator.eval()

    in_sample = test_dir is None
    data_root = cfg.dataset.labeled_path if in_sample else test_dir

    # Deterministic eval transforms (no random augmentation).
    ds = LabeledMelanomaDataset(data_root, train=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = D(imgs.to(device))[:, : cfg.dataset.num_classes]
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    y = torch.cat(all_labels).numpy()                      # 0=benign, 1=malignant
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(1).numpy()
    score_mal = probs[:, 1]

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics = {
        "checkpoint": checkpoint,
        "epoch": ck.get("epoch"),
        "in_sample": in_sample,
        "data_root": str(data_root),
        "n_samples": int(len(y)),
        "n_benign": int((y == 0).sum()),
        "n_malignant": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, preds)),
        "precision_melanoma": float(precision_score(y, preds, pos_label=1, zero_division=0)),
        "recall_melanoma": float(recall_score(y, preds, pos_label=1, zero_division=0)),
        "f1_melanoma": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score_mal)) if len(np.unique(y)) > 1 else None,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return metrics


def print_report(m: dict) -> None:
    print(f"Checkpoint: {m['checkpoint']}  (epoch {m['epoch']})")
    if m["in_sample"]:
        print("\n" + "!" * 72)
        print("!! IN-SAMPLE: evaluated on the TRAINING images. Measures memorization,")
        print("!! NOT generalization. NOT valid for a paper. Use --test-dir for a")
        print("!! held-out estimate.")
        print("!" * 72)
    print(f"\nData: {m['data_root']}  "
          f"(n={m['n_samples']}: {m['n_benign']} benign + {m['n_malignant']} malignant)\n")
    rows = [
        ("Accuracy", m["accuracy"]),
        ("Precision (melanoma)", m["precision_melanoma"]),
        ("Recall (melanoma)", m["recall_melanoma"]),
        ("F1-score (melanoma)", m["f1_melanoma"]),
        ("ROC-AUC", m["roc_auc"]),
        ("Sensitivity (melanoma)", m["sensitivity"]),
        ("Specificity (benign)", m["specificity"]),
    ]
    for name, val in rows:
        print(f"  {name:<26}{val:.4f}" if val is not None else f"  {name:<26}N/A")
    c = m["confusion_matrix"]
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("              pred_benign  pred_malig")
    print(f"  true_benign   {c['tn']:>6}       {c['fp']:>6}")
    print(f"  true_malig    {c['fn']:>6}       {c['tp']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--test-dir", default=None,
                    help="Directory of UNSEEN labeled images (benign/ + malignant/). "
                         "Omit to evaluate in-sample on the training images.")
    ap.add_argument("--out", default=None, help="Optional path to save metrics JSON.")
    args = ap.parse_args()

    m = evaluate(args.checkpoint, args.config, args.test_dir)
    print_report(m)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"\nSaved metrics -> {args.out}")


if __name__ == "__main__":
    main()
