"""
Create a FIXED, reproducible stratified train/test split of the 200 labeled images.

This must be run ONCE, BEFORE any (re)training, so the test set is genuinely held out.
All subsequent experiments (SGAN variants + supervised baseline) train on the train
split only and report on the test split.

Output layout (image copies, so the existing dataloader works unchanged):
    data/labeled_train/{benign,malignant}/   (default 80 + 80)
    data/labeled_test/{benign,malignant}/    (default 20 + 20)
A manifest (data/splits/split_seed{S}.json) records exactly which file went where,
so the split is auditable and reproducible.

Usage:
    python -m evaluation.make_split --test-frac 0.2 --seed 42
"""

import argparse
import json
import shutil
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def stratified_split(src_root: Path, test_frac: float, seed: int):
    import random
    rng = random.Random(seed)
    split = {"train": {}, "test": {}}
    for cls in ("benign", "malignant"):
        files = sorted(p.name for p in (src_root / cls).iterdir()
                       if p.suffix.lower() in VALID_EXTS)
        rng.shuffle(files)
        n_test = round(len(files) * test_frac)
        split["test"][cls] = sorted(files[:n_test])
        split["train"][cls] = sorted(files[n_test:])
    return split


def materialize(src_root: Path, split: dict, train_root: Path, test_root: Path):
    for subset, root in (("train", train_root), ("test", test_root)):
        for cls in ("benign", "malignant"):
            out = root / cls
            out.mkdir(parents=True, exist_ok=True)
            for name in split[subset][cls]:
                shutil.copy2(src_root / cls / name, out / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/labeled")
    ap.add_argument("--train-out", default="data/labeled_train")
    ap.add_argument("--test-out", default="data/labeled_test")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    train_root, test_root = Path(args.train_out), Path(args.test_out)
    if train_root.exists() or test_root.exists():
        raise SystemExit(
            f"Refusing to overwrite existing split dirs ({train_root}, {test_root}). "
            "Delete them first if you really want to re-split — but note that changing "
            "the split invalidates any model already trained on the old one."
        )

    split = stratified_split(src, args.test_frac, args.seed)
    materialize(src, split, train_root, test_root)

    manifest_path = Path("data/splits") / f"split_seed{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "test_frac": args.test_frac,
                "src": str(src), "split": split}
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for subset in ("train", "test"):
        b = len(split[subset]["benign"]); m = len(split[subset]["malignant"])
        print(f"{subset:>5}: {b} benign + {m} malignant = {b + m}")
    print(f"manifest -> {manifest_path}")
    print(f"train images -> {train_root}\ntest images  -> {test_root}")


if __name__ == "__main__":
    main()
