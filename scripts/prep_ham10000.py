"""
Prepare HAM10000 + ISIC2018 Task-3 test set for the semi-supervised melanoma SGAN.

Produces a binary (melanoma vs non-melanoma) dataset at 64x64 in the directory
layout the existing loaders expect:

    data/ham64/labeled/{benign,malignant}/   small balanced labeled subset
    data/ham64/unlabeled/images/             the rest of HAM10000 (unlabeled)
    data/ham64/test/{benign,malignant}/      official ISIC2018 Task-3 test set

Mapping: dx == 'mel'  -> malignant ;  everything else -> benign.

Run AFTER the raw zips have downloaded into data/ham10000/raw/:
    python -m scripts.prep_ham10000 --labeled-per-class 100 --size 64 --seed 42
"""

import argparse
import random
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/ham10000/raw")
OUT = Path("data/ham64")


def _read_tab(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Strip the quoting Dataverse adds to string cells.
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip('"')
    return df


def _extract_all(zips, dest: Path) -> dict:
    """Extract zips, return {image_id: path_to_jpg}."""
    dest.mkdir(parents=True, exist_ok=True)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    index = {}
    for p in dest.rglob("*.jpg"):
        index[p.stem] = p           # ISIC_0027419.jpg -> 'ISIC_0027419'
    return index


def _save_resized(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-per-class", type=int, default=100)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- labels ---------------------------------------------------------
    meta = _read_tab(RAW / "HAM10000_metadata.tab")
    test_gt = _read_tab(RAW / "ISIC2018_Test_GroundTruth.tab")
    meta["malignant"] = (meta["dx"] == "mel").astype(int)
    test_gt["malignant"] = (test_gt["dx"] == "mel").astype(int)

    # ---- extract images -------------------------------------------------
    print("Extracting HAM10000 images ...")
    ham_idx = _extract_all(
        [RAW / "HAM10000_images_part_1.zip", RAW / "HAM10000_images_part_2.zip"],
        RAW / "ham_extracted",
    )
    print(f"  {len(ham_idx)} HAM images")
    print("Extracting ISIC2018 test images ...")
    test_idx = _extract_all([RAW / "ISIC2018_Test_Images.zip"], RAW / "test_extracted")
    print(f"  {len(test_idx)} test images")

    # ---- choose labeled subset (balanced) -------------------------------
    mel_ids = [i for i in meta[meta.malignant == 1].image_id if i in ham_idx]
    ben_ids = [i for i in meta[meta.malignant == 0].image_id if i in ham_idx]
    rng.shuffle(mel_ids); rng.shuffle(ben_ids)
    k = args.labeled_per_class
    lab_mel, lab_ben = mel_ids[:k], ben_ids[:k]
    labeled_set = set(lab_mel) | set(lab_ben)

    # ---- write labeled --------------------------------------------------
    print(f"Writing labeled ({k}/class) ...")
    for i, mid in enumerate(lab_mel):
        _save_resized(ham_idx[mid], OUT / "labeled/malignant" / f"{i}_1.jpg", args.size)
    for i, bid in enumerate(lab_ben):
        _save_resized(ham_idx[bid], OUT / "labeled/benign" / f"{i}_0.jpg", args.size)

    # ---- write unlabeled (everything else in HAM) -----------------------
    print("Writing unlabeled ...")
    n = 0
    for img_id, src in ham_idx.items():
        if img_id in labeled_set:
            continue
        _save_resized(src, OUT / "unlabeled/images" / f"{n}.jpg", args.size)
        n += 1

    # ---- write official test --------------------------------------------
    print("Writing test ...")
    tn = {"benign": 0, "malignant": 0}
    for _, row in test_gt.iterrows():
        img_id = row.image_id
        if img_id not in test_idx:
            continue
        cls = "malignant" if row.malignant == 1 else "benign"
        _save_resized(test_idx[img_id], OUT / "test" / cls / f"{tn[cls]}.jpg", args.size)
        tn[cls] += 1

    print("\nDone.")
    print(f"  labeled : {k} malignant + {k} benign")
    print(f"  unlabeled: {n}")
    print(f"  test    : {tn['malignant']} malignant + {tn['benign']} benign")


if __name__ == "__main__":
    main()
