import os
import random
import yaml
from PIL import Image

random.seed(42)

with open("configs/config.yaml") as f:
    image_size = yaml.safe_load(f)["dataset"]["image_size"]

src_dirs = [
    "data/labeled/benign",
    "data/labeled/malignant",
    "data/unlabeled/images"
]

dst = "evaluation/real_samples"

os.makedirs(dst, exist_ok=True)

for f in os.listdir(dst):
    os.remove(os.path.join(dst, f))

all_files = []

for d in src_dirs:
    for f in os.listdir(d):
        all_files.append(os.path.join(d, f))

print(f"Total dataset images available: {len(all_files)}")

sample_size = min(1000, len(all_files))
sample = random.sample(all_files, sample_size)

for i, file_path in enumerate(sample):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    img.save(os.path.join(dst, f"real_{i}.png"))

print(f"Prepared {sample_size} real images at {image_size}x{image_size} for FID.")
