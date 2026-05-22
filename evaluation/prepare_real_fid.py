import os
import shutil
import random

random.seed(42)

src_dirs = [
    "data/labeled/benign",
    "data/labeled/malignant",
    "data/unlabeled/images"
]

dst = "evaluation/real_samples"

os.makedirs(dst, exist_ok=True)

# clear folder
for f in os.listdir(dst):
    os.remove(os.path.join(dst, f))

all_files = []

for d in src_dirs:

    files = os.listdir(d)

    for f in files:

        all_files.append(os.path.join(d, f))

print(f"Total dataset images available: {len(all_files)}")

sample_size = min(1000, len(all_files))

sample = random.sample(all_files, sample_size)

for i, file_path in enumerate(sample):

    ext = os.path.splitext(file_path)[1]

    shutil.copy(
        file_path,
        os.path.join(dst, f"real_{i}{ext}")
    )

print(f"Prepared {sample_size} real images for FID.")