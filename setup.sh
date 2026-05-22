#!/bin/bash

set -e

echo "======================================"
echo "Melanoma SGAN Project Setup"
echo "======================================"

PROJECT_ROOT=$(pwd)

echo "Project root: $PROJECT_ROOT"

# -------------------------------------
# Create required directories
# -------------------------------------

echo "Creating dataset directories..."

mkdir -p data/labeled/benign
mkdir -p data/labeled/malignant
mkdir -p data/unlabeled/images
mkdir -p data/generated

mkdir -p outputs/checkpoints
mkdir -p outputs/samples
mkdir -p outputs/logs

# -------------------------------------
# Download dataset
# -------------------------------------

DATA_URL="https://lp-prod-resources.s3.amazonaws.com/278/45149/2021-02-19-19-47-43/MelanomaDetection.zip"

if [ ! -f "MelanomaDetection.zip" ]; then
    echo "Downloading dataset..."
    wget $DATA_URL
else
    echo "Dataset zip already exists."
fi

# -------------------------------------
# Extract dataset
# -------------------------------------

if [ ! -d "MelanomaDetection" ]; then
    echo "Extracting dataset..."
    unzip MelanomaDetection.zip
else
    echo "Dataset already extracted."
fi

# -------------------------------------
# Organize labeled dataset
# -------------------------------------

echo "Organizing labeled dataset..."

SRC_LABELED="MelanomaDetection/MelanomaDetection/labeled"

for file in $SRC_LABELED/*.jpg
do
    filename=$(basename "$file")

    label=$(echo $filename | cut -d "_" -f2 | cut -d "." -f1)

    if [ "$label" = "0" ]; then
        mv "$file" data/labeled/benign/
    else
        mv "$file" data/labeled/malignant/
    fi

done

# -------------------------------------
# Move unlabeled dataset
# -------------------------------------

echo "Moving unlabeled images..."

mv MelanomaDetection/MelanomaDetection/unlabeled/* data/unlabeled/images/ || true

# -------------------------------------
# Install Python dependencies
# -------------------------------------

echo "Installing dependencies..."

pip install -r requirements.txt

# -------------------------------------
# Dataset sanity check
# -------------------------------------

echo "Running dataset sanity check..."

python <<EOF
from datasets.melanoma_dataset import LabeledMelanomaDataset
from datasets.melanoma_dataset import UnlabeledMelanomaDataset

l = LabeledMelanomaDataset("data/labeled")
u = UnlabeledMelanomaDataset("data/unlabeled/images")

print("Labeled samples:", len(l))
print("Unlabeled samples:", len(u))

img, label = l[0]
print("Image shape:", img.shape)
print("Label:", label)
EOF

echo "======================================"
echo "Setup completed successfully."
echo "======================================"
echo ""
echo "You can now start training:"
echo ""
echo "python main.py --mode train"
echo ""
