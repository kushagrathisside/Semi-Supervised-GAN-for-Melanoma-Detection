# =========================================
# Melanoma SGAN Makefile
# =========================================

PYTHON=python
CONFIG=configs/config.yaml
CHECKPOINT=outputs/checkpoints/model_100.pt

.PHONY: help setup install train generate tensorboard clean

# -----------------------------------------
# Help
# -----------------------------------------

help:
	@echo "Available commands:"
	@echo ""
	@echo "make setup        -> Full project setup (dataset + install)"
	@echo "make install      -> Install Python dependencies"
	@echo "make train        -> Start SGAN training"
	@echo "make generate     -> Generate synthetic images"
	@echo "make tensorboard  -> Launch TensorBoard"
	@echo "make clean        -> Remove generated files"
	@echo ""

# -----------------------------------------
# Setup
# -----------------------------------------

setup:
	chmod +x setup.sh
	./setup.sh

# -----------------------------------------
# Install dependencies
# -----------------------------------------

install:
	pip install -r requirements.txt

# -----------------------------------------
# Train SGAN
# -----------------------------------------

train:
	$(PYTHON) main.py --mode train --config $(CONFIG)

# -----------------------------------------
# Generate images
# -----------------------------------------

generate:
	$(PYTHON) main.py --mode generate --checkpoint $(CHECKPOINT) --num_images 2000

# -----------------------------------------
# Tensorboard
# -----------------------------------------

tensorboard:
	tensorboard --logdir outputs/logs

# -----------------------------------------
# Clean generated outputs
# -----------------------------------------

clean:
	rm -rf outputs/samples/*
	rm -rf outputs/logs/*
	rm -rf outputs/checkpoints/*
	rm -rf data/generated/*
