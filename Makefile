# =========================================
# Melanoma SGAN Makefile
# =========================================

PYTHON=python
CONFIG=configs/config.yaml
CHECKPOINT=outputs/checkpoints/best_generator.pt

.PHONY: help setup install train generate tensorboard clean test sanity lint coverage

# -----------------------------------------
# Help
# -----------------------------------------

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        -> Full project setup (dataset + install)"
	@echo "  make install      -> Install Python dependencies"
	@echo "  make sanity       -> Run sanity checks"
	@echo ""
	@echo "Training & Generation:"
	@echo "  make train        -> Start SGAN training"
	@echo "  make generate     -> Generate synthetic images"
	@echo "  make tensorboard  -> Launch TensorBoard"
	@echo ""
	@echo "Testing & Validation:"
	@echo "  make test         -> Run all tests"
	@echo "  make test-v       -> Run tests with verbose output"
	@echo "  make coverage     -> Generate coverage report"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        -> Remove generated files"
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
# Sanity checks
# -----------------------------------------

sanity:
	$(PYTHON) sanitytest.py

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
# Testing
# -----------------------------------------

test:
	pytest

test-v:
	pytest -v

test-unit:
	pytest tests/ -v

# -----------------------------------------
# Coverage
# -----------------------------------------

coverage:
	pytest --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# -----------------------------------------
# Code quality
# -----------------------------------------

lint:
	@echo "Running code quality checks..."
	@echo "Note: Install pylint with: pip install pylint"

# -----------------------------------------
# Clean generated outputs
# -----------------------------------------

clean:
	rm -rf outputs/samples/*
	rm -rf outputs/logs/*
	rm -rf outputs/checkpoints/*
	rm -rf data/generated/*
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
