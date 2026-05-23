# Testing Guide

Comprehensive testing suite for the Melanoma SGAN project using pytest.

## Installation

Tests are included with the project dependencies:

```bash
pip install -r requirements.txt
```

Key testing dependencies:
- `pytest==7.4.0` - Testing framework
- `pytest-cov==4.1.0` - Coverage reporting

## Running Tests

### Run all tests

```bash
pytest
```

### Run with verbose output

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/test_losses.py
```

### Run specific test class

```bash
pytest tests/test_config.py::TestSGANConfig
```

### Run specific test function

```bash
pytest tests/test_losses.py::TestLosses::test_supervised_loss_shape
```

### Run with coverage report

```bash
pytest --cov=. --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run only unit tests

```bash
pytest -m unit
```

### Stop after first failure

```bash
pytest -x
```

### Run with detailed output on failures

```bash
pytest -vv --tb=long
```

## Test Organization

```
tests/
├── test_losses.py           # Loss function tests
├── test_config.py           # Configuration validation tests
├── test_datasets.py         # Dataset loading and error handling tests
├── test_lr_scheduler.py     # Learning rate scheduler tests
└── __init__.py
```

## Test Suites

### 1. Loss Functions (test_losses.py)

Tests for all loss functions used in training:

```bash
pytest tests/test_losses.py -v
```

**Coverage:**
- Supervised loss (classification loss)
- Unlabeled real loss (semi-supervised)
- Fake loss (discriminator training)
- Feature matching loss (generator training)
- Gradient computation
- Output shapes and value ranges

**Key tests:**
- ✓ Loss returns scalar values
- ✓ Loss values in reasonable ranges
- ✓ Feature matching loss is zero when features are equal
- ✓ Gradients are computed correctly

### 2. Configuration (test_config.py)

Tests for configuration loading, validation, and persistence:

```bash
pytest tests/test_config.py -v
```

**Coverage:**
- YAML file parsing
- Configuration validation with Pydantic
- Field bounds checking (image size, learning rates, etc.)
- Scheduler type validation
- Missing field detection
- Config persistence (save/load)

**Key tests:**
- ✓ Valid configs load successfully
- ✓ Invalid values raise errors
- ✓ Missing required fields are detected
- ✓ All valid scheduler types accepted
- ✓ Config roundtrip (save and reload)

### 3. Datasets (test_datasets.py)

Tests for dataset loading with error handling:

```bash
pytest tests/test_datasets.py -v
```

**Coverage:**
- Dataset initialization
- Image loading and preprocessing
- Directory validation
- File filtering (image formats)
- Error handling (corrupted images, missing directories)
- Data transformations

**Key tests:**
- ✓ Loads valid datasets correctly
- ✓ Handles missing directories gracefully
- ✓ Filters non-image files
- ✓ Supports multiple image formats (JPG, PNG, BMP)
- ✓ Raises informative errors
- ✓ __getitem__ returns correct tensor shapes

### 4. Learning Rate Scheduler (test_lr_scheduler.py)

Tests for learning rate scheduling strategies:

```bash
pytest tests/test_lr_scheduler.py -v
```

**Coverage:**
- All scheduler types (none, step, exponential, cosine)
- LR decay mechanics
- State dict save/load
- Invalid scheduler detection

**Key tests:**
- ✓ No scheduler keeps constant LR
- ✓ Step scheduler decays LR at intervals
- ✓ Exponential scheduler decays smoothly
- ✓ Cosine scheduler follows cosine annealing
- ✓ Invalid scheduler type raises error
- ✓ State checkpoint/restore works correctly

## Sanity Checks

Run comprehensive sanity checks before training:

```bash
python sanitytest.py
```

**Checks:**
- ✓ All dependencies available
- ✓ Configuration file valid
- ✓ Dataset directories exist and contain images
- ✓ Generator initializes and runs forward pass
- ✓ Discriminator initializes and runs forward pass
- ✓ SGAN model works end-to-end
- ✓ Compute device (CPU/CUDA) available

## Continuous Integration

Run all quality checks:

```bash
# Run tests with coverage
pytest --cov=. --cov-report=term-missing

# Check code style (if using pylint/flake8)
pylint **/*.py

# Type checking (if using mypy)
mypy . --ignore-missing-imports
```

## Test Coverage Goals

- **Losses**: 95%+ coverage
- **Config validation**: 90%+ coverage
- **Datasets**: 85%+ coverage (due to external dependencies)
- **Scheduler**: 90%+ coverage
- **Overall**: 85%+ target

## Common Issues

### "ModuleNotFoundError" when running tests

Ensure you're in the project root directory:

```bash
cd /path/to/melanoma-sgan
pytest
```

### CUDA errors during tests

Tests run on both CPU and CUDA. If you see CUDA errors:

```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" pytest
```

### Slow test execution

Run tests in parallel:

```bash
pytest -n auto  # Requires pytest-xdist
```

## Adding New Tests

To add tests for new functionality:

1. Create test file: `tests/test_<module_name>.py`
2. Follow naming conventions:
   - Test classes: `Test<ClassName>`
   - Test functions: `test_<functionality>`
3. Use fixtures for reusable setup
4. Include docstrings explaining what's tested
5. Run: `pytest tests/test_<module_name>.py -v`

Example:

```python
"""
Tests for my new module.
"""

import pytest
from my_module import my_function

class TestMyFunction:
    """Test my_function."""

    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {"input": 42, "expected": 84}

    def test_basic_case(self, setup_data):
        """Test basic functionality."""
        result = my_function(setup_data["input"])
        assert result == setup_data["expected"]

    def test_edge_case(self):
        """Test edge cases."""
        with pytest.raises(ValueError):
            my_function(-1)
```

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/how-to-use-hooks.html)
