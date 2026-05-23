# Contributing to Semi-Supervised GAN for Melanoma Detection

First off, thank you for considering contributing to this project! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Provide a step-by-step reproduction** of the problem
* **Provide specific examples** to demonstrate those steps
* **Describe the behavior you observed** and point out what exactly is the problem with that behavior
* **Explain which behavior you expected** to see instead and why
* **Include screenshots and animated GIFs** if possible

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description** of the suggested enhancement
* **Provide specific examples** to demonstrate the steps
* **Describe the current behavior** and **expected enhancement behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Python style guide
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip or conda

### Setup Instructions

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Semi-Supervised-GAN-for-Melanoma-Detection.git
   cd Semi-Supervised-GAN-for-Melanoma-Detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Run sanity checks**
   ```bash
   make sanity
   ```

6. **Run tests**
   ```bash
   make test
   ```

## Style Guide

### Python Code

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additional guidelines:

* Use type hints for all function parameters and return values
* Write docstrings for all public functions and classes
* Use meaningful variable names
* Keep functions small and focused (max ~50 lines)
* Use f-strings for string formatting
* Maximum line length: 100 characters

**Example:**
```python
def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions (N, num_classes)
        labels: Ground truth labels (N,)
    
    Returns:
        Accuracy as float in [0, 1]
    """
    correct = (predictions.argmax(dim=1) == labels).sum().item()
    return correct / len(labels)
```

### Documentation

* Use clear, concise language
* Include code examples where appropriate
* Update README.md for significant changes
* Keep docstrings up-to-date

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

**Example:**
```
Add learning rate scheduling to trainer

- Implement StepLR, ExponentialLR, and CosineAnnealingLR
- Add tests for all scheduler types
- Update configuration with scheduler parameters

Fixes #123
```

## Testing

All code should have tests. We use pytest for testing.

### Running Tests

```bash
# Run all tests
make test

# Run with verbose output
make test-v

# Run specific test
pytest tests/test_config.py::TestSGANConfig

# Run with coverage
make coverage
```

### Writing Tests

1. Create test file in `tests/` directory
2. Use descriptive test names: `test_<function>_<scenario>`
3. Use pytest fixtures for setup
4. Include docstrings explaining what's tested

**Example:**
```python
def test_config_validation_rejects_invalid_image_size():
    """Test that invalid image size is rejected during validation."""
    with pytest.raises(ValueError):
        DatasetConfig(image_size=16)  # Too small
```

## Documentation

### README

The README should be updated for:
- New features
- API changes
- Installation instructions
- Configuration changes

### Docstrings

All functions, classes, and modules should have docstrings:

```python
def train_epoch(self, epoch: int) -> Tuple[float, float]:
    """
    Run a single training epoch.
    
    Handles discriminator and generator training steps with
    automatic mixed precision and gradient accumulation.

    Args:
        epoch: Epoch number for logging

    Returns:
        Tuple of (average_discriminator_loss, average_generator_loss)

    Raises:
        RuntimeError: If training step fails
    """
```

## Project Structure

```
melanoma-sgan/
├── main.py                 # Entry point
├── configs/                # Configuration files
├── data/                   # Datasets (gitignored)
├── datasets/               # Dataset loading
├── models/                 # Neural network models
├── training/               # Training pipeline
├── evaluation/             # Evaluation metrics
├── augmentation/           # Image generation
├── utils/                  # Utility functions
├── tests/                  # Test suite
└── outputs/                # Results (gitignored)
```

## Performance Considerations

When making changes:

* Profile code with large datasets
* Use efficient data loading (num_workers, pin_memory)
* Consider memory usage for large models
* Document performance implications

## Release Process

1. Update version in setup.py/setup.cfg
2. Update CHANGELOG.md
3. Create release notes
4. Tag commit: `git tag v1.0.0`
5. Push tag to trigger GitHub Actions

## Additional Notes

### Questions?

* Check existing issues
* Review documentation
* Ask in discussions

### Thank You!

Your contributions are greatly appreciated!
