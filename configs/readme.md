# Configuration Files

This directory contains experiment configuration files used by the project.

All hyperparameters and experiment settings are defined in `config.yaml`.
Keeping configuration separate from code ensures reproducibility and
simplifies experimentation.

## Why Configuration Files?

Separating configuration from code allows researchers to:

- modify hyperparameters without changing source code
- reproduce experiments reliably
- run multiple experiments with different configurations

## Key Configuration Sections

### experiment

General experiment information.

- experiment name
- random seed

### dataset

Dataset settings including:

- dataset paths
- image resolution
- number of classes

### generator

Hyperparameters for the generator network.

### discriminator

Hyperparameters for the discriminator network.

### training

Training hyperparameters including:

- batch size
- number of epochs
- learning rates
- optimizer parameters

### output

Defines where experiment outputs are stored:

- model checkpoints
- generated samples
- logs
