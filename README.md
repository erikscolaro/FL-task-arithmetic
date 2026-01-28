# FL-task-arithmetic: A Flower / PyTorch app

Federated Learning experiments with Task Arithmetic and Sparse Fine-tuning on CIFAR-100, using the Flower framework and DINO vision transformer backbone.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
  - [Baseline Experiments](#1-baseline-fedavg-experiments)
  - [Task Arithmetic (TaLoS)](#2-task-arithmetic-talos-experiments)
  - [Most Sensitive Masking](#3-most-sensitive-masking-experiments)
  - [Random Masking](#4-random-masking-experiments)
  - [Magnitude-based Masking](#5-magnitude-based-masking-experiments)
- [Configuration Files](#configuration-files)
- [Deployment](#deployment)

## Installation

Install the dependencies listed in `pyproject.toml`:

```bash
pip install -e .
```

## Quick Start

Run a basic federated learning simulation with default settings:

```bash
flwr run .
```

This uses the configuration in `pyproject.toml` (100 clients, 10% participation, sparse fine-tuning enabled).

## Running Experiments

The `config/` directory contains pre-configured TOML files for different experimental setups. Each configuration file **overrides** specific parameters from the base `pyproject.toml`.

### General Usage

To run an experiment with a specific configuration:

```bash
flwr run . --run-config config/<configuration-file>.toml
```

---

### 1. Baseline (FedAvg) Experiments

Standard Federated Averaging without sparse fine-tuning, testing different data heterogeneity levels and local epochs.

**Configuration pattern:** `baseline_nc{N}_j{J}.toml`
- `N` = number of classes per client partition (1, 5, 10, 50, 100)
- `J` = local training epochs (4, 8, 16)

**Examples:**

```bash
# Extreme non-IID: 1 class per client, 4 local epochs
flwr run . --run-config config/baseline_nc1_j4.toml

# Moderate non-IID: 10 classes per client, 8 local epochs
flwr run . --run-config config/baseline_nc10_j8.toml

# IID: 100 classes per client, 16 local epochs
flwr run . --run-config config/baseline_nc100_j16.toml
```

**Available configurations:**
- Non-IID levels: `nc1`, `nc5`, `nc10`, `nc50`, `nc100`
- Local epochs: `j4`, `j8`, `j16`

---

### 2. Task Arithmetic (TaLoS) Experiments

Sparse fine-tuning using gradient-based task arithmetic for mask calibration.

**Configuration pattern:** `task_arithmetic_s{S}_r{R}.toml`
- `S` = sparsity percentage (50, 70, 90, 95)
- `R` = number of calibration rounds (1, 2, 4, 8)

**Examples:**

```bash
# 90% sparsity (freeze 90% of params), 2 calibration rounds
flwr run . --run-config config/task_arithmetic_s90_r2.toml

# 95% sparsity (very aggressive), 4 calibration rounds
flwr run . --run-config config/task_arithmetic_s95_r4.toml

# 50% sparsity (moderate), 1 calibration round
flwr run . --run-config config/task_arithmetic_s50_r1.toml
```

**Available configurations:**
- Sparsity levels: `s50`, `s70`, `s90`, `s95`
- Calibration rounds: `r1`, `r2`, `r4`, `r8`

---

### 3. Most Sensitive Masking Experiments

Freeze parameters based on gradient sensitivity (keep most sensitive parameters trainable).

**Configuration pattern:** `most_sens_s{S}_r{R}.toml`
- `S` = sparsity percentage (50, 70, 90)
- `R` = number of calibration rounds (1, 2, 4)

**Examples:**

```bash
# 70% sparsity with sensitivity-based masking, 2 rounds
flwr run . --run-config config/most_sens_s70_r2.toml

# 90% sparsity with sensitivity-based masking, 1 round
flwr run . --run-config config/most_sens_s90_r1.toml
```

---

### 4. Random Masking Experiments

Baseline sparse fine-tuning with random parameter selection.

**Configuration pattern:** `random_s{S}.toml`
- `S` = sparsity percentage (50, 70, 90)

**Examples:**

```bash
# Random masking with 70% sparsity
flwr run . --run-config config/random_s70.toml

# Random masking with 90% sparsity
flwr run . --run-config config/random_s90.toml
```

---

### 5. Magnitude-based Masking Experiments

Freeze parameters based on their magnitude values.

**High magnitude masking** (freeze high magnitude params):
```bash
flwr run . --run-config config/high_mag_s70.toml
```

**Low magnitude masking** (freeze low magnitude params):
```bash
flwr run . --run-config config/low_mag_s70.toml
```

---

## Configuration Files

Each configuration file overrides specific parameters from `pyproject.toml`. Key parameters include:

- `num-classes-per-partition`: Data heterogeneity (1 = extreme non-IID, 100 = IID)
- `local-epochs`: Local training epochs per round
- `use-sparse-finetuning`: Enable/disable sparse fine-tuning
- `sparsity-ratio`: Fraction of parameters to freeze (0.0-1.0)
- `num-calibration-rounds`: Rounds for mask calibration
- `mask-calibration-type`: Calibration strategy
  - `0` = Task arithmetic
  - `1` = Most sensitive
  - `2` = Random
  - `3` = Lowest magnitude
  - `4` = Highest magnitude
- `group`: Weights & Biases experiment group name

You can create custom configurations by copying and modifying existing files in `config/`.
