## FL-task-arithmetic

Repository with code and notebooks to reproduce federated learning experiments (Task Arithmetic and sparse fine‑tuning) on CIFAR‑100.

## Installation

Install the package and its dependencies (editable mode):


```python
pip install -e .
```
We recommend to create a virtual environment first.


## Environment

Remember to create a `.env` file in the root of the project containing the key to access WandB for automatic checkpointing management.

```.env
WANDB_API_KEY=XXXXXXX
```


## Running experiments

Two main workflows are provided: centralized training and federated experiments (Flower).

- Centralized baseline:

Launch the script in `notebooks/run_baseline.ipynb` and manually set the configurations for the centralized model training, such as the scheduler, learning rate, etc.

- Federated experiments (Flower): pass a TOML configuration from the `config/` directory:

Start the notebook `run_federated.ipynb` in the `notebooks/` folder. The notebook provides two sections: one for execution on Colab and one for local execution. In either case, you must specify which configuration file to launch. The configuration file defines the experiment.

## Configuration & reproducibility

All experiment parameters are controlled by TOML files in the `config/` directory. Default values and parameter descriptions are available in `pyproject.toml`.

- `config/`: pre-built experiment configs (sparsity, calibration rounds, heterogeneity, etc.).
- `pyproject.toml`: default parameter values and comments. Some of these values can be overrided by the other conf.toml in the config folder.

## Experiment reference (config patterns)

- **FedAvg baseline** — `baseline_nc{N}_j{J}.toml` — `N` = classes per client, `J` = local epochs per round.
- **Least sensitive training** — `task_arithmetic_s{S}_r{R}.toml` — `S` = sparsity (%), `R` = calibration rounds.
- **Most sensitive training** — `most_sens_s{S}_r{R}.toml` — sensitivity‑based masking.
- **Random masking** — `random_s{S}.toml`.
- **Magnitude masking** — `high_mag_s{S}.toml` / `low_mag_s{S}.toml`.