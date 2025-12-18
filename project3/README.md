# Project 3: Physics-Informed Neural Networks for the One-Dimensional Heat Equation

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Description
In this project we implement Physics-Informed Neural Networks (PINNs) with hard boundary conditions to solve the one-dimensional heat equation. We compare the PINN to a finite-difference (FD) reference solution and explore how network architectures and activation functions affect accuracy.

## Project Structure

```
├── code
│   ├── notebooks
│   │   └── main.ipynb
│   └── data               # pre-computed sweep results
│   ├── figs               # generated figures
│   └── src
│       ├── __init__.py
│       ├── experiment.py     # part B/D helpers and sweeps
│       ├── pde.py            # FD solver and analytical solution
│       ├── pinn.py           # PINN model, loss, training, eval
│       └── plotting.py       # plotting utilities
├── FYS-STK4155_project3.pdf # project report
├── pyproject.toml
├── uv.lock
└── README.md

```

## Setup and Installation

### Prerequisites
- Python 3.11 (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
    cd project3
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS/Linux
   brew install uv
   
   # Or using curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or with pip
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```
   This creates a virtual environment in `.venv/` and installs all required packages.

## Usage

### Running the Jupyter Notebook

To reproduce all results and generate all figures in one place:
```bash
uv run jupyter notebook code/notebooks/main.ipynb
```
Additionally, you can run each cell in the notebook individually to generate figures for each problem. For convenience, the repository contains the data files and figures generated from running the notebook.

## Key Features

- **Hard-BC PINN** – trial solution enforces boundary/initial conditions analytically; only the PDE residual is trained.
- **FD reference** – explicit finite-difference scheme for the 1D heat equation for benchmarking.
- **Architecture sweeps** – run grid searches over widths/depths/activations with aggregated error metrics.
- **Unified plotting** – surfaces, error curves, training loss, and sweep heatmaps in `src/plotting.py`.
- **Notebook workflow** – `code/notebooks/main.ipynb` walks through parts B–D and regenerates figures.


## Output

All generated figures are saved in a folder `figs/`, that is generated automatically when running the cells in the notebook.


## Dependencies

Main packages (managed via `pyproject.toml`):
- `jax`, `jaxlib` - Autodiff/array backend
- `flax` - NNX modules for the PINN
- `optax` - Optimizers and schedules
- `numpy`, `pandas` - Numerical computations / data handling
- `matplotlib` - Plotting

Development packages:
- `jupyter` / `notebook` - Interactive notebooks
- `ipython` - Enhanced Python shell
- `ipykernel` - Jupyter kernel for Python

### Adding New Dependencies

To add a new package:
```bash
uv add <package-name>       # Adds to dependencies
uv add --dev <package-name> # Adds to dev dependencies
```

## Course Information

**Course**: FYS-STK3155/FYS-STK4155 - Applied Data Analysis and Machine Learning  
**Institution**: University of Oslo  
