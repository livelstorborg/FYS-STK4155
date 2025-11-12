# Project 2: Comparison of Feed Forward Neural Networks with Classical Polynomial Regression Methods

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Description
In this project we compare feed forward neural networks with different hyperparameters, architectures, optimizers and regularization with classical linear regression methods. We analyze the best combinations of the neural network architecture and measure them against how the classical polynomial linear regression approaches perform. We also analyze the performance of the neural networks on the MNIST dataset, to see how well it performs in picture classification.

## Project Structure

```
├── project2
│   ├── code
│   │   ├── notebooks
│   │   │   ├── figs
│   │   │   ├── results
│   │   │   ├── Makefile
│   │   │   └── results.ipynb
│   │   └── src
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── losses.py
│   │       ├── metrics.py
│   │       ├── neural_network.py
│   │       ├── optimizers.py
│   │       ├── plotting.py
│   │       ├── training.py
│   │       ├── utils_b.py
│   │       ├── utils_d.py
│   │       ├── utils_e.py
│   │       ├── utils_f.py
│   │       └── utils.py
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
└── README.md
├── report.pdf

```

## Setup and Installation

### Prerequisites
- Python 3.11 (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project2
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
uv run jupyter notebook code/notebooks/results.ipynb
```
additionally, you can run each cell in the notebook individually to generate figures for each problem. For the convenience of the user, the repository contains the data files and figures generated from running the notebook.

## Key Features

- **Custom FFNN engine** – plug-and-play activations, arbitrary depth/width, BatchNorm toggle, and configurable losses via the shared `src/` modules.
- **Optimizer & LR tooling** – GD, RMSprop, Adam (with optional BN updates) plus learning-rate calibration for any architecture.
- **Per-layer diagnostics** – tracking of gradient norms, dead-neuron fractions to monitor layer health.
- **Unified plotting toolkit** – centralized `plotting.py` renders heatmaps, learning/validation curves, and gradient evolution PDFs (and can display inline).
- **Reproducible workflow** – training artefacts land in `code/notebooks/results/part_*`, figures in `code/notebooks/figs/part_*`, enabling reruns or plots-only regeneration without retraining.


## Output

All generated figures are saved in a folder `figs/`, that is generated automatically when running the cells in the notebook.


## Dependencies

Main packages (managed via `pyproject.toml`):
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning utilities and comparison
- `pandas` - Data manipulation
- `seaborn` - Statistical data visualization

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
