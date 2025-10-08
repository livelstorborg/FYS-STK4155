# Project 2: 

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Description


## Project Structure

```
project1/
├── code/
│   ├── src/                # Core modules
│   │   ├── __init__.py     # Makes src callable as a package
│   │   ├── class.py        # Class
│   │   ├── plotting.py     # Visualization utilities
│   │   └── utils.py        # Helper functions for class and main files
│   ├── main.py             # Part x
│   ├── results.ipynb       # Notebook for reproducing all results
│   └── Makefile            # Makefile for cleaning generated figures and figs/ folder
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version specification
├── README.md               # This file
└── Project2.pdf            # Report

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

### Running Individual Analysis Scripts

```bash
# Part x: 
uv run code/main.py
```



### Running the Jupyter Notebook

To reproduce all results and generate all figures in one place:
```bash
uv run jupyter notebook code/results.ipynb
```
additionally, you can run each cell in the notebook individually to generate figures for each problem.

## Key Features

- **Part x**:
- **Part y**:
- **Part z**:

## Output

All generated figures are saved in a folder `figs/`, that is generated automatically when running the scripts or the notebook.


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

**Course**: FYSSTK3155/FYS4155 - Applied Data Analysis and Machine Learning  
**Institution**: University of Oslo  