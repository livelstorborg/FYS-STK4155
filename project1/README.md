# Project 1: Regression Analysis of Runge's Function

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug


## Description
This project implements and compares various regression methods for fitting the Runge function
$$
\frac{1}{1 + 25x^2}.
$$
We analyze Ordinary Least Squares (OLS), Ridge regression, and Lasso regression, exploring the bias-variance tradeoff and implementing different gradient descent optimization methods.

## Directory structure (project1)
project1/
├── code/
│   ├── src/                # Core modules
│   │   ├── regression.py   # OLS, Ridge, LASSO implementations
│   │   ├── plotting.py     # Visualization utilities
│   │   └── utils.py        # Helper functions
│   ├── main_ols.py         # Part a: OLS analysis
│   ├── main_ridge.py       # Part b: Ridge regression
│   ├── main_lasso.py       # Part e: LASSO regression
│   ├── main_gd.py          # Part c-d: Gradient descent and optimizers
│   ├── main_stochastic.py  # Part f: SGD
│   ├── main_resampling.py  # Parts g-h: Bootstrap & CV
│   └── results.ipynb       # Notebook for reproducing results
├── cv_data/                # Precomputed data for plotting comparison of Ridge and Lasso vs OLS
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependency versions
└── README.md               # This file



## Run the Code 
### Using uv as package manager
If you don't have uv, install it using homebrew:
```bash
brew install uv
```

Sync the dependencies:
```bash
uv sync
```

Run files:
```bash
uv run main_<name>.py
```

**To run all main files and generate all figures, you can also use the Jupyter Notebook `results.ipynb`.**

## Note: main_resampling.py
This file takes over one hour to run due to calculating around 2 million models for bias-variance comparison. The precomputed data is stored in `cv_results` and used, by default, to plot the bias-variance heatmaps. The main_resampling.py file will still take around one minute to run, due to loading the precomputed results.








---------------------------




# Project 1: Regression Analysis of Runge's Function

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Description
This project implements and compares various regression methods for fitting the Runge function:

$$f(x) = \frac{1}{1 + 25x^2}, \quad x \in [-1, 1]$$

We analyze Ordinary Least Squares (OLS), Ridge regression, and Lasso regression, exploring the bias-variance tradeoff and implementing different gradient descent optimization methods including momentum, AdaGrad, RMSprop, and ADAM.

## Project Structure

```
project1/
├── code/
│   ├── src/                # Core modules
│   │   ├── __init__.py
│   │   ├── regression.py   # OLS, Ridge, LASSO implementations
│   │   ├── plotting.py     # Visualization utilities
│   │   └── utils.py        # Helper functions
│   ├── main_ols.py         # Part a: OLS analysis
│   ├── main_ridge.py       # Part b: Ridge regression
│   ├── main_lasso.py       # Part e: LASSO regression
│   ├── main_gd.py          # Part c-d: Gradient descent and optimizers
│   ├── main_stochastic.py  # Part f: Stochastic gradient descent
│   ├── main_resampling.py  # Parts g-h: Bootstrap & cross-validation
│   └── results.ipynb       # Notebook for reproducing all results
├── cv_data/                # Precomputed data for plotting comparison of Ridge and Lasso vs OLS
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version specification
└── README.md               # This file
```

## Setup and Installation

### Prerequisites
- Python 3.10+ (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project1
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
# Part a: OLS Regression Analysis
uv run code/main_ols.py

# Part b: Ridge Regression
uv run code/main_ridge.py

# Part c-d: Gradient Descent with Various Optimizers
uv run code/main_gd.py

# Part e: LASSO Regression
uv run code/main_lasso.py

# Part f: Stochastic Gradient Descent
uv run code/main_stochastic.py

# Parts g-h: Bias-Variance and Cross-Validation Analysis
uv run code/main_resampling.py
```

### Running the Jupyter Notebook

To reproduce all results and generate all figures:

```bash
uv run jupyter notebook code/results.ipynb
# or
uv run jupyter lab
```

The notebook provides a comprehensive overview of all analyses and allows for interactive exploration of results.

## Key Features

- **Part a**: OLS regression with MSE and R² analysis
- **Part b**: Ridge regression with λ hyperparameter tuning
- **Part c**: Gradient descent implementation with fixed learning rate
- **Part d**: Advanced optimizers (Momentum, AdaGrad, RMSprop, ADAM)
- **Part e**: LASSO regression using gradient descent
- **Part f**: Stochastic gradient descent comparison
- **Parts g-h**: Bootstrap resampling and k-fold cross-validation for bias-variance analysis

## Output

All generated figures and results are saved in the `results/` directory:
- Regression plots
- MSE and R² comparisons
- Bias-variance tradeoff visualizations
- Cross-validation performance metrics

## Important Note on Computational Time

⚠️ **`main_resampling.py` Runtime**: This script calculates approximately 2 million models for comprehensive bias-variance analysis and takes **over 1 hour** to complete.

**Recommended approach:**
- Precomputed results are stored in `cv_results/` and loaded by default for plotting
- Even with precomputed data, the script takes ~1 minute to run due to data loading
- To regenerate the data from scratch, modify the flag in the script (not recommended unless necessary)

## Dependencies

Main packages (managed via `pyproject.toml`):
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning utilities and comparison
- `pandas` - Data manipulation
- `jupyter` - Interactive notebooks

To add new dependencies:
```bash
uv add <package-name>
```

## Course Information

**Course**: FYSSTK3155/FYS4155 - Applied Data Analysis and Machine Learning  
**Institution**: University of Oslo  
**Project Deadline**: October 6, 2025

## License

[Specify your license here, e.g., MIT, GPL, etc.]