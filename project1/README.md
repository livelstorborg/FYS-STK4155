# Project 1: Regression Analysis of Runge's Function

## Authors
- Live L. Storborg
- Adam Falchenberg
- Simon S. Thommesen
- Henrik Haug

## Directory structure 
- 'figs' : contains all figures generated
- 'src' : 
  - __init__.py: makes src callable as a package
  - plotting.py: plotting functions
  - regression.py: class for regression analysis
  - utils.py: helper functions for RegressionAnalysis class
- main_ols.py
- main_ridge.py
- main_gd.py
- main_gd_comparison.py
- Makefile : for running the main files
- pyproject.toml: for package management (using uv)
- uv.lock: for package management (using uv)
- requirements.txt: for package management



## Run the Code 
### Using uv as package manager (recommended)
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

### Using pip
Install dependencies:
```bash
pip install -r requirements.txt
```

Run files using Python:
```bash
python3 main_<name>.py
```

## Description
This project implements and compares various regression methods for fitting the Runge function f(x) = 1/(1 + 25x²). We analyze Ordinary Least Squares (OLS), Ridge regression, and Lasso regression, exploring the bias-variance tradeoff and implementing different gradient descent optimization methods.