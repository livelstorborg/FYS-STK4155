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
- 'code' : contains all code files
  - 'figs' : contains all figures generated, folder is created when running the main files or Jupyter Notebook (if plt.savefig is uncommented)
  - 'src' : 
    - __init__.py: makes src callable as a package
    - plotting.py: plotting functions
    - regression.py: class for regression analysis
    - utils.py: helper functions for RegressionAnalysis class
  - .python-version: specifies python version (needed for uv)
  - main_gd.py
  - main_lasso.py
  - main_ols.py
  - main_resampling.py
  - main_ridge.py
  - main_stochastic.py
  - results.ipynb: Jupyter Notebook to run all main files and generate all figures
  - Makefile: for cleaning the figs folder
  - pyproject.toml: for package management (uv)
  - uv.lock: for package management (uv)



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