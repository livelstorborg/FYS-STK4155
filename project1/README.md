# Project 1: Regression Analysis of Runge's Function


## Directory structure 
- 'figs' : contains all figures generated
- 'src' : 
  - __init__.py: makes src callable as a package
  - analysis.py: functions for analyzing results
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

## Run the code 
### Using uv (recommended)
If you dont have uv, install by using pip:
```pip install uv
```

Then sync the dependencies by running:
```uv sync
```

Finally, you can run files
```uv run main_<name>.py
```

### Using pip
Install the dependencies by running:
```pip install -r requirements.txt
```
Run files using python:
```python3 main_<name>.py
```