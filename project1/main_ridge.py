import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.plotting import *
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge
from src.analysis import analyze_dependence_lambda_datapoints


sample_sizes = [500, 1000, 10000]
lambda_values = np.logspace(-5, 3, 8)
all_results = {}

for N in sample_sizes:
    x = np.linspace(-1, 1, N)
    y_true = runge(x) + np.random.normal(0, 0.1, size=N)
    degrees = range(1, 16)

    # Split data once for all degrees to ensure consistency
    data_splits = {}
    for deg in degrees:
        X = polynomial_features(x, deg)
        X_norm, y_centered, y_mean = scale_data(X, y_true)
        X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
            X_norm, y_centered, x, test_size=0.2
        )

        data_splits[deg] = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]

    results_by_lambda = {}

    for lam in lambda_values:
        results_current = {
            "degrees": list(degrees),
            "train_mse": [],
            "test_mse": [],
            "train_r2": [],
            "test_r2": [],
            "instances": [],
        }

        for deg in degrees:
            analysis = RegressionAnalysis(
                data_splits[deg], degree=deg, lam=lam, eta=None, num_iters=None
            )

            analysis.fit_one('ridge', 'analytical')
            analysis.fit_one('ols', 'analytical')  # Also fit OLS for comparison in plots

            results_current["train_mse"].append(analysis.get_metric('ridge', 'analytical', 'train_mse'))
            results_current["test_mse"].append(analysis.get_metric('ridge', 'analytical', 'test_mse'))
            results_current["train_r2"].append(analysis.get_metric('ridge', 'analytical', 'train_r2'))
            results_current["test_r2"].append(analysis.get_metric('ridge', 'analytical', 'test_r2'))
            results_current["instances"].append(analysis)

        results_by_lambda[lam] = results_current

    all_results[N] = results_by_lambda

    theta_evolution_lambdas(
        all_results, lambda_values, degrees, sample_size=N, include_ols=True
    )


for degree in degrees:  # Only key degrees to reduce output
    df = analyze_dependence_lambda_datapoints(
        all_results, lambda_values, sample_sizes, degree
    )


for N in sample_sizes:
    mse_degree_lambdas(all_results, lambda_values, sample_size=N)
    r2_degree_lambdas(all_results, lambda_values, sample_size=N)
