import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import (
    mse_degree_ols,
    r2_degree_ols,
    theta_evolution_ols,
    mse_degree_multiple,
)
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


sample_sizes = [500, 1000, 10000]
all_results = {}

for N in sample_sizes:
    x = np.linspace(-1, 1, N)
    y_true = runge(x)
    degrees = range(1, 16)

    # Split data once for all degrees to ensure consistency
    data_splits = {}  # Dictionary to store splits for each degree
    for deg in degrees:
        X = polynomial_features(x, deg)
        X_norm, y_centered, y_mean = scale_data(X, y_true)
        X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
            X_norm, y_centered, x, test_size=0.2, random_state=42
        )

        data_splits[deg] = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]

    # Create instances for all degrees
    results_current = {
        "degrees": list(degrees),
        "train_mse": [],
        "test_mse": [],
        "train_r2": [],
        "test_r2": [],
        "instances": [],
    }

    theta1_evolution = []  # Initialize for all N, but only use for N=1000

    for deg in degrees:
        analysis = RegressionAnalysis(
            data_splits[deg], degree=deg, lam=None, eta=None, num_iters=None
        )

        analysis.fit_analytical()
        analysis.predict()
        analysis.calculate_metrics()

        results_current["train_mse"].append(analysis.train_mse_ols_analytical)
        results_current["test_mse"].append(analysis.mse_ols_analytical)
        results_current["train_r2"].append(analysis.train_r2_ols_analytical)
        results_current["test_r2"].append(analysis.r2_ols_analytical)
        results_current["instances"].append(analysis)

        # Store theta1 evolution for all N (only plot for N=1000)
        theta1 = analysis.theta_ols_analytical[0]
        theta1_evolution.append(theta1)

    all_results[N] = results_current

    # Further analysis for N=1000
    if N == 1000:
        mse_degree_ols(all_results, sample_size=1000)  # MSE for N=1000 (a)
        r2_degree_ols(results_current, sample_size=1000)  # R2 for N=1000 (a)
        theta_evolution_ols(
            degrees, theta1_evolution, sample_size=1000
        )  # Evolution of theta1 for N=1000 (a)

# Plot for multiple sample sizes
mse_degree_multiple(all_results, sample_sizes)
