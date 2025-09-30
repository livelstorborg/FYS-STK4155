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


sample_sizes = [50, 100, 150, 200, 300, 400, 500, 1000, 10000]
all_results = {}

for N in sample_sizes:

    x = np.linspace(-1, 1, N)
    np.random.seed(42)
    random_noise = np.random.normal(0, 0.1, N)
    y_true = runge(x)
    y_noise = y_true + random_noise
    degrees = range(1, 50)

    # Split data once for all degrees to ensure consistency
    data_splits = {}  # Dictionary to store splits for each degree
    for deg in degrees:
        X = polynomial_features(x, deg)
        X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.3, random_state=42)
        x_train = X_train[:, 0] 
        x_test = X_test[:, 0] 

        # Scaling the training data and using the same parameters to scale the test data (to avoid data leakage)
        X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
        X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)

        data_splits[deg] = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]

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
            data_splits[deg], degree=deg, lam=0.0, eta=None, num_iters=None
        )

        analysis.fit(models='ols', opts='analytical')

        results_current["train_mse"].append(analysis.get_metric('ols', 'analytical', 'train_mse'))
        results_current["test_mse"].append(analysis.get_metric('ols', 'analytical', 'test_mse'))
        results_current["train_r2"].append(analysis.get_metric('ols', 'analytical', 'train_r2'))
        results_current["test_r2"].append(analysis.get_metric('ols', 'analytical', 'test_r2'))
        results_current["instances"].append(analysis)

        # Store theta1 evolution for all N (only plot for N=1000)
        theta1 = analysis.get_theta('ols', 'analytical')[0]
        theta1_evolution.append(theta1)

    all_results[N] = results_current



    # Further analysis for N=1000
    if N == 300:
        mse_degree_ols(all_results, sample_size=N)  # MSE for N=1000 (a)
        r2_degree_ols(results_current, sample_size=N)  # R2 for N=1000 (a)
        theta_evolution_ols(
            degrees, theta1_evolution, sample_size=N
        )  # Evolution of theta1 for N=1000 (a)




mse_degree_multiple(all_results, sample_sizes)
