import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import solution_comparison, solution_comparison_gd
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge

N = 100  # sample size
degree = 8  # polynomial degree
lam = 1e-2  # Ridge lambda
eta = 1e-2  # learning rate for GD
num_iters = 10000

x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x) + random_noise

X = polynomial_features(x, degree)
X_norm, y_centered, y_mean = scale_data(X, y_true)

# Use full dataset mode
analysis = RegressionAnalysis(
    [X_norm, y_centered], 
    degree=degree, 
    lam=lam, 
    eta=eta, 
    num_iters=num_iters,
    full_dataset=True
)

# Fit all methods
analysis.fit_analytical()
analysis.fit_gradient_descent()
analysis.fit_momentum()
analysis.fit_adagrad()
analysis.fit_rmsprop()
analysis.fit_adam()
analysis.predict()

# Use the class properties directly - no manual calculations needed!
solutions_ols = [
    analysis.y_pred_ols_analytical,
    analysis.y_pred_ols_gd,
    analysis.y_pred_ols_momentum,
    analysis.y_pred_ols_adagrad,
    analysis.y_pred_ols_rmsprop,
    analysis.y_pred_ols_adam,
    y_true,
]

solutions_ridge = [
    analysis.y_pred_ridge_analytical,
    analysis.y_pred_ridge_gd,
    analysis.y_pred_ridge_momentum,
    analysis.y_pred_ridge_adagrad,
    analysis.y_pred_ridge_rmsprop,
    analysis.y_pred_ridge_adam,
    y_true,
]

solution_comparison_gd(x, solutions=solutions_ols, sample_size=N, degree=degree, lam=lam)
solution_comparison_gd(x, solutions=solutions_ridge, sample_size=N, degree=degree, lam=lam)