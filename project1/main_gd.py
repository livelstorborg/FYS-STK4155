import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import solution_comparison
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

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
    X_norm, y_centered, x, test_size=0.2, random_state=42
)

data = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]


analysis = RegressionAnalysis(
    data, degree=degree, lam=lam, eta=eta, num_iters=num_iters
)
analysis.fit_analytical()
analysis.fit_gradient_descent()
analysis.predict()
analysis.calculate_metrics()


X_full = polynomial_features(x, degree)
X_full_norm, _, _ = scale_data(
    X_full, y_true
)  # normalize consistently with full dataset

y_pred_ols_analytical = X_full_norm @ analysis.theta_ols_analytical + y_mean
y_pred_ols_gd = X_full_norm @ analysis.theta_ols_gd + y_mean
y_pred_ridge_analytical = X_full_norm @ analysis.theta_ridge_analytical + y_mean
y_pred_ridge_gd = X_full_norm @ analysis.theta_ridge_gd + y_mean


solutions = [
    y_pred_ols_analytical,
    y_pred_ols_gd,
    y_pred_ridge_analytical,
    y_pred_ridge_gd,
    y_true,
]

solution_comparison(x, solutions=solutions, sample_size=N, degree=degree, lam=lam)
