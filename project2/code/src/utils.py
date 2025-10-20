import numpy as np


def runge(x):
    """Runge function: f(x) = 1 / (1 + 25x^2)"""
    return 1.0 / (1 + 25 * x**2)


def polynomial_features(x, p, intercept=False):
    """
    Generate polynomial features from input data.
    
    Args:
        x: Input array (1D)
        p: Polynomial degree
        intercept: Whether to include intercept column
    
    Returns:
        X: Design matrix with polynomial features
    """
    n = len(x)
    offset = 1 if intercept else 0
    X = np.ones((n, p + offset))
    for i in range(offset, p + offset):
        X[:, i] = x ** (i + 1 - offset)
    return X


def scale_data(X, y, X_mean=None, X_std=None, y_mean=None):
    """
    Args:
        X: Feature matrix
        y: Target values
        X_mean, X_std, y_mean: Scaling parameters (None for training mode)
    
    Returns:
        X_scaled, y_scaled, X_mean, X_std, y_mean
    """
    if X_mean is None:  # Training mode
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        y_mean = np.mean(y)

    X_scaled = (X - X_mean) / X_std
    y_scaled = y - y_mean

    return X_scaled, y_scaled, X_mean, X_std, y_mean


def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def Ridge_parameters(X, y, lam):
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(n)) @ X.T @ y


def inverse_scale_y(y_scaled, y_mean):
    """
    Inverse scale y values.
    
    Args:
        y_scaled: Scaled y values
        y_mean: Mean used for scaling
    
    Returns:
        Original scale y values
    """
    return y_scaled + y_mean