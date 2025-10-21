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

def lasso_gd(X, y, lam=0.01, eta=0.01, max_iter=1000, tol=1e-6):
    """
    Lasso regression using Gradient Descent.
    
    Minimizes: Loss = ||y - Xθ||² + λ||θ||₁
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target values (n_samples,)
        lam: L1 regularization strength (lambda)
        eta: Learning rate
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        theta: Learned parameters
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    for iteration in range(max_iter):
        # Predictions
        y_pred = X @ theta
        
        # Gradient of MSE loss: -(2/n) * X^T * (y - y_pred)
        residual = y - y_pred
        grad_mse = -(2.0 / n_samples) * (X.T @ residual)
        
        # Gradient of L1 penalty: λ * sign(θ)
        grad_l1 = lam * np.sign(theta)
        
        # Total gradient
        grad = grad_mse + grad_l1
        
        # Update parameters
        theta_new = theta - eta * grad
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Converged at iteration {iteration}")
            break
        
        theta = theta_new
    
    return theta


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