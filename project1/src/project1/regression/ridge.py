import numpy as np


def ridge_parameters(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Compute Ridge regression coefficients using the closed-form solution.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    lambda_ : float
        Regularization parameter controlling the amount of shrinkage.

    Returns
    -------
    np.ndarray
        Estimated coefficient vector of shape (n_features,).
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y


def gradient_descent_ridge(
    X: np.ndarray, y: np.ndarray, lam: float, eta: float, n_iter: int
) -> np.ndarray:
    """
    Estimate Ridge regression coefficients using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    lam : float
        Regularization parameter controlling the amount of shrinkage.
    eta : float
        Learning rate for gradient descent.
    n_iter : int
        Number of iterations to run gradient descent.

    Returns
    -------
    np.ndarray
        Estimated coefficient vector of shape (n_features,).
    """
    n = X.shape[0]
    m = X.shape[1]
    theta = np.zeros(m)
    for _ in range(n_iter):
        gradient = (2 / n) * (X.T @ (X @ theta - y)) + 2 * lam * theta
        theta -= eta * gradient
    return theta
