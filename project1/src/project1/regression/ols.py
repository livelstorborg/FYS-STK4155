import numpy as np


def OLS_parameters(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Ordinary Least Squares (OLS) regression coefficients.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).

    Returns
    -------
    np.ndarray
        Estimated coefficient vector of shape (n_features,).
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y


def gradient_descent_ols(
    X: np.ndarray, y: np.ndarray, eta: float, n_iter: int
) -> np.ndarray:
    """
    Estimate OLS regression coefficients using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
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
        gradient = (2 / n) * X.T @ (X @ theta - y)
        theta -= eta * gradient
    return theta
