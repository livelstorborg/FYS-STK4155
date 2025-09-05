import numpy as np


def mse(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) for a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features).
    y : np.ndarray
        True target values of shape (n_samples,).
    theta : np.ndarray
        Coefficient vector of shape (n_features,).

    Returns
    -------
    float
        The mean squared error between predictions and true values.
    """
    return np.mean((X @ theta - y) ** 2)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R^2) regression score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        The R^2 score, which indicates the proportion of the variance
        in the dependent variable that is predictable from the independent variables.
        R^2 = 1.0 means perfect prediction, and can be negative if the model
        performs worse than a constant baseline.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
