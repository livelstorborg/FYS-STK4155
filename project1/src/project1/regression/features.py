import numpy as np


def polynomial_features(x: np.ndarray, p: int, intercept: bool = False) -> np.ndarray:
    """
    Generate polynomial features from input data.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n_samples,). Typically a 1D vector of input values.
    p : int
        The degree of the polynomial features.
    intercept : bool, optional
        If True, includes a column of ones as the intercept (bias term).
        If False, only polynomial terms are included. Default is False.

    Returns
    -------
    np.ndarray
        Matrix of polynomial features with shape (n_samples, p) if intercept=False,
        or (n_samples, p+1) if intercept=True.
    """
    n = len(x)

    offset = 1 if intercept else 0
    X = np.ones((n, p + offset))

    for i in range(offset, p + offset):
        X[:, i] = x ** (i + 1 - offset)

    return X
