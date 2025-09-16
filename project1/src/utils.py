import numpy as np
from sklearn.metrics import mean_squared_error




def runge(x):
    """Runge function: f(x) = 1 / (1 + 25x^2)"""
    return 1.0 / (1 + 25 * x**2)



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






def scale_data(X, y):
    """Standardize the feature matrix X and center the target vector y."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    # Avoid division by zero - add small epsilon to std
    X_std = np.where(X_std < 1e-8, 1.0, X_std)
    X_norm = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_centered = y - y_mean

    return X_norm, y_centered, y_mean




def Ridge_parameters(X, y, lam):
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(n)) @ X.T @ y

def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def OLS_gradient(X, y, theta):
    n = X.shape[0]
    gradient = -(2/n) * (y - X @ theta) @ X
    return gradient

def Ridge_gradient(X, y, theta, lam):
    n = X.shape[0]
    gradient = (2/n) * (X.T @ X @ theta - X.T @ y + n * lam * theta)
    return gradient


# Gradient descent for OLS with fixed learning rate 
def gd_OLS(X, y, eta, num_iters):
    n_features = X.shape[1]
    theta_OLS = np.zeros(n_features)
    mse_history = []

    for t in range(num_iters):
        # Calculate prediction and MSE
        y_pred = X @ theta_OLS
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)
        
        # Gradient descent step
        grad_OLS = OLS_gradient(X, y, theta_OLS)
        step = eta * grad_OLS
        theta_OLS = theta_OLS - step
    
    return theta_OLS, mse_history

# Gradient descent for Ridge with fixed learning rate
def gd_Ridge(X, y, eta, num_iters, lam, stopping_param):
    n_features = X.shape[1]
    theta_Ridge = np.zeros(n_features)
    mse_history = []

    for t in range(num_iters):
        # Calculate prediction and MSE
        y_pred = X @ theta_Ridge
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)
        
        # Gradient descent step
        grad_Ridge = Ridge_gradient(X, y, theta_Ridge, lam)
        step = eta * grad_Ridge
        theta_Ridge = theta_Ridge - step
    
    return theta_Ridge, mse_history