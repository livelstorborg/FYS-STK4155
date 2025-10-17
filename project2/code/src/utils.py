import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ========================================
# DATA GENERATION
# ========================================

def runge(x):
    return 1.0 / (1 + 25 * x**2)


def generate_runge_data(n_samples=100, x_range=(-1, 1), noise_level=0.0, seed=None):
    """
    Generate data from the Runge function.
    
    Args:
        n_samples: Number of data points
        x_range: Tuple (min, max) for x values
        noise_level: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
    
    Returns:
        X, y: Input features and targets (as column vectors)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = runge(x)
    
    if noise_level > 0:
        y += np.random.normal(0, noise_level, size=y.shape)
    
    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    return X, y


# ========================================
# DATA PREPROCESSING
# ========================================

def prepare_data(X, y, test_size=0.2, scale=True, seed=None):
    """
    Split and optionally scale data.
    
    Args:
        X: Features
        y: Targets
        test_size: Fraction of data for testing
        scale: Whether to scale features
        seed: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test: Split (and scaled) data
        scaler_X, scaler_y: Scalers (or None if scale=False)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    scaler_X = None
    scaler_y = None
    
    if scale:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def inverse_scale(y, scaler):
    """Inverse transform scaled predictions."""
    if scaler is None:
        return y
    return scaler.inverse_transform(y)


# ========================================
# PROJECT 1 COMPARISON (for Part b)
# ========================================

def polynomial_features(x, p, intercept=False):
    """
    Generate polynomial features from input data.
    Useful for comparing NN with polynomial regression from Project 1.
    
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


def OLS_parameters(X, y):
    """
    Compute OLS parameters analytically.
    Useful for comparing with neural network results.
    
    Args:
        X: Design matrix
        y: Target values
    
    Returns:
        theta: OLS parameters
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y


def Ridge_parameters(X, y, lam):
    """
    Compute Ridge parameters analytically.
    Useful for comparing with neural network + L2 regularization.
    
    Args:
        X: Design matrix
        y: Target values
        lam: Regularization parameter
    
    Returns:
        theta: Ridge parameters
    """
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(n)) @ X.T @ y


def compare_with_project1(X, y, X_test, y_test, polynomial_degree=5):
    """
    Helper function to compare NN results with Project 1 OLS/Ridge.
    
    Args:
        X, y: Training data (original scale)
        X_test, y_test: Test data (original scale)
        polynomial_degree: Degree for polynomial features
    
    Returns:
        dict: Results with OLS and Ridge MSE
    """
    from sklearn.metrics import mean_squared_error
    
    x_train = X.ravel()
    x_test = X_test.ravel()
    
    # Create polynomial features
    X_poly_train = polynomial_features(x_train, polynomial_degree, intercept=True)
    X_poly_test = polynomial_features(x_test, polynomial_degree, intercept=True)
    
    # OLS
    theta_ols = OLS_parameters(X_poly_train, y)
    y_pred_ols = X_poly_test @ theta_ols
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    
    # Ridge (with small lambda)
    theta_ridge = Ridge_parameters(X_poly_train, y, lam=0.01)
    y_pred_ridge = X_poly_test @ theta_ridge
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    
    return {
        'ols_mse': mse_ols,
        'ridge_mse': mse_ridge,
        'polynomial_degree': polynomial_degree
    }