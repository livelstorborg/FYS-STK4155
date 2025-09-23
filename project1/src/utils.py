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
    return np.linalg.pinv(X.T @ X + lam * np.eye(n)) @ X.T @ y


def OLS_parameters(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def OLS_gradient(X, y, theta):
    n = X.shape[0]
    gradient = -(2 / n) * (y - X @ theta) @ X
    return gradient


def Ridge_gradient(X, y, theta, lam):
    n = X.shape[0]
    gradient = (2 / n) * (X.T @ X @ theta - X.T @ y + n * lam * theta)
    return gradient


# Gradient descent for OLS with fixed learning rate
def gd_OLS(X, y, eta, num_iters, stopping_param):
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

        if np.linalg.norm(step) < stopping_param:
            break

    return theta_OLS, mse_history


# Gradient descent for Ridge with fixed learning rate
def gd_Ridge(X, y, eta, lam, num_iters, stopping_param):
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

        if np.linalg.norm(step) < stopping_param:
            break

    return theta_Ridge, mse_history


def gd_OLS_momentum(X, y, eta=1e-2, num_iters=10_000, beta=0.9, tol=1e-10):
    """
    Gradient Descent with momentum for OLS.
    Returns: theta, mse_history
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    v = np.zeros_like(theta)  # velocity
    mse_history = []

    for t in range(num_iters):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient + momentum step
        grad = OLS_gradient(X, y, theta)
        v = beta * v + grad
        theta = theta - eta * v

        # early stop (by grad norm)
        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_Ridge_momentum(X, y, eta=1e-2, num_iters=10_000, lam=1e-2, beta=0.9, tol=1e-10):
    """
    Gradient Descent with (classical) momentum for Ridge.
    Returns: theta, mse_history
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    v = np.zeros_like(theta)  # velocity
    mse_history = []

    for t in range(num_iters):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient + momentum step
        grad = Ridge_gradient(X, y, theta, lam)
        v = beta * v + grad
        theta = theta - eta * v

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_OLS_adagrad(X, y, eta=1e-2, num_iters=10_000, eps=1e-8, tol=1e-10):
    """
    Adagrad for OLS.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    G = np.zeros(d)
    mse_history = []

    for _ in range(num_iters):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # gradient: (2/n) X^T (Xθ - y)
        grad = (2.0 / n) * (X.T @ r)

        # Adagrad scaling
        G += grad**2
        scaled_grad = grad / (np.sqrt(G) + eps)

        theta -= eta * scaled_grad

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_Ridge_adagrad(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    lam=1e-2,
    eps=1e-8,
    tol=1e-10,
    regularize_bias=False,
):
    """
    Adagrad for Ridge.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    G = np.zeros(d)
    mse_history = []

    for _ in range(num_iters):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # ridge gradient: (2/n) X^T (Xθ - y) + 2λθ (optionally skip bias)
        grad = (2.0 / n) * (X.T @ r) + 2.0 * lam * theta
        if not regularize_bias:
            grad = grad.copy()
            grad[0] -= 2.0 * lam * theta[0]

        G += grad**2
        scaled_grad = grad / (np.sqrt(G) + eps)

        theta -= eta * scaled_grad

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_OLS_rmsprop(X, y, eta=1e-2, num_iters=10_000, beta=0.9, eps=1e-8, tol=1e-10):
    """
    RMSProp for OLS.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    S = np.zeros(d)  # EMA of squared gradients
    mse_history = []

    for t in range(num_iters):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # gradient: (2/n) X^T (Xθ - y)
        grad = (2.0 / n) * (X.T @ r)

        # RMSProp accumulator and update
        S = beta * S + (1.0 - beta) * (grad**2)
        theta -= eta * grad / (np.sqrt(S) + eps)

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_Ridge_rmsprop(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    lam=1e-2,
    beta=0.9,
    eps=1e-8,
    tol=1e-10,
    regularize_bias=False,
):
    """
    RMSProp for Ridge.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    S = np.zeros(d)  # EMA of squared gradients
    mse_history = []

    for t in range(num_iters):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # ridge gradient: (2/n) X^T (Xθ - y) + 2λθ (optionally skip bias)
        grad = (2.0 / n) * (X.T @ r) + 2.0 * lam * theta
        if not regularize_bias:
            grad = grad.copy()
            grad[0] -= 2.0 * lam * theta[0]

        S = beta * S + (1.0 - beta) * (grad**2)
        theta -= eta * grad / (np.sqrt(S) + eps)

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_OLS_adam(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    tol=1e-10,
    amsgrad=False,
):
    """
    Adam for OLS.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)  # first moment
    v = np.zeros(d)  # second moment
    v_max = np.zeros(d)  # for AMSGrad
    mse_history = []

    for t in range(1, num_iters + 1):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # grad: (2/n) X^T (Xθ - y)
        grad = (2.0 / n) * (X.T @ r)

        # moments
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)

        # bias correction
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        if amsgrad:
            v_max = np.maximum(v_max, v_hat)
            denom = np.sqrt(v_max) + eps
        else:
            denom = np.sqrt(v_hat) + eps

        theta -= eta * (m_hat / denom)

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_Ridge_adam(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    lam=1e-2,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    tol=1e-10,
    regularize_bias=False,
    amsgrad=False,
):
    """
    Adam for Ridge.
    Returns: theta, mse_history
    """
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    v_max = np.zeros(d)
    mse_history = []

    for t in range(1, num_iters + 1):
        r = X @ theta - y
        mse_history.append(np.mean(r**2))

        # ridge grad: (2/n) X^T (Xθ - y) + 2λθ  (optionally skip bias)
        grad = (2.0 / n) * (X.T @ r) + 2.0 * lam * theta
        if not regularize_bias:
            grad = grad.copy()
            grad[0] -= 2.0 * lam * theta[0]

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        if amsgrad:
            v_max = np.maximum(v_max, v_hat)
            denom = np.sqrt(v_max) + eps
        else:
            denom = np.sqrt(v_hat) + eps

        theta -= eta * (m_hat / denom)

        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history
