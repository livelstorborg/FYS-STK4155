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


def scale_data(X, y, X_mean=None, X_std=None, y_mean=None):
    if X_mean is None:  # Training mode
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        y_mean = np.mean(y)
    
    X_norm = (X - X_mean) / X_std
    y_centered = y - y_mean
    
    return X_norm, y_centered, X_mean, X_std, y_mean


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


def analytical_solution(X, y, method="ols", lam=None):
    """
    Unified analytical solution function for OLS and Ridge regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str
        Regression method: 'ols' or 'ridge'
    lam : float, optional
        Regularization parameter (required for ridge)

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    """
    if method.lower() not in ["ols", "ridge"]:
        raise ValueError("Analytical solution only available for 'ols' and 'ridge'")

    if method.lower() == "ridge" and lam is None:
        raise ValueError("Ridge regression requires lam parameter")

    if method.lower() == "ols":
        return OLS_parameters(X, y)
    elif method.lower() == "ridge":
        return Ridge_parameters(X, y, lam)


def soft_threshold(x, threshold):
    """
    Soft thresholding operator for Lasso regression.

    This is the proximal operator for the L1 norm:
    soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s)
    threshold : float
        Threshold parameter (λ)

    Returns
    -------
    float or np.ndarray
        Soft-thresholded value(s)
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def gradient_descent(X, y, eta, num_iters, method="ols", lam=None, tol=1e-8):
    """
    Unified gradient descent function for OLS, Ridge, and Lasso regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    eta : float
        Learning rate
    num_iters : int
        Maximum number of iterations
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    tol : float
        Convergence tolerance

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_features = X.shape[1]
    theta = np.zeros(n_features)
    mse_history = []

    for t in range(num_iters):
        # Calculate prediction and MSE
        y_pred = X @ theta
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)

        if method.lower() == "ols":
            # Standard OLS gradient descent
            grad = OLS_gradient(X, y, theta)
            theta_new = theta - eta * grad

        elif method.lower() == "ridge":
            # Ridge gradient descent
            grad = Ridge_gradient(X, y, theta, lam)
            theta_new = theta - eta * grad

        elif method.lower() == "lasso":
            # Lasso proximal gradient descent (ISTA)
            grad_smooth = OLS_gradient(X, y, theta)  # Smooth part gradient
            theta_temp = theta - eta * grad_smooth
            theta_new = soft_threshold(theta_temp, eta * lam)  # Proximal step

        # Check convergence
        step_size = np.linalg.norm(theta_new - theta)
        theta = theta_new

        if step_size < tol:
            break

    return theta, mse_history


def gd_momentum(
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, beta=0.9, tol=1e-10
):
    """
    Unified Gradient Descent with momentum for OLS, Ridge, and Lasso.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    eta : float
        Learning rate
    num_iters : int
        Maximum number of iterations
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    beta : float
        Momentum parameter
    tol : float
        Convergence tolerance

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_features = X.shape[1]
    theta = np.zeros(n_features)
    v = np.zeros_like(theta)  # velocity
    mse_history = []

    for t in range(num_iters):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient + momentum step based on method
        if method.lower() == "ols":
            grad = OLS_gradient(X, y, theta)
            v = beta * v + grad
            theta = theta - eta * v

        elif method.lower() == "ridge":
            grad = Ridge_gradient(X, y, theta, lam)
            v = beta * v + grad
            theta = theta - eta * v

        elif method.lower() == "lasso":
            # For Lasso, apply momentum to smooth part, then proximal step
            grad_smooth = OLS_gradient(X, y, theta)
            v = beta * v + grad_smooth
            theta_temp = theta - eta * v
            theta = soft_threshold(theta_temp, eta * lam)

        # early stop (by grad norm)
        if method.lower() == "lasso":
            # For Lasso, check convergence on smooth gradient
            grad_for_check = OLS_gradient(X, y, theta)
        else:
            grad_for_check = grad

        if np.linalg.norm(grad_for_check) < tol:
            break

    return theta, mse_history


def gd_adagrad(
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, eps=1e-8, tol=1e-10
):
    """
    Unified AdaGrad for OLS, Ridge, and Lasso.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    eta : float
        Learning rate
    num_iters : int
        Maximum number of iterations
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    eps : float
        Small value to prevent division by zero
    tol : float
        Convergence tolerance

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_features = X.shape[1]
    theta = np.zeros(n_features)
    G = np.zeros(n_features)  # Accumulated squared gradients
    mse_history = []

    for t in range(num_iters):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient based on method
        if method.lower() == "ols":
            grad = OLS_gradient(X, y, theta)

        elif method.lower() == "ridge":
            grad = Ridge_gradient(X, y, theta, lam)

        elif method.lower() == "lasso":
            # For Lasso, apply AdaGrad to smooth part, then proximal step
            grad = OLS_gradient(X, y, theta)

        # AdaGrad update
        G += grad**2
        adapted_grad = grad / (np.sqrt(G) + eps)

        if method.lower() == "lasso":
            # Apply proximal step for Lasso
            theta_temp = theta - eta * adapted_grad
            theta = soft_threshold(theta_temp, eta * lam)
        else:
            theta = theta - eta * adapted_grad

        # early stop (by grad norm)
        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_rmsprop(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    method="ols",
    lam=None,
    beta=0.9,
    eps=1e-8,
    tol=1e-10,
):
    """
    Unified RMSProp for OLS, Ridge, and Lasso.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    eta : float
        Learning rate
    num_iters : int
        Maximum number of iterations
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    beta : float
        Decay rate for moving average of squared gradients
    eps : float
        Small value to prevent division by zero
    tol : float
        Convergence tolerance

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_features = X.shape[1]
    theta = np.zeros(n_features)
    S = np.zeros(n_features)  # EMA of squared gradients
    mse_history = []

    for t in range(num_iters):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient based on method
        if method.lower() == "ols":
            grad = OLS_gradient(X, y, theta)

        elif method.lower() == "ridge":
            grad = Ridge_gradient(X, y, theta, lam)

        elif method.lower() == "lasso":
            # For Lasso, apply RMSProp to smooth part, then proximal step
            grad = OLS_gradient(X, y, theta)

        # RMSProp update
        S = beta * S + (1.0 - beta) * (grad**2)
        adapted_grad = grad / (np.sqrt(S) + eps)

        if method.lower() == "lasso":
            # Apply proximal step for Lasso
            theta_temp = theta - eta * adapted_grad
            theta = soft_threshold(theta_temp, eta * lam)
        else:
            theta = theta - eta * adapted_grad

        # early stop (by grad norm)
        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def gd_adam(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    method="ols",
    lam=None,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    tol=1e-10,
    amsgrad=False,
):
    """
    Unified Adam optimizer for OLS, Ridge, and Lasso.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    eta : float
        Learning rate
    num_iters : int
        Maximum number of iterations
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    beta1 : float
        Exponential decay rate for first moment estimates
    beta2 : float
        Exponential decay rate for second moment estimates
    eps : float
        Small value to prevent division by zero
    tol : float
        Convergence tolerance
    amsgrad : bool
        Whether to use AMSGrad variant

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_features = X.shape[1]
    theta = np.zeros(n_features)
    m = np.zeros(n_features)  # first moment
    v = np.zeros(n_features)  # second moment
    v_max = np.zeros(n_features)  # for AMSGrad
    mse_history = []

    for t in range(1, num_iters + 1):
        # predict + MSE
        y_pred = X @ theta
        mse = mean_squared_error(y_true=y, y_pred=y_pred)
        mse_history.append(mse)

        # gradient based on method
        if method.lower() == "ols":
            grad = OLS_gradient(X, y, theta)

        elif method.lower() == "ridge":
            grad = Ridge_gradient(X, y, theta, lam)

        elif method.lower() == "lasso":
            # For Lasso, apply Adam to smooth part, then proximal step
            grad = OLS_gradient(X, y, theta)

        # Adam update
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

        adapted_grad = m_hat / denom

        if method.lower() == "lasso":
            # Apply proximal step for Lasso
            theta_temp = theta - eta * adapted_grad
            theta = soft_threshold(theta_temp, eta * lam)
        else:
            theta = theta - eta * adapted_grad

        # early stop (by grad norm)
        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history


def lasso_gradient(X, y, theta, lam, regularize_bias=False):
    """
    Compute gradient/subgradient for LASSO regression.

    Based on: ∂C/∂θ = (2/n)X^T(Xθ - y) + λ·sgn(θ)

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features)
    y : np.ndarray
        Target values
    theta : np.ndarray
        Current parameters
    lam : float
        Regularization parameter λ
    regularize_bias : bool
        Whether to apply L1 penalty to bias term (usually False)

    Returns
    -------
    np.ndarray
        Gradient vector
    """
    n = X.shape[0]

    # Standard MSE gradient: (2/n)X^T(Xθ - y)
    residual = X @ theta - y
    mse_grad = (2.0 / n) * (X.T @ residual)

    # L1 penalty gradient: λ·sgn(θ)
    # For θ = 0, we choose sgn(0) = 0 (could be any value in [-1,1])
    l1_grad = lam * np.sign(theta)

    # Combine gradients
    gradient = mse_grad + l1_grad

    # Optionally don't regularize bias term (first coefficient)
    if not regularize_bias and len(theta) > 0:
        gradient[0] = mse_grad[0]  # Remove L1 penalty from bias

    return gradient


def gd_lasso_basic(
    X, y, eta=1e-4, num_iters=10_000, lam=1e-2, tol=1e-10, regularize_bias=False
):
    """
    Basic gradient descent for LASSO regression.
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    mse_history = []

    for t in range(num_iters):
        # Calculate MSE for monitoring
        y_pred = X @ theta
        mse = mean_squared_error(y, y_pred)
        mse_history.append(mse)

        # Compute gradient
        grad = lasso_gradient(X, y, theta, lam, regularize_bias)

        # Update parameters
        theta = theta - eta * grad

        # Early stopping
        if np.linalg.norm(grad) < tol:
            break

    return theta, mse_history




##################################################
#          Stochastic gradient descent
##################################################
import autograd.numpy as np
from autograd import grad



def CostOLS(X, y, theta):
    """OLS cost function for autograd"""
    n = X.shape[0]
    return (1.0/n) * np.sum((y - X @ theta)**2)


def CostRidge(X, y, theta, lam):
    """Ridge cost function for autograd"""
    n = X.shape[0]
    return (1.0/n) * np.sum((y - X @ theta)**2) + lam * np.sum(theta**2)


def CostLassoSmooth(X, y, theta):
    """Smooth part of Lasso cost (MSE only) for proximal gradient"""
    return np.sum((y - X @ theta)**2)




def soft_threshold_autograd(x, threshold):
    """Soft thresholding operator compatible with autograd"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def create_minibatches(X, y, batch_size, shuffle=True):
    """Create mini-batches from data"""
    n = X.shape[0]
    if shuffle:
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    else:
        X_shuffled, y_shuffled = X, y
    
    batches = []
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batches.append((X_shuffled[i:end_idx], y_shuffled[i:end_idx]))
    
    return batches





def stochastic_gd(method, ...):
    if method = = "ols":
        cost = CostOLS(theta)
        grad = grad(cost)
        
    elif method == "ridge":
        cost = CostRidge(theta)
        grad = grad(cost)

    elif method == "lasso":
        cost = CostLasso(theta)
        grad = grad(cost)







def stochastic_gd(X, y, method="ols", lam=None, eta=0.01, n_epochs=50, 
                  batch_size=32, tol=1e-8, random_state=None):
    """
    Basic Stochastic Gradient Descent
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray  
        Target vector
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    eta : float
        Learning rate
    n_epochs : int
        Number of epochs
    batch_size : int
        Size of mini-batches
    tol : float
        Convergence tolerance
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each epoch
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_features = X.shape[1]
    theta = np.random.randn(n_features) * 0.01  # Small random initialization
    mse_history = []
    
    # Create gradient functions using autograd
    if method.lower() == "ols":
        training_gradient = grad(CostOLS, 2)  # gradient w.r.t. theta (3rd argument)
        cost_func = lambda th: CostOLS(y, X, th)
        
    elif method.lower() == "ridge":
        training_gradient = grad(CostRidge, 2)  # gradient w.r.t. theta
        cost_func = lambda th: CostRidge(y, X, th, lam)
        
    elif method.lower() == "lasso":
        training_gradient = grad(CostLassoSmooth, 2)  # gradient of smooth part only
        cost_func = lambda th: CostOLS(y, X, th) + lam * np.sum(np.abs(th))
    
    # Training loop
    for epoch in range(n_epochs):
        # Create mini-batches for this epoch
        batches = create_minibatches(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batches:
            # Compute gradient on mini-batch
            if method.lower() in ["ols", "lasso"]:
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta)
            else:  # ridge
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta, lam)
            
            # Standard gradient update
            theta = theta - eta * gradients
            
            # Apply proximal operator for Lasso
            if method.lower() == "lasso":
                theta = soft_threshold_autograd(theta, eta * lam)
        
        # Record MSE at end of epoch
        mse = mean_squared_error(y, X @ theta)
        mse_history.append(mse)
        
        # Early stopping check (optional)
        if len(mse_history) > 1 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break
    
    return theta, mse_history


def stochastic_gd_momentum(X, y, method="ols", lam=None, eta=0.01, n_epochs=50,
                          batch_size=32, beta=0.9, tol=1e-8, random_state=None):
    """Stochastic Gradient Descent with Momentum"""
    
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_features = X.shape[1]
    theta = np.random.randn(n_features) * 0.01
    velocity = np.zeros(n_features)  # momentum term
    mse_history = []
    
    # Create gradient functions
    if method.lower() == "ols":
        training_gradient = grad(CostOLS, 2)
    elif method.lower() == "ridge":
        training_gradient = grad(CostRidge, 2)
    elif method.lower() == "lasso":
        training_gradient = grad(CostLassoSmooth, 2)
    
    # Training loop
    for epoch in range(n_epochs):
        batches = create_minibatches(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batches:
            # Compute gradient
            if method.lower() in ["ols", "lasso"]:
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta)
            else:  # ridge
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta, lam)
            
            # Momentum update
            velocity = beta * velocity + gradients
            theta = theta - eta * velocity
            
            # Lasso proximal step
            if method.lower() == "lasso":
                theta = soft_threshold_autograd(theta, eta * lam)
        
        # Record MSE
        mse = mean_squared_error(y, X @ theta)
        mse_history.append(mse)
        
        if len(mse_history) > 1 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break
    
    return theta, mse_history


def stochastic_gd_adagrad(X, y, method="ols", lam=None, eta=0.01, n_epochs=50,
                         batch_size=32, eps=1e-8, tol=1e-8, random_state=None):
    """Stochastic Gradient Descent with AdaGrad"""
    
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_features = X.shape[1]
    theta = np.random.randn(n_features) * 0.01
    G = np.zeros(n_features)  # Accumulated squared gradients
    mse_history = []
    
    # Create gradient functions
    if method.lower() == "ols":
        training_gradient = grad(CostOLS, 2)
    elif method.lower() == "ridge":
        training_gradient = grad(CostRidge, 2)
    elif method.lower() == "lasso":
        training_gradient = grad(CostLassoSmooth, 2)
    
    # Training loop
    for epoch in range(n_epochs):
        batches = create_minibatches(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batches:
            # Compute gradient
            if method.lower() in ["ols", "lasso"]:
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta)
            else:  # ridge
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta, lam)
            
            # AdaGrad update
            G += gradients**2
            adapted_grad = gradients / (np.sqrt(G) + eps)
            theta = theta - eta * adapted_grad
            
            # Lasso proximal step
            if method.lower() == "lasso":
                theta = soft_threshold_autograd(theta, eta * lam)
        
        # Record MSE
        mse = mean_squared_error(y, X @ theta)
        mse_history.append(mse)
        
        if len(mse_history) > 1 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break
    
    return theta, mse_history


def stochastic_gd_rmsprop(X, y, method="ols", lam=None, eta=0.01, n_epochs=50,
                         batch_size=32, beta=0.9, eps=1e-8, tol=1e-8, random_state=None):
    """Stochastic Gradient Descent with RMSprop"""
    
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_features = X.shape[1]
    theta = np.random.randn(n_features) * 0.01
    S = np.zeros(n_features)  # Exponential moving average of squared gradients
    mse_history = []
    
    # Create gradient functions
    if method.lower() == "ols":
        training_gradient = grad(CostOLS, 2)
    elif method.lower() == "ridge":
        training_gradient = grad(CostRidge, 2)
    elif method.lower() == "lasso":
        training_gradient = grad(CostLassoSmooth, 2)
    
    # Training loop
    for epoch in range(n_epochs):
        batches = create_minibatches(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batches:
            # Compute gradient
            if method.lower() in ["ols", "lasso"]:
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta)
            else:  # ridge
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta, lam)
            
            # RMSprop update
            S = beta * S + (1.0 - beta) * (gradients**2)
            adapted_grad = gradients / (np.sqrt(S) + eps)
            theta = theta - eta * adapted_grad
            
            # Lasso proximal step
            if method.lower() == "lasso":
                theta = soft_threshold_autograd(theta, eta * lam)
        
        # Record MSE
        mse = mean_squared_error(y, X @ theta)
        mse_history.append(mse)
        
        if len(mse_history) > 1 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break
    
    return theta, mse_history


def stochastic_gd_adam(X, y, method="ols", lam=None, eta=0.01, n_epochs=50,
                      batch_size=32, beta1=0.9, beta2=0.999, eps=1e-8, 
                      tol=1e-8, random_state=None):
    """Stochastic Gradient Descent with Adam"""
    
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_features = X.shape[1]
    theta = np.random.randn(n_features) * 0.01
    m = np.zeros(n_features)  # First moment estimate
    v = np.zeros(n_features)  # Second moment estimate
    t = 0  # Time step
    mse_history = []
    
    # Create gradient functions
    if method.lower() == "ols":
        training_gradient = grad(CostOLS, 2)
    elif method.lower() == "ridge":
        training_gradient = grad(CostRidge, 2)
    elif method.lower() == "lasso":
        training_gradient = grad(CostLassoSmooth, 2)
    
    # Training loop
    for epoch in range(n_epochs):
        batches = create_minibatches(X, y, batch_size, shuffle=True)
        
        for X_batch, y_batch in batches:
            t += 1  # Increment time step
            
            # Compute gradient
            if method.lower() in ["ols", "lasso"]:
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta)
            else:  # ridge
                gradients = (1.0/len(y_batch)) * training_gradient(y_batch, X_batch, theta, lam)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * gradients
            v = beta2 * v + (1 - beta2) * (gradients**2)
            
            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Parameter update
            theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)
            
            # Lasso proximal step
            if method.lower() == "lasso":
                theta = soft_threshold_autograd(theta, eta * lam)
        
        # Record MSE
        mse = mean_squared_error(y, X @ theta)
        mse_history.append(mse)
        
        if len(mse_history) > 1 and abs(mse_history[-1] - mse_history[-2]) < tol:
            break
    
    return theta, mse_history

