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


def gradient_descent(X, y, eta, num_iters, method="ols", lam=None, tol=1e-8, stochastic=False, batch_size=32):
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
        Maximum number of iterations (for batch GD) or epochs (for SGD)
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    tol : float
        Convergence tolerance
    stochastic : bool
        If True, use stochastic gradient descent
    batch_size : int
        Size of minibatches for SGD

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration/epoch
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    mse_history = []

    if stochastic:
        # Stochastic Gradient Descent with minibatches
        m = int(n_samples / batch_size)  # number of minibatches
        n_epochs = num_iters  # treat num_iters as epochs for SGD
        
        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_mse = 0
            for i in range(m):
                # Pick random minibatch
                k = np.random.randint(m)
                start_idx = k * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradient on minibatch
                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                    theta_new = theta - eta * grad
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                    theta_new = theta - eta * grad
                elif method.lower() == "lasso":
                    grad_smooth = OLS_gradient(X_batch, y_batch, theta)
                    theta_temp = theta - eta * grad_smooth
                    theta_new = soft_threshold(theta_temp, eta * lam)
                
                theta = theta_new
            
            # Calculate MSE on full dataset at end of epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y, y_pred_full)
            mse_history.append(epoch_mse)
            
            # Check convergence (optional for SGD)
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
                break
                
    else:
        # Standard Batch Gradient Descent (your original implementation)
        for t in range(num_iters):
            y_pred = X @ theta
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

            if method.lower() == "ols":
                grad = OLS_gradient(X, y, theta)
                theta_new = theta - eta * grad
            elif method.lower() == "ridge":
                grad = Ridge_gradient(X, y, theta, lam)
                theta_new = theta - eta * grad
            elif method.lower() == "lasso":
                grad_smooth = OLS_gradient(X, y, theta)
                theta_temp = theta - eta * grad_smooth
                theta_new = soft_threshold(theta_temp, eta * lam)

            step_size = np.linalg.norm(theta_new - theta)
            theta = theta_new

            if step_size < tol:
                break

    return theta, mse_history

def gd_momentum(
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, beta=0.9, tol=1e-10, 
    stochastic=False, batch_size=32
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
        Maximum number of iterations (epochs if stochastic=True)
    method : str
        Regression method: 'ols', 'ridge', or 'lasso'
    lam : float, optional
        Regularization parameter (required for ridge and lasso)
    beta : float
        Momentum parameter
    tol : float
        Convergence tolerance
    stochastic : bool
        If True, use stochastic gradient descent
    batch_size : int
        Size of minibatches for SGD

    Returns
    -------
    theta : np.ndarray
        Fitted parameters
    mse_history : list
        MSE values at each iteration/epoch
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    v = np.zeros_like(theta)  # velocity
    mse_history = []

    if stochastic:
        # Stochastic Gradient Descent with momentum
        m = int(n_samples / batch_size)  # number of minibatches
        n_epochs = num_iters  # treat num_iters as epochs for SGD
        
        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                # Pick random minibatch
                k = np.random.randint(m)
                start_idx = k * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # gradient + momentum step based on method
                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                    v = beta * v + grad
                    theta = theta - eta * v

                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                    v = beta * v + grad
                    theta = theta - eta * v

                elif method.lower() == "lasso":
                    # For Lasso, apply momentum to smooth part, then proximal step
                    grad_smooth = OLS_gradient(X_batch, y_batch, theta)
                    v = beta * v + grad_smooth
                    theta_temp = theta - eta * v
                    theta = soft_threshold(theta_temp, eta * lam)
            
            # Calculate MSE on full dataset at end of epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)
            
            # Check convergence (optional for SGD)
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
                break
    else:
        # Standard Batch Gradient Descent with momentum
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
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, eps=1e-8, tol=1e-10,
    stochastic=False, batch_size=32
):
    """
    Unified AdaGrad for OLS, Ridge, and Lasso.
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    G = np.zeros(n_features)  # Accumulated squared gradients
    mse_history = []

    if stochastic:
        # Stochastic AdaGrad
        m = int(n_samples / batch_size)  # number of minibatches
        n_epochs = num_iters  # treat num_iters as epochs for SGD
        
        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                # Pick random minibatch
                k = np.random.randint(m)
                start_idx = k * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # gradient based on method
                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    # For Lasso, apply AdaGrad to smooth part, then proximal step
                    grad = OLS_gradient(X_batch, y_batch, theta)

                # AdaGrad update
                G += grad**2
                adapted_grad = grad / (np.sqrt(G) + eps)

                if method.lower() == "lasso":
                    # Apply proximal step for Lasso
                    theta_temp = theta - eta * adapted_grad
                    theta = soft_threshold(theta_temp, eta * lam)
                else:
                    theta = theta - eta * adapted_grad
            
            # Calculate MSE on full dataset at end of epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)
            
            # Check convergence (optional for SGD)
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
                break
    else:
        # Standard Batch AdaGrad
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
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, beta=0.9, eps=1e-8, tol=1e-10,
    stochastic=False, batch_size=32
):
    """
    Unified RMSProp for OLS, Ridge, and Lasso.
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    S = np.zeros(n_features)  # EMA of squared gradients
    mse_history = []

    if stochastic:
        # Stochastic RMSProp
        m = int(n_samples / batch_size)  # number of minibatches
        n_epochs = num_iters  # treat num_iters as epochs for SGD
        
        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                # Pick random minibatch
                k = np.random.randint(m)
                start_idx = k * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # gradient based on method
                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    # For Lasso, apply RMSProp to smooth part, then proximal step
                    grad = OLS_gradient(X_batch, y_batch, theta)

                # RMSProp update
                S = beta * S + (1.0 - beta) * (grad**2)
                adapted_grad = grad / (np.sqrt(S) + eps)

                if method.lower() == "lasso":
                    # Apply proximal step for Lasso
                    theta_temp = theta - eta * adapted_grad
                    theta = soft_threshold(theta_temp, eta * lam)
                else:
                    theta = theta - eta * adapted_grad
            
            # Calculate MSE on full dataset at end of epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)
            
            # Check convergence (optional for SGD)
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
                break
    else:
        # Standard Batch RMSProp
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
    X, y, eta=1e-2, num_iters=10_000, method="ols", lam=None, beta1=0.9, beta2=0.999, 
    eps=1e-8, tol=1e-10, amsgrad=False, stochastic=False, batch_size=32
):
    """
    Unified Adam optimizer for OLS, Ridge, and Lasso.
    """
    # Validate inputs
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")

    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    m = np.zeros(n_features)  # first moment
    v = np.zeros(n_features)  # second moment
    v_max = np.zeros(n_features)  # for AMSGrad
    mse_history = []

    if stochastic:
        # Stochastic Adam
        m_batches = int(n_samples / batch_size)  # number of minibatches
        n_epochs = num_iters  # treat num_iters as epochs for SGD
        t = 0  # global step counter for bias correction
        
        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m_batches):
                t += 1  # increment global step counter
                
                # Pick random minibatch
                k = np.random.randint(m_batches)
                start_idx = k * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # gradient based on method
                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    # For Lasso, apply Adam to smooth part, then proximal step
                    grad = OLS_gradient(X_batch, y_batch, theta)

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
            
            # Calculate MSE on full dataset at end of epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)
            
            # Check convergence (optional for SGD)
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tol:
                break
    else:
        # Standard Batch Adam
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





