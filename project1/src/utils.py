import numpy as np
from sklearn.metrics import mean_squared_error


def runge(x):
    """Runge function: f(x) = 1 / (1 + 25x^2)"""
    return 1.0 / (1 + 25 * x**2)


def polynomial_features(x: np.ndarray, p: int, intercept: bool = False) -> np.ndarray:
    """Generate polynomial features from input data."""
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
    """Unified analytical solution function for OLS and Ridge regression."""
    if method.lower() not in ["ols", "ridge"]:
        raise ValueError("Analytical solution only available for 'ols' and 'ridge'")
    if method.lower() == "ridge" and lam is None:
        raise ValueError("Ridge regression requires lam parameter")
    if method.lower() == "ols":
        return OLS_parameters(X, y)
    elif method.lower() == "ridge":
        return Ridge_parameters(X, y, lam)


def soft_threshold(x, threshold):
    """Soft thresholding operator for Lasso regression."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def learning_rate_schedule(t, eta0, t0=5, t1=50):
    """
    Learning rate schedule for SGD: eta(t) = eta0 * t0 / (t + t1)

    Parameters
    ----------
    t : int
        Current iteration/step number
    eta0 : float
        Initial learning rate
    t0, t1 : float
        Schedule parameters controlling decay rate
    """
    return eta0 * t0 / (t + t1)


def check_convergence_stochastic(
    method, eta, epoch, epoch_mse, mse_history, tol_relative
):
    status = False
    if np.isnan(epoch_mse) or np.isinf(epoch_mse):
        print(f"({method}) eta = {eta}, diverged (NaN/Inf) at epoch {epoch}")
        status = True
    # 2. Catastrophic explosion (10× initial MSE)
    if epoch_mse > 10 * mse_history[0]:
        print(f"({method}) eta = {eta}, diverged (exploded) at epoch {epoch}")
        status = True

    # 3. Sustained upward trend after warmup period
    if epoch >= 10 and len(mse_history) >= 5:
        last_5 = mse_history[-5:]
        # Check if all last 5 epochs are increasing
        all_increasing = all(last_5[i] > last_5[i - 1] for i in range(1, 5))
        # And current MSE is significantly worse than start
        if all_increasing and epoch_mse > 1.5 * mse_history[0]:
            print(
                f"({method}) eta = {eta}, diverged (sustained increase) at epoch {epoch}"
            )
            status = True

    # Test for convergence (after at least 1 epoch)
    if epoch > 0:
        abs_change = np.abs(mse_history[epoch] - mse_history[epoch - 1])
        rel_change = (
            abs_change / mse_history[epoch - 1]
            if mse_history[epoch - 1] > 1e-10
            else abs_change
        )

        if rel_change < tol_relative or abs_change < 1e-10:
            print(f"({method}) eta = {eta}, converged at epoch {epoch}")
            status = True

    return status


def check_convergence_non_stochastic(
    method, eta, t, mse, mse_history, tol_relative, ADAM=False
):
    status = False
    # Multi-criteria divergence detection
    # 1. Immediate catastrophic failure
    if np.isnan(mse) or np.isinf(mse):
        print(f"({method}) eta = {eta}, diverged (NaN/Inf)")
        status = True

    # 2. Catastrophic explosion (10× initial MSE)
    if mse > 10 * mse_history[0]:
        print(f"({method}) eta = {eta}, diverged (exploded)")
        status = True

    # 3. Sustained upward trend after warmup
    if t >= 50 and len(mse_history) >= 5:
        last_5 = mse_history[-5:]
        all_increasing = all(last_5[i] > last_5[i - 1] for i in range(1, 5))
        if all_increasing and mse > 1.5 * mse_history[0]:
            print(f"({method}) eta = {eta}, diverged (sustained increase)")
            status = True

    # Test for convergence
    if ADAM:
        if t > 1 and len(mse_history) >= 2:
            current_mse_idx = len(mse_history) - 1
            prev_mse_idx = len(mse_history) - 2
            abs_change = np.abs(
                mse_history[current_mse_idx] - mse_history[prev_mse_idx]
            )
            rel_change = (
                abs_change / mse_history[prev_mse_idx]
                if mse_history[prev_mse_idx] > 1e-10
                else abs_change
            )

            if rel_change < tol_relative or abs_change < 1e-10:
                print(f"({method}) eta = {eta}, converged at iteration {t}")
                status = True
    else:
        if t > 0:
            abs_change = np.abs(mse_history[t] - mse_history[t - 1])
            rel_change = (
                abs_change / mse_history[t - 1]
                if mse_history[t - 1] > 1e-10
                else abs_change
            )

            if rel_change < tol_relative or abs_change < 1e-10:
                print(f"({method}) eta = {eta}, converged at iteration {t}")
                status = True
    return status


def gradient_descent(
    X,
    y,
    eta,
    num_iters,
    method="ols",
    lam=None,
    tol_relative=1e-6,
    stochastic=False,
    batch_size=32,
    use_schedule=False,
    t0=5,
    t1=50,
):
    """
    Unified gradient descent function for OLS, Ridge, and Lasso regression.
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    mse_history = []

    if stochastic:
        m = int(n_samples / batch_size)
        n_epochs = num_iters
        t_global = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m):
                t_global += 1
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                current_eta = (
                    learning_rate_schedule(t_global, eta, t0, t1)
                    if use_schedule
                    else eta
                )

                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                    theta_new = theta - current_eta * grad
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                    theta_new = theta - current_eta * grad
                elif method.lower() == "lasso":
                    grad_smooth = OLS_gradient(X_batch, y_batch, theta)
                    theta_temp = theta - current_eta * grad_smooth
                    theta_new = soft_threshold(theta_temp, current_eta * lam)

                theta = theta_new

            # Evaluate on full dataset after each epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y, y_pred_full)
            mse_history.append(epoch_mse)

            status = check_convergence_stochastic(
                "Gradient Descent",
                eta=eta,
                epoch=epoch,
                epoch_mse=epoch_mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )
            if status:
                break

    else:
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

            theta = theta_new

            status = check_convergence_non_stochastic(
                method=method,
                eta=eta,
                t=t,
                mse=mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )

            if status:
                break

    return theta, mse_history


def gd_momentum(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    method="ols",
    lam=None,
    beta=0.9,
    tol_relative=1e-6,
    stochastic=False,
    batch_size=32,
    use_schedule=False,
    t0=5,
    t1=50,
):
    """
    Unified Gradient Descent with momentum for OLS, Ridge, and Lasso.
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    v = np.zeros_like(theta)
    mse_history = []

    if stochastic:
        m = int(n_samples / batch_size)
        n_epochs = num_iters
        t_global = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m):
                t_global += 1
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                current_eta = (
                    learning_rate_schedule(t_global, eta, t0, t1)
                    if use_schedule
                    else eta
                )

                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                    v = beta * v + grad
                    theta = theta - current_eta * v
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                    v = beta * v + grad
                    theta = theta - current_eta * v
                elif method.lower() == "lasso":
                    grad_smooth = OLS_gradient(X_batch, y_batch, theta)
                    v = beta * v + grad_smooth
                    theta_temp = theta - current_eta * v
                    theta = soft_threshold(theta_temp, current_eta * lam)

            # Evaluate on full dataset after each epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)

            status = check_convergence_stochastic(
                "Gradient Descent",
                eta=eta,
                epoch=epoch,
                epoch_mse=epoch_mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )
            if status:
                break

    else:
        for t in range(num_iters):
            y_pred = X @ theta
            mse = mean_squared_error(y_true=y, y_pred=y_pred)
            mse_history.append(mse)

            if method.lower() == "ols":
                grad = OLS_gradient(X, y, theta)
                v = beta * v + grad
                theta = theta - eta * v
            elif method.lower() == "ridge":
                grad = Ridge_gradient(X, y, theta, lam)
                v = beta * v + grad
                theta = theta - eta * v
            elif method.lower() == "lasso":
                grad_smooth = OLS_gradient(X, y, theta)
                v = beta * v + grad_smooth
                theta_temp = theta - eta * v
                theta = soft_threshold(theta_temp, eta * lam)

            status = check_convergence_non_stochastic(
                method=method,
                eta=eta,
                t=t,
                mse=mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )

            if status:
                break

    return theta, mse_history


def gd_adagrad(
    X,
    y,
    eta=1e-2,
    num_iters=10_000,
    method="ols",
    lam=None,
    eps=1e-8,
    tol_relative=1e-6,
    stochastic=False,
    batch_size=32,
):
    """
    Unified AdaGrad for OLS, Ridge, and Lasso.
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    G = np.zeros(n_features)
    mse_history = []

    if stochastic:
        m = int(n_samples / batch_size)
        n_epochs = num_iters

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    grad = OLS_gradient(X_batch, y_batch, theta)

                G += grad**2
                adapted_grad = grad / (np.sqrt(G) + eps)

                if method.lower() == "lasso":
                    theta_temp = theta - eta * adapted_grad
                    theta = soft_threshold(theta_temp, eta * lam)
                else:
                    theta = theta - eta * adapted_grad

            # Evaluate on full dataset after each epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)

            status = check_convergence_stochastic(
                "Gradient Descent",
                eta=eta,
                epoch=epoch,
                epoch_mse=epoch_mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )
            if status:
                break

    else:
        for t in range(num_iters):
            y_pred = X @ theta
            mse = mean_squared_error(y_true=y, y_pred=y_pred)
            mse_history.append(mse)

            if method.lower() == "ols":
                grad = OLS_gradient(X, y, theta)
            elif method.lower() == "ridge":
                grad = Ridge_gradient(X, y, theta, lam)
            elif method.lower() == "lasso":
                grad = OLS_gradient(X, y, theta)

            G += grad**2
            adapted_grad = grad / (np.sqrt(G) + eps)

            if method.lower() == "lasso":
                theta_temp = theta - eta * adapted_grad
                theta = soft_threshold(theta_temp, eta * lam)
            else:
                theta = theta - eta * adapted_grad

            status = check_convergence_non_stochastic(
                method=method,
                eta=eta,
                t=t,
                mse=mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )

            if status:
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
    tol_relative=1e-6,
    stochastic=False,
    batch_size=32,
):
    """
    Unified RMSProp for OLS, Ridge, and Lasso.
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    S = np.zeros(n_features)
    mse_history = []

    if stochastic:
        m = int(n_samples / batch_size)
        n_epochs = num_iters

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    grad = OLS_gradient(X_batch, y_batch, theta)

                S = beta * S + (1.0 - beta) * (grad**2)
                adapted_grad = grad / (np.sqrt(S) + eps)

                if method.lower() == "lasso":
                    theta_temp = theta - eta * adapted_grad
                    theta = soft_threshold(theta_temp, eta * lam)
                else:
                    theta = theta - eta * adapted_grad

            # Evaluate on full dataset after each epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)

            status = check_convergence_stochastic(
                "Gradient Descent",
                eta=eta,
                epoch=epoch,
                epoch_mse=epoch_mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )
            if status:
                break

    else:
        for t in range(num_iters):
            y_pred = X @ theta
            mse = mean_squared_error(y_true=y, y_pred=y_pred)
            mse_history.append(mse)

            if method.lower() == "ols":
                grad = OLS_gradient(X, y, theta)
            elif method.lower() == "ridge":
                grad = Ridge_gradient(X, y, theta, lam)
            elif method.lower() == "lasso":
                grad = OLS_gradient(X, y, theta)

            S = beta * S + (1.0 - beta) * (grad**2)
            adapted_grad = grad / (np.sqrt(S) + eps)

            if method.lower() == "lasso":
                theta_temp = theta - eta * adapted_grad
                theta = soft_threshold(theta_temp, eta * lam)
            else:
                theta = theta - eta * adapted_grad

            status = check_convergence_non_stochastic(
                method=method,
                eta=eta,
                t=t,
                mse=mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )

            if status:
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
    tol_relative=1e-6,
    amsgrad=False,
    stochastic=False,
    batch_size=32,
):
    """
    Unified Adam optimizer for OLS, Ridge, and Lasso.
    """
    if method.lower() not in ["ols", "ridge", "lasso"]:
        raise ValueError("method must be 'ols', 'ridge', or 'lasso'")
    if method.lower() in ["ridge", "lasso"] and lam is None:
        raise ValueError(f"{method} regression requires lam parameter")

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    m_moment = np.zeros(n_features)
    v_moment = np.zeros(n_features)
    v_max = np.zeros(n_features)
    mse_history = []

    if stochastic:
        m_batches = int(n_samples / batch_size)
        n_epochs = num_iters
        t = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m_batches):
                t += 1
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                if method.lower() == "ols":
                    grad = OLS_gradient(X_batch, y_batch, theta)
                elif method.lower() == "ridge":
                    grad = Ridge_gradient(X_batch, y_batch, theta, lam)
                elif method.lower() == "lasso":
                    grad = OLS_gradient(X_batch, y_batch, theta)

                m_moment = beta1 * m_moment + (1.0 - beta1) * grad
                v_moment = beta2 * v_moment + (1.0 - beta2) * (grad**2)

                m_hat = m_moment / (1.0 - beta1**t)
                v_hat = v_moment / (1.0 - beta2**t)

                if amsgrad:
                    v_max = np.maximum(v_max, v_hat)
                    denom = np.sqrt(v_max) + eps
                else:
                    denom = np.sqrt(v_hat) + eps

                adapted_grad = m_hat / denom

                if method.lower() == "lasso":
                    theta_temp = theta - eta * adapted_grad
                    theta = soft_threshold(theta_temp, eta * lam)
                else:
                    theta = theta - eta * adapted_grad

            # Evaluate on full dataset after each epoch
            y_pred_full = X @ theta
            epoch_mse = mean_squared_error(y_true=y, y_pred=y_pred_full)
            mse_history.append(epoch_mse)

            status = check_convergence_stochastic(
                "Gradient Descent",
                eta=eta,
                epoch=epoch,
                epoch_mse=epoch_mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
            )
            if status:
                break

    else:
        for t in range(1, num_iters + 1):
            y_pred = X @ theta
            mse = mean_squared_error(y_true=y, y_pred=y_pred)
            mse_history.append(mse)

            if method.lower() == "ols":
                grad = OLS_gradient(X, y, theta)
            elif method.lower() == "ridge":
                grad = Ridge_gradient(X, y, theta, lam)
            elif method.lower() == "lasso":
                grad = OLS_gradient(X, y, theta)

            m_moment = beta1 * m_moment + (1.0 - beta1) * grad
            v_moment = beta2 * v_moment + (1.0 - beta2) * (grad**2)

            m_hat = m_moment / (1.0 - beta1**t)
            v_hat = v_moment / (1.0 - beta2**t)

            if amsgrad:
                v_max = np.maximum(v_max, v_hat)
                denom = np.sqrt(v_max) + eps
            else:
                denom = np.sqrt(v_hat) + eps

            adapted_grad = m_hat / denom

            if method.lower() == "lasso":
                theta_temp = theta - eta * adapted_grad
                theta = soft_threshold(theta_temp, eta * lam)
            else:
                theta = theta - eta * adapted_grad

            status = check_convergence_non_stochastic(
                method=method,
                eta=eta,
                t=t,
                mse=mse,
                mse_history=mse_history,
                tol_relative=tol_relative,
                ADAM=True,
            )

            if status:
                break

    return theta, mse_history
