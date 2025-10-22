import numpy as np
from .neural_network import NeuralNetwork
from .training import train
from .losses import MSE
from .metrics import mse
from .optimizers import Adam, RMSprop, GD


def runge(x):
    """Runge function: f(x) = 1 / (1 + 25x^2)"""
    return 1.0 / (1 + 25 * x**2)


def polynomial_features(x, p, intercept=False):
    """
    Generate polynomial features from input data.
    
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


def scale_data(X, y, X_mean=None, X_std=None, y_mean=None):
    """
    Args:
        X: Feature matrix
        y: Target values
        X_mean, X_std, y_mean: Scaling parameters (None for training mode)
    
    Returns:
        X_scaled, y_scaled, X_mean, X_std, y_mean
    """
    if X_mean is None:  # Training mode
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        y_mean = np.mean(y)

    X_scaled = (X - X_mean) / X_std
    y_scaled = y - y_mean

    return X_scaled, y_scaled, X_mean, X_std, y_mean


def OLS_parameters(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def Ridge_parameters(X, y, lam):
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(n)) @ X.T @ y

def lasso_gd(X, y, lam=0.01, eta=0.01, max_iter=1000, tol=1e-6):
    """
    Lasso regression using Gradient Descent.
    
    Minimizes: Loss = ||y - Xθ||² + λ||θ||₁
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target values (n_samples,)
        lam: L1 regularization strength (lambda)
        eta: Learning rate
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        theta: Learned parameters
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    
    for iteration in range(max_iter):
        # Predictions
        y_pred = X @ theta
        
        # Gradient of MSE loss: -(2/n) * X^T * (y - y_pred)
        residual = y - y_pred
        grad_mse = -(2.0 / n_samples) * (X.T @ residual)
        
        # Gradient of L1 penalty: λ * sign(θ)
        grad_l1 = lam * np.sign(theta)
        
        # Total gradient
        grad = grad_mse + grad_l1
        
        # Update parameters
        theta_new = theta - eta * grad
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Converged at iteration {iteration}")
            break
        
        theta = theta_new
    
    return theta


def inverse_scale_y(y_scaled, y_mean):
    """
    Inverse scale y values.
    
    Args:
        y_scaled: Scaled y values
        y_mean: Mean used for scaling
    
    Returns:
        Original scale y values
    """
    return y_scaled + y_mean

def find_best_eta(layer_sizes, activations, X_train, y_train, X_test, y_test, 
                  y_mean, optimizer_type='adam', eta_vals=None, 
                  epochs=500, batch_size=32, seed=42):
    """
    Find the best learning rate for a given architecture.
    
    Parameters:
    -----------
    layer_sizes : list
        Network architecture (e.g., [50, 1])
    activations : list
        Activation functions for each layer
    X_train, y_train : scaled training data
    X_test, y_test : scaled test data
    y_mean : for inverse scaling
    optimizer_type : 'adam', 'rmsprop', or 'gd'
    eta_vals : array of learning rates to test
    
    Returns:
    --------
    best_eta : float
        Best learning rate found
    best_test_mse : float
        MSE with best learning rate
    results : dict
        All results for plotting
    """
    if eta_vals is None:
        eta_vals = np.logspace(-5, -1, 5)
    
    test_mse_list = []
    train_mse_list = []
    
    for eta in eta_vals:
        # Create model
        model = NeuralNetwork(
            network_input_size=1,
            layer_output_sizes=layer_sizes,
            activations=activations,
            loss=MSE(),
            seed=seed,
            lambda_reg=0.0,
            reg_type=None,
            weight_init='xavier'
        )
        
        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(eta=eta)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(eta=eta)
        else:  # Plain GD
            optimizer = GD(eta=eta)  # Or your GD class if you have one
        
        # Train
        train(
            nn=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size if optimizer_type != 'gd' else len(X_train),
            stochastic=(optimizer_type != 'gd'),
            task='regression',
            early_stopping=True,
            patience=50,
            verbose=False,
            seed=seed
        )
        
        # Evaluate
        y_train_pred = inverse_scale_y(model.predict(X_train), y_mean)
        y_test_pred = inverse_scale_y(model.predict(X_test), y_mean)
        y_train_real = inverse_scale_y(y_train, y_mean)
        y_test_real = inverse_scale_y(y_test, y_mean)
        
        train_mse_val = mse(y_train_real, y_train_pred)
        test_mse_val = mse(y_test_real, y_test_pred)
        
        test_mse_list.append(test_mse_val)
        train_mse_list.append(train_mse_val)
        
        print(f"  η={eta:.6f}: Train MSE={train_mse_val:.6f}, Test MSE={test_mse_val:.6f}")
    
    # Find best
    best_idx = np.argmin(test_mse_list)
    best_eta = eta_vals[best_idx]
    best_test_mse = test_mse_list[best_idx]
    best_train_mse = train_mse_list[best_idx]
    
    results = {
        'eta_vals': eta_vals,
        'train_mse': np.array(train_mse_list),
        'test_mse': np.array(test_mse_list),
        'best_eta': best_eta,
        'best_train_mse': best_train_mse,
        'best_test_mse': best_test_mse
    }
    
    return best_eta, best_test_mse, results