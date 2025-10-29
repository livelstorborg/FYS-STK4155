import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
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
            weight_init='normal'
        )
        
        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(eta=eta)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(eta=eta)
        else:  
            optimizer = GD(eta=eta)  
        
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
            early_stopping=False,
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

def find_best_N_eta(N_list, eta_vals, layer_sizes, activations, 
                    optimizer_type='adam', epochs=500, batch_size=32, 
                    noise_std=0.1, test_size=0.2, seed=42):
    """
    Find the best combination of N (sample size) and eta (learning rate).
    
    Returns:
    --------
    test_mse_matrix : 2D array (n_etas x n_Ns)
    train_mse_matrix : 2D array (n_etas x n_Ns)
    best_eta : float
    best_N : int
    """
    n_etas = len(eta_vals)
    n_Ns = len(N_list)
    
    # Initialize result matrices
    test_mse_matrix = np.zeros((n_etas, n_Ns))
    train_mse_matrix = np.zeros((n_etas, n_Ns))
    

    
    for j, N in enumerate(N_list):

        
        # Generate data with fixed seed
        np.random.seed(seed)
        x = np.linspace(-1, 1, N)
        y_true = runge(x)
        y_noise = y_true + np.random.normal(0, noise_std, N)
        
        # Split data
        X_train_raw, X_test_raw, y_train_nn, y_test_nn = train_test_split(
            x.reshape(-1, 1), y_noise.reshape(-1, 1), 
            test_size=test_size, random_state=seed
        )
        
        # Scale data
        X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(
            X_train_raw, y_train_nn
        )
        X_test_s, y_test_s, _, _, _ = scale_data(
            X_test_raw, y_test_nn, X_mean, X_std, y_mean
        )
        
        for i, eta in enumerate(eta_vals):
            # Create model
            model = NeuralNetwork(
                network_input_size=1,
                layer_output_sizes=layer_sizes,
                activations=activations,
                loss=MSE(),
                seed=seed,
                lambda_reg=0.0,
                reg_type=None,
                weight_init='normal'
            )
            
            # Select optimizer
            if optimizer_type == 'adam':
                optimizer = Adam(eta=eta)
            elif optimizer_type == 'rmsprop':
                optimizer = RMSprop(eta=eta)
            else:  
                optimizer = GD(eta=eta)
            
            # Train
            train(
                nn=model,
                X_train=X_train_s,
                y_train=y_train_s,
                X_val=X_test_s,
                y_val=y_test_s,
                optimizer=optimizer,
                epochs=epochs,
                batch_size=batch_size if optimizer_type != 'gd' else len(X_train_s),
                stochastic=(optimizer_type != 'gd'),
                task='regression',
                early_stopping=False,
                patience=50,
                verbose=False,
                seed=seed
            )
            
            # Evaluate
            y_train_pred = inverse_scale_y(model.predict(X_train_s), y_mean)
            y_test_pred = inverse_scale_y(model.predict(X_test_s), y_mean)
            y_train_real = inverse_scale_y(y_train_s, y_mean)
            y_test_real = inverse_scale_y(y_test_s, y_mean)
            
            train_mse_val = mse(y_train_real, y_train_pred)
            test_mse_val = mse(y_test_real, y_test_pred)
            
            train_mse_matrix[i, j] = train_mse_val
            test_mse_matrix[i, j] = test_mse_val
    
    # Find best combination
    best_idx = np.unravel_index(np.argmin(test_mse_matrix), test_mse_matrix.shape)
    best_eta_idx, best_N_idx = best_idx
    best_eta = eta_vals[best_eta_idx]
    best_N = N_list[best_N_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST COMBINATION:")
    print(f"  N = {best_N}")
    print(f"  η = {best_eta:.6f}")
    print(f"  Test MSE = {test_mse_matrix[best_eta_idx, best_N_idx]:.6f}")
    print(f"  Train MSE = {train_mse_matrix[best_eta_idx, best_N_idx]:.6f}")
    print(f"{'='*60}\n")
    
    return test_mse_matrix, train_mse_matrix, best_eta, best_N










def save_results_to_csv(results, activation_name, save_dir='results'):
    """
    Save complexity analysis results to CSV file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary organized by number of layers
    activation_name : str
        Name of activation function (used in filename)
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    str : Path to saved file
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Flatten results dictionary into list of records
    records = []
    for n_layers, layer_results in results.items():
        for result in layer_results:
            records.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Sort by number of parameters for cleaner file
    df = df.sort_values('n_params').reset_index(drop=True)
    
    # Save to CSV
    filename = f"{activation_name.lower().replace(' ', '_')}_complexity_results.csv"
    filepath = save_path / filename
    df.to_csv(filepath, index=False)
    
    print(f"Results saved to: {filepath}")
    return str(filepath)


def load_results_from_csv(filepath):
    """
    Load complexity analysis results from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    
    Returns:
    --------
    dict : Results dictionary organized by number of layers
    """
    df = pd.read_csv(filepath)
    
    # Reconstruct nested dictionary structure
    results = {}
    for n_layers in sorted(df['n_layers'].unique()):
        layer_data = df[df['n_layers'] == n_layers]
        results[n_layers] = layer_data.to_dict('records')
    
    return results


def save_all_results(all_results, save_dir='results'):
    """
    Save all activation results and create a metadata file.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with activation names as keys, results as values
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    dict : Mapping of activation names to saved file paths
    """
    saved_files = {}
    
    for activation_name, results in all_results.items():
        filepath = save_results_to_csv(results, activation_name, save_dir)
        saved_files[activation_name] = filepath
    
    # Save metadata
    metadata = {
        'activations': list(all_results.keys()),
        'files': saved_files,
        'total_architectures': sum(
            sum(len(layer_results) for layer_results in results.values())
            for results in all_results.values()
        )
    }
    
    metadata_path = Path(save_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    return saved_files


def load_all_results(save_dir='results'):
    """
    Load all saved results using metadata file.
    
    Parameters:
    -----------
    save_dir : str
        Directory containing saved results
    
    Returns:
    --------
    dict : All results organized by activation name
    """
    metadata_path = Path(save_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata file found in {save_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_results = {}
    for activation_name, filepath in metadata['files'].items():
        all_results[activation_name] = load_results_from_csv(filepath)
    
    print(f"Loaded {len(all_results)} activation function(s)")
    return all_results


def print_results_summary(results, activation_name):
    """
    Print a summary of loaded results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    activation_name : str
        Name of activation function
    """
    print(f"\n{'='*60}")
    print(f"{activation_name} - Results Summary")
    print(f"{'='*60}")
    
    for n_layers in sorted(results.keys()):
        n_archs = len(results[n_layers])
        params_range = [r['n_params'] for r in results[n_layers]]
        print(f"{n_layers} layer(s): {n_archs} architectures, "
              f"{min(params_range):,} - {max(params_range):,} params")
    
    # Find best
    all_archs = [r for layer_results in results.values() for r in layer_results]
    best = min(all_archs, key=lambda x: x['test_loss'])
    
    print(f"\nBest: {best['n_layers']} layers × {best['n_neurons']} neurons")
    print(f"Test MSE: {best['test_loss']:.6f}")