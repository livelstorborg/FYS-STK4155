"""
Classification Experiments for Project 2 - Part F
Comprehensive script for MNIST classification experiments with various neural network configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime

import sys, os
from pathlib import Path
notebook_dir = Path().resolve()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, Softmax
from src.losses import CrossEntropy
from src.optimizers import Adam, RMSprop
from src.training import train
from src.metrics import accuracy


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_mnist(test_size=0.2, seed=42):
    """
    Load and prepare MNIST dataset with standardization
    
    Parameters:
    -----------
    test_size : float
        Fraction of data to use for testing
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded)
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target.astype(int)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Scale pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels for neural network
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    print(f"Training set: {X_train.shape}, {y_train_encoded.shape}")
    print(f"Test set: {X_test.shape}, {y_test_encoded.shape}")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def one_hot_encode(y, n_classes=10):
    """One-hot encode labels"""
    n_samples = len(y)
    y_encoded = np.zeros((n_samples, n_classes))
    y_encoded[np.arange(n_samples), y] = 1
    return y_encoded


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_single_model(eta, lam, optimizer_name, X_train, y_train_encoded, X_test, y_test_encoded,
                       y_train, y_test, hidden_layers, epochs, batch_size, patience, seed):
    """
    Train a single neural network model with given hyperparameters
    
    Parameters:
    -----------
    eta : float
        Learning rate
    lam : float
        Regularization parameter (lambda)
    optimizer_name : str
        Name of optimizer ('adam' or 'rmsprop')
    X_train, y_train_encoded : array-like
        Training data
    X_test, y_test_encoded : array-like
        Test data
    y_train, y_test : array-like
        Original labels (not one-hot encoded)
    hidden_layers : list
        List of hidden layer sizes
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    patience : int
        Early stopping patience
    seed : int
        Random seed
        
    Returns:
    --------
    dict : Results including model, accuracies, and training history
    """
    np.random.seed(seed)
    
    # Initialize neural network
    network_input_size = X_train.shape[1]
    # layer_output_sizes expects a list with sizes for each layer including output
    layer_output_sizes = list(hidden_layers) + [10]
    # activations is a list of Activation objects (one per layer)
    activations = [ReLU() for _ in hidden_layers] + [Softmax()]

    nn = NeuralNetwork(
        network_input_size=network_input_size,
        layer_output_sizes=layer_output_sizes,
        activations=activations,
        loss=CrossEntropy(),
        seed=seed,
        lambda_reg=lam
    )
    
    # Initialize optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(eta=eta)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = RMSprop(eta=eta)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Train the model
    history = train(
        nn,
        X_train,
        y_train_encoded,
        X_test,
        y_test_encoded,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=False
    )
    
    # Calculate accuracies (predictions -> class labels handled inside accuracy)
    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)

    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    
    return {
        'model': nn,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'history': history,
        'eta': eta,
        'lambda': lam,
        'optimizer': optimizer_name,
        'hidden_layers': hidden_layers
    }


# ============================================================================
# EXPERIMENT 1: HYPERPARAMETER GRID SEARCH
# ============================================================================

def hyperparameter_grid_search(X_train, y_train_encoded, X_test, y_test_encoded,
                                y_train, y_test, eta_values, lambda_values,
                                optimizer_name='adam', hidden_layers=[64],
                                epochs=20, batch_size=64, patience=5, seed=42,
                                n_jobs=-1):
    """
    Perform grid search over learning rates and regularization parameters
    
    Parameters:
    -----------
    eta_values : array-like
        Learning rates to test
    lambda_values : array-like
        Regularization parameters to test
    optimizer_name : str
        Optimizer to use ('adam' or 'rmsprop')
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
        
    Returns:
    --------
    dict : Results organized by eta and lambda
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER GRID SEARCH - Optimizer: {optimizer_name.upper()}")
    print(f"{'='*70}")
    print(f"Testing {len(eta_values)} learning rates x {len(lambda_values)} lambda values")
    print(f"Total models to train: {len(eta_values) * len(lambda_values)}")
    print(f"Hidden layers: {hidden_layers}")
    
    # Create parameter combinations
    param_combinations = [(eta, lam) for eta in eta_values for lam in lambda_values]
    
    # Train models in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_model)(
            eta, lam, optimizer_name, X_train, y_train_encoded, X_test, y_test_encoded,
            y_train, y_test, hidden_layers, epochs, batch_size, patience, seed
        )
        for eta, lam in param_combinations
    )
    
    # Organize results into grid
    results_grid = {}
    for result in results:
        eta = result['eta']
        lam = result['lambda']
        if eta not in results_grid:
            results_grid[eta] = {}
        results_grid[eta][lam] = result
    
    # Find best model
    best_result = max(results, key=lambda x: x['test_acc'])
    
    print(f"\nBest {optimizer_name.upper()} model:")
    print(f"  Learning rate: {best_result['eta']}")
    print(f"  Lambda: {best_result['lambda']}")
    print(f"  Test accuracy: {best_result['test_acc']:.4f}")
    print(f"  Train accuracy: {best_result['train_acc']:.4f}")
    
    return {
        'results_grid': results_grid,
        'best_result': best_result,
        'eta_values': eta_values,
        'lambda_values': lambda_values,
        'optimizer': optimizer_name
    }


# ============================================================================
# EXPERIMENT 2: FIXED LEARNING RATE, VARYING ARCHITECTURE
# ============================================================================

def fixed_lr_varying_architecture(X_train, y_train_encoded, X_test, y_test_encoded,
                                   y_train, y_test, learning_rate, hidden_configs,
                                   optimizer_name='adam', lambda_val=0.0,
                                   epochs=20, batch_size=64, patience=5, seed=42,
                                   n_jobs=-1):
    """
    Train models with fixed learning rate but varying architectures
    
    Parameters:
    -----------
    learning_rate : float
        Fixed learning rate to use
    hidden_configs : list of lists
        Different hidden layer configurations to test
        Example: [[32], [64], [128], [64, 64], [128, 128]]
    optimizer_name : str
        Optimizer to use ('adam' or 'rmsprop')
    lambda_val : float
        Regularization parameter
        
    Returns:
    --------
    dict : Results for each architecture
    """
    print(f"\n{'='*70}")
    print(f"FIXED LEARNING RATE EXPERIMENT - Optimizer: {optimizer_name.upper()}")
    print(f"{'='*70}")
    print(f"Learning rate: {learning_rate}")
    print(f"Lambda: {lambda_val}")
    print(f"Testing {len(hidden_configs)} different architectures")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_model)(
            learning_rate, lambda_val, optimizer_name,
            X_train, y_train_encoded, X_test, y_test_encoded,
            y_train, y_test, config, epochs, batch_size, patience, seed
        )
        for config in hidden_configs
    )
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    print(f"\nResults (sorted by test accuracy):")
    print(f"{'Architecture':<25} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 50)
    for result in results_sorted:
        arch_str = str(result['hidden_layers'])
        print(f"{arch_str:<25} {result['train_acc']:.4f}       {result['test_acc']:.4f}")
    
    return {
        'results': results_sorted,
        'learning_rate': learning_rate,
        'lambda': lambda_val,
        'optimizer': optimizer_name
    }


# ============================================================================
# EXPERIMENT 3: FIXED WIDTH, VARYING DEPTH
# ============================================================================

def fixed_width_varying_depth(X_train, y_train_encoded, X_test, y_test_encoded,
                               y_train, y_test, learning_rate, width, depths,
                               optimizer_name='adam', lambda_val=0.0,
                               epochs=20, batch_size=64, patience=5, seed=42,
                               n_jobs=-1):
    """
    Train models with fixed number of neurons per layer but varying number of layers
    
    Parameters:
    -----------
    learning_rate : float
        Learning rate to use
    width : int
        Number of neurons in each hidden layer
    depths : list of ints
        Number of hidden layers to test
    optimizer_name : str
        Optimizer to use ('adam' or 'rmsprop')
    lambda_val : float
        Regularization parameter
        
    Returns:
    --------
    dict : Results for each depth
    """
    print(f"\n{'='*70}")
    print(f"FIXED WIDTH, VARYING DEPTH - Optimizer: {optimizer_name.upper()}")
    print(f"{'='*70}")
    print(f"Width (neurons per layer): {width}")
    print(f"Learning rate: {learning_rate}")
    print(f"Lambda: {lambda_val}")
    print(f"Testing depths: {depths}")
    
    # Create configurations
    hidden_configs = [[width] * depth for depth in depths]
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_model)(
            learning_rate, lambda_val, optimizer_name,
            X_train, y_train_encoded, X_test, y_test_encoded,
            y_train, y_test, config, epochs, batch_size, patience, seed
        )
        for config in hidden_configs
    )
    
    print(f"\nResults:")
    print(f"{'Depth':<10} {'Architecture':<25} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 60)
    for depth, result in zip(depths, results):
        arch_str = str(result['hidden_layers'])
        print(f"{depth:<10} {arch_str:<25} {result['train_acc']:.4f}       {result['test_acc']:.4f}")
    
    return {
        'results': results,
        'depths': depths,
        'width': width,
        'learning_rate': learning_rate,
        'lambda': lambda_val,
        'optimizer': optimizer_name
    }


# ============================================================================
# EXPERIMENT 4: OPTIMIZER COMPARISON
# ============================================================================

def compare_optimizers(X_train, y_train_encoded, X_test, y_test_encoded,
                       y_train, y_test, learning_rate, hidden_layers,
                       lambda_val=0.0, epochs=20, batch_size=64,
                       patience=5, seed=42):
    """
    Compare Adam and RMSprop optimizers with same hyperparameters
    
    Returns:
    --------
    dict : Results for both optimizers
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZER COMPARISON")
    print(f"{'='*70}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Learning rate: {learning_rate}")
    print(f"Lambda: {lambda_val}")
    
    results = {}
    
    for optimizer_name in ['adam', 'rmsprop']:
        print(f"\nTraining with {optimizer_name.upper()}...")
        result = train_single_model(
            learning_rate, lambda_val, optimizer_name,
            X_train, y_train_encoded, X_test, y_test_encoded,
            y_train, y_test, hidden_layers, epochs, batch_size, patience, seed
        )
        results[optimizer_name] = result
        print(f"{optimizer_name.upper()} - Train: {result['train_acc']:.4f}, Test: {result['test_acc']:.4f}")
    
    return results


# ============================================================================
# BASELINE MODELS
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test,
                               max_iter=500, solver='lbfgs', seed=42):
    """Train logistic regression baseline"""
    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION BASELINE")
    print(f"{'='*70}")
    
    model = LogisticRegression(
        solver=solver,
        multi_class='multinomial',
        max_iter=max_iter,
        random_state=seed,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return {'model': model, 'train_acc': train_acc, 'test_acc': test_acc}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_hyperparameter_heatmap(results_grid, eta_values, lambda_values,
                                 metric='test_acc', title_suffix=''):
    """
    Create heatmap of results from grid search
    
    Parameters:
    -----------
    results_grid : dict
        Results from hyperparameter_grid_search
    eta_values : array-like
        Learning rates tested
    lambda_values : array-like
        Lambda values tested
    metric : str
        Metric to plot ('test_acc' or 'train_acc')
    title_suffix : str
        Additional text for title
    """
    # Create matrix of values
    matrix = np.zeros((len(eta_values), len(lambda_values)))
    for i, eta in enumerate(eta_values):
        for j, lam in enumerate(lambda_values):
            matrix[i, j] = results_grid[eta][lam][metric]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='lower')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(lambda_values)))
    ax.set_yticks(np.arange(len(eta_values)))
    ax.set_xticklabels([f'{lam:.0e}' for lam in lambda_values], rotation=45, ha='right')
    ax.set_yticklabels([f'{eta:.0e}' for eta in eta_values])
    
    ax.set_xlabel('Lambda (Regularization)', fontsize=12)
    ax.set_ylabel('Learning Rate (η)', fontsize=12)
    ax.set_title(f'Test Accuracy Heatmap{title_suffix}', fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=12)
    
    # Add text annotations
    for i in range(len(eta_values)):
        for j in range(len(lambda_values)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_architecture_comparison(results, x_label='Architecture'):
    """
    Plot bar chart comparing different architectures
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    x_label : str
        Label for x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    architectures = [str(r['hidden_layers']) for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    x = np.arange(len(architectures))
    width = 0.35
    
    ax.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison Across Architectures', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_depth_analysis(results, depths):
    """
    Plot how accuracy changes with network depth
    
    Parameters:
    -----------
    results : list
        List of result dictionaries from fixed_width_varying_depth
    depths : list
        List of depths tested
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    ax.plot(depths, train_accs, marker='o', label='Train Accuracy', linewidth=2)
    ax.plot(depths, test_accs, marker='s', label='Test Accuracy', linewidth=2)
    
    ax.set_xlabel('Number of Hidden Layers (Depth)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of Network Depth on Accuracy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)
    
    plt.tight_layout()
    return fig


def plot_training_curves(history, title='Training History'):
    """
    Plot training and validation loss over epochs
    
    Parameters:
    -----------
    history : dict
        Training history with 'train_loss' and 'val_loss' keys
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# SAVE AND LOAD RESULTS
# ============================================================================

def save_results(results, filename, output_dir='results'):
    """Save results to pickle file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {filepath}")


def load_results(filename, output_dir='results'):
    """Load results from pickle file"""
    filepath = Path(output_dir) / filename
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_all_experiments(save_results_flag=True, n_jobs=-1):
    """
    Run all classification experiments
    
    Parameters:
    -----------
    save_results_flag : bool
        Whether to save results to disk
    n_jobs : int
        Number of parallel jobs
    """
    print("="*80)
    print("MNIST CLASSIFICATION EXPERIMENTS - PROJECT 2 PART F")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 64
    PATIENCE = 5
    TEST_SIZE = 0.2
    
    # Load data
    X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = \
        load_and_prepare_mnist(test_size=TEST_SIZE, seed=SEED)
    
    all_results = {}
    
    # ========================================================================
    # EXPERIMENT 1: Hyperparameter Grid Search - Adam
    # ========================================================================
    eta_values = np.logspace(-4, -2, 7)  # 1e-4 to 1e-2
    lambda_values = np.logspace(-5, -1, 7)  # 1e-5 to 1e-1
    
    grid_search_adam = hyperparameter_grid_search(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, eta_values, lambda_values,
        optimizer_name='adam', hidden_layers=[64],
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        seed=SEED, n_jobs=n_jobs
    )
    all_results['grid_search_adam'] = grid_search_adam
    
    # ========================================================================
    # EXPERIMENT 2: Hyperparameter Grid Search - RMSprop
    # ========================================================================
    grid_search_rmsprop = hyperparameter_grid_search(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, eta_values, lambda_values,
        optimizer_name='rmsprop', hidden_layers=[64],
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        seed=SEED, n_jobs=n_jobs
    )
    all_results['grid_search_rmsprop'] = grid_search_rmsprop
    
    # ========================================================================
    # EXPERIMENT 3: Fixed LR, Varying Architecture - Adam
    # ========================================================================
    fixed_lr = 1e-3  # Use a good learning rate
    hidden_configs = [
        [32], [64], [128], [256],  # Single layer, varying width
        [64, 32], [64, 64], [128, 64], [128, 128],  # Two layers
        [128, 64, 32], [128, 128, 64]  # Three layers
    ]
    
    arch_adam = fixed_lr_varying_architecture(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, learning_rate=fixed_lr,
        hidden_configs=hidden_configs, optimizer_name='adam',
        lambda_val=0.0, epochs=EPOCHS, batch_size=BATCH_SIZE,
        patience=PATIENCE, seed=SEED, n_jobs=n_jobs
    )
    all_results['architecture_adam'] = arch_adam
    
    # ========================================================================
    # EXPERIMENT 4: Fixed LR, Varying Architecture - RMSprop
    # ========================================================================
    arch_rmsprop = fixed_lr_varying_architecture(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, learning_rate=fixed_lr,
        hidden_configs=hidden_configs, optimizer_name='rmsprop',
        lambda_val=0.0, epochs=EPOCHS, batch_size=BATCH_SIZE,
        patience=PATIENCE, seed=SEED, n_jobs=n_jobs
    )
    all_results['architecture_rmsprop'] = arch_rmsprop
    
    # ========================================================================
    # EXPERIMENT 5: Fixed Width, Varying Depth - Adam
    # ========================================================================
    depths = [1, 2, 3, 4, 5]
    width = 128
    
    depth_adam = fixed_width_varying_depth(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, learning_rate=fixed_lr, width=width,
        depths=depths, optimizer_name='adam', lambda_val=0.0,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        seed=SEED, n_jobs=n_jobs
    )
    all_results['depth_adam'] = depth_adam
    
    # ========================================================================
    # EXPERIMENT 6: Fixed Width, Varying Depth - RMSprop
    # ========================================================================
    depth_rmsprop = fixed_width_varying_depth(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, learning_rate=fixed_lr, width=width,
        depths=depths, optimizer_name='rmsprop', lambda_val=0.0,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        seed=SEED, n_jobs=n_jobs
    )
    all_results['depth_rmsprop'] = depth_rmsprop
    
    # ========================================================================
    # EXPERIMENT 7: Direct Optimizer Comparison
    # ========================================================================
    optimizer_comp = compare_optimizers(
        X_train, y_train_encoded, X_test, y_test_encoded,
        y_train, y_test, learning_rate=fixed_lr,
        hidden_layers=[128, 64], lambda_val=0.0,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        seed=SEED
    )
    all_results['optimizer_comparison'] = optimizer_comp
    
    # ========================================================================
    # BASELINE: Logistic Regression
    # ========================================================================
    logreg_results = train_logistic_regression(
        X_train, y_train, X_test, y_test, seed=SEED
    )
    all_results['logistic_regression'] = logreg_results
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Best Adam (Grid Search): {grid_search_adam['best_result']['test_acc']:.4f}")
    print(f"Best RMSprop (Grid Search): {grid_search_rmsprop['best_result']['test_acc']:.4f}")
    print(f"Best Architecture (Adam): {arch_adam['results'][0]['test_acc']:.4f}")
    print(f"Best Architecture (RMSprop): {arch_rmsprop['results'][0]['test_acc']:.4f}")
    print(f"Logistic Regression: {logreg_results['test_acc']:.4f}")
    print(f"{'='*80}")
    
    # Save results
    if save_results_flag:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(all_results, f'classification_results_{timestamp}.pkl')
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


# ============================================================================
# PLOTTING FUNCTION FOR ALL RESULTS
# ============================================================================

def create_all_plots(results, output_dir='figures'):
    """
    Create all plots from experiment results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all experiment results
    output_dir : str
        Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Heatmap for Adam grid search
    fig1 = plot_hyperparameter_heatmap(
        results['grid_search_adam']['results_grid'],
        results['grid_search_adam']['eta_values'],
        results['grid_search_adam']['lambda_values'],
        title_suffix=' - Adam Optimizer'
    )
    fig1.savefig(output_path / 'heatmap_adam.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Heatmap for RMSprop grid search
    fig2 = plot_hyperparameter_heatmap(
        results['grid_search_rmsprop']['results_grid'],
        results['grid_search_rmsprop']['eta_values'],
        results['grid_search_rmsprop']['lambda_values'],
        title_suffix=' - RMSprop Optimizer'
    )
    fig2.savefig(output_path / 'heatmap_rmsprop.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Architecture comparison - Adam
    fig3 = plot_architecture_comparison(
        results['architecture_adam']['results'][:10]  # Top 10
    )
    fig3.savefig(output_path / 'architecture_adam.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Architecture comparison - RMSprop
    fig4 = plot_architecture_comparison(
        results['architecture_rmsprop']['results'][:10]  # Top 10
    )
    fig4.savefig(output_path / 'architecture_rmsprop.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # Depth analysis - Adam
    fig5 = plot_depth_analysis(
        results['depth_adam']['results'],
        results['depth_adam']['depths']
    )
    fig5.savefig(output_path / 'depth_adam.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # Depth analysis - RMSprop
    fig6 = plot_depth_analysis(
        results['depth_rmsprop']['results'],
        results['depth_rmsprop']['depths']
    )
    fig6.savefig(output_path / 'depth_rmsprop.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print(f"All plots saved to {output_path}/")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments(save_results_flag=True, n_jobs=-1)
    
    # Create plots
    create_all_plots(results)
    
    print("\nAll experiments completed!")