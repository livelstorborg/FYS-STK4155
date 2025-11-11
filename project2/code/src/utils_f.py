import sys
import concurrent.futures
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import product
import time

from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from src.losses import CrossEntropy
from src.optimizers import Adam, RMSprop
from src.training import train
from src.metrics import accuracy
from src.utils import load_mnist_data
from src.plotting import (
    plot_lambda_eta_heatmaps,
    plot_architecture_comparison_heatmaps,
    plot_confusion_matrix,
)


def create_network_architecture(n_layers, n_nodes, activation):
    """Create network architecture specification"""
    if activation == "ReLU":
        activation_fn = ReLU()
    elif activation == "LeakyReLU":
        activation_fn = LeakyReLU()
    elif activation == "Sigmoid":
        activation_fn = Sigmoid()
    layer_sizes = [n_nodes] * n_layers + [10]
    activations = [activation_fn] * n_layers + [Softmax()]
    return layer_sizes, activations


def train_and_evaluate_wrapper(args):
    """Unpacks arguments and calls the main training function for parallel execution."""
    (
        X_train,
        y_train,
        y_train_encoded,
        X_val,
        y_val,
        y_val_encoded,
        n_layers,
        n_nodes,
        lambda_reg,
        eta,
        batch_size,
        verbose,
        activation,
        optimizer_name,
        epochs,
        patience,
        seed,
    ) = args

    val_accuracy, train_accuracy, model = train_and_evaluate(
        X_train,
        y_train,
        y_train_encoded,
        X_val,
        y_val,
        y_val_encoded,
        n_layers,
        n_nodes,
        lambda_reg,
        eta,
        batch_size,
        verbose,
        activation,
        optimizer_name,
        epochs,
        patience,
        seed,
    )
    return lambda_reg, eta, val_accuracy, train_accuracy


def train_and_evaluate(
    X_train,
    y_train,
    y_train_encoded,
    X_val,
    y_val,
    y_val_encoded,
    n_layers,
    n_nodes,
    lambda_reg,
    eta,
    batch_size=100,
    verbose=False,
    activation="LeakyReLU",
    optimizer_name="RMSprop",
    epochs=100,
    patience=10,
    seed=42,
):
    """
    Train a single model configuration and return validation accuracy

    Parameters:
    -----------
    X_train, y_train, y_train_encoded : array
        Training data (y_train for calculating train accuracy)
    X_val, y_val, y_val_encoded : array
        Validation data
    n_layers : int
        Number of hidden layers
    n_nodes : int
        Nodes per hidden layer
    lambda_reg : float
        L2 regularization parameter
    eta : float
        Learning rate
    batch_size : int
        Batch size for training
    verbose : bool
        Print training progress
    activation : str
        Activation function name
    optimizer_name : str
        Optimizer name ('Adam' or 'RMSprop')
    epochs : int
        Number of training epochs
    patience : int
        Early stopping patience
    seed : int
        Random seed

    Returns:
    --------
    val_accuracy : float
        Validation accuracy
    train_accuracy : float
        Training accuracy
    model : NeuralNetwork
        Trained model
    """
    # Create network
    layer_sizes, activations = create_network_architecture(
        n_layers, n_nodes, activation
    )

    model = NeuralNetwork(
        network_input_size=784,
        layer_output_sizes=layer_sizes,
        activations=activations,
        loss=CrossEntropy(),
        seed=seed,
        lambda_reg=lambda_reg,
        reg_type="l2" if lambda_reg > 0 else None,
    )

    if optimizer_name == "Adam":
        optimizer = Adam(eta=eta)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(eta=eta)

    train(
        nn=model,
        X_train=X_train,
        y_train=y_train_encoded,
        X_val=X_val,
        y_val=y_val_encoded,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        stochastic=True,
        task="classification",
        early_stopping=True,
        patience=patience,
        verbose=verbose,
        seed=seed,
    )

    # Evaluate on validation set
    y_val_pred_proba = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    val_accuracy = accuracy(y_val, y_val_pred)

    # Evaluate on training set
    y_train_pred_proba = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    train_accuracy = accuracy(y_train, y_train_pred)

    return val_accuracy, train_accuracy, model


def grid_search_lambda_eta(
    X_train,
    y_train,
    y_train_encoded,
    X_val,
    y_val,
    y_val_encoded,
    n_layers,
    n_nodes,
    lambda_values,
    eta_values,
    batch_size=100,
    activation="LeakyReLU",
    optimizer_name="RMSprop",
    epochs=100,
    patience=10,
    seed=42,
):
    """
    Grid search over lambda and eta for a specific network architecture with parallel execution

    Returns:
    --------
    best_lambda : float
        Best regularization parameter
    best_eta : float
        Best learning rate
    best_accuracy : float
        Best validation accuracy achieved
    best_train_accuracy : float
        Training accuracy corresponding to the best validation accuracy
    grid_results : dict
        Full grid search results with accuracy for each lambda-eta combination
    """
    print(f"\n{'=' * 70}")
    print(f"Grid search for: {n_layers} layers, {n_nodes} nodes per layer")
    print(f"{'=' * 70}")

    best_accuracy = 0.0
    best_train_accuracy = 0.0
    best_lambda = None
    best_eta = None

    # Store full grid results
    grid_results = {
        "lambda_values": [],
        "eta_values": [],
        "val_accuracies": [],
        "train_accuracies": [],
    }

    # Prepare argument list for parallel execution
    tasks = []
    for lambda_val, eta_val in product(lambda_values, eta_values):
        tasks.append(
            (
                X_train,
                y_train,
                y_train_encoded,
                X_val,
                y_val,
                y_val_encoded,
                n_layers,
                n_nodes,
                lambda_val,
                eta_val,
                batch_size,
                False,
                activation,
                optimizer_name,
                epochs,
                patience,
                seed,
            )
        )

    total_combos = len(tasks)

    print(f"Starting parallel training of {total_combos} models...")

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_and_evaluate_wrapper, task) for task in tasks]

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                lambda_val, eta_val, val_accuracy_val, train_accuracy_val = (
                    future.result()
                )

                # Store results
                grid_results["lambda_values"].append(lambda_val)
                grid_results["eta_values"].append(eta_val)
                grid_results["val_accuracies"].append(val_accuracy_val)
                grid_results["train_accuracies"].append(train_accuracy_val)

                if val_accuracy_val > best_accuracy:
                    best_accuracy = val_accuracy_val
                    best_train_accuracy = train_accuracy_val
                    best_lambda = lambda_val
                    best_eta = eta_val

                # Progress update
                if i % 5 == 0 or i == total_combos:
                    print(
                        f"  Progress: {i}/{total_combos} | "
                        f"Current Best: λ={best_lambda:.2e}, η={best_eta:.2e}, val_acc={best_accuracy:.4f}"
                    )

            except Exception as e:
                print(f"  Warning: A training run failed in the pool: {str(e)}")

    print(f"\nBest configuration (based on Validation Accuracy):")
    if best_lambda is not None and best_eta is not None:
        print(f"  λ = {best_lambda:.2e}")
        print(f"  η = {best_eta:.2e}")
        print(f"  Validation accuracy = {best_accuracy:.4f}")
        print(f"  Training accuracy = {best_train_accuracy:.4f}")
    else:
        print(f"  WARNING: No successful configuration found!")
        print(f"  All {len(lambda_values) * len(eta_values)} attempts failed.")

    return best_lambda, best_eta, best_accuracy, best_train_accuracy, grid_results


def run_architecture_comparison(
    X_train,
    y_train,
    y_train_encoded,
    X_val,
    y_val,
    y_val_encoded,
    n_layers_list,
    n_nodes_list,
    lambda_values,
    eta_values,
    batch_size=100,
    activation="LeakyReLU",
    optimizer_name="RMSprop",
    epochs=100,
    patience=10,
    seed=42,
):
    """
    Compare different network architectures

    For each architecture (n_layers, n_nodes):
    1. Find optimal lambda and eta via grid search
    2. Record best validation accuracy and corresponding training accuracy

    Returns:
    --------
    results : dict
        Dictionary with results for each architecture
    """
    results = {
        "n_layers": [],
        "n_nodes": [],
        "best_lambda": [],
        "best_eta": [],
        "val_accuracy": [],
        "train_accuracy": [],
        "grid_results": [],
    }

    total_configs = len(n_layers_list) * len(n_nodes_list)
    print(f"\n{'=' * 70}")
    print(f"ARCHITECTURE COMPARISON: {total_configs} configurations")
    print(f"{'=' * 70}")
    print(f"Lambda values: {len(lambda_values)}")
    print(f"Eta values: {len(eta_values)}")
    print(f"Total models per architecture: {len(lambda_values) * len(eta_values)}")
    print(
        f"Total models overall: {total_configs * len(lambda_values) * len(eta_values)}"
    )
    print(f"{'=' * 70}\n")

    start_time = time.time()

    for config_num, (n_layers, n_nodes) in enumerate(
        product(n_layers_list, n_nodes_list), 1
    ):
        print(
            f"\n[{config_num}/{total_configs}] Architecture: {n_layers} layers × {n_nodes} nodes"
        )

        # Find optimal hyperparameters for this architecture
        best_lambda, best_eta, best_val_accuracy, best_train_accuracy, grid_results = (
            grid_search_lambda_eta(
                X_train,
                y_train,
                y_train_encoded,
                X_val,
                y_val,
                y_val_encoded,
                n_layers,
                n_nodes,
                lambda_values,
                eta_values,
                batch_size,
                activation,
                optimizer_name,
                epochs,
                patience,
                seed,
            )
        )

        # Store results
        results["n_layers"].append(n_layers)
        results["n_nodes"].append(n_nodes)
        results["best_lambda"].append(best_lambda)
        results["best_eta"].append(best_eta)
        results["val_accuracy"].append(best_val_accuracy)
        results["train_accuracy"].append(best_train_accuracy)
        results["grid_results"].append(grid_results)

    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Grid search complete! Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"{'=' * 70}\n")

    return results


def print_stage1_results_table(results):
    """Print Stage 1 results in a formatted table"""
    print("\n" + "=" * 110)
    print("STAGE 1: RESULTS SUMMARY")
    print("=" * 110)
    print(
        f"{'#':<4} {'Layers':<8} {'Nodes':<8} {'Best λ':<12} {'Best η':<12} {'Val Accuracy':<15} {'Train Accuracy':<15} {'Rank':<6}"
    )
    print("-" * 110)

    # Sort by accuracy
    sorted_indices = np.argsort(results["val_accuracy"])[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        marker = "★" if rank == 1 else " "
        print(
            f"{marker} {idx + 1:<2} {results['n_layers'][idx]:<8} "
            f"{results['n_nodes'][idx]:<8} {results['best_lambda'][idx]:<12.2e} "
            f"{results['best_eta'][idx]:<12.2e} {results['val_accuracy'][idx]:<15.4f} "
            f"{results['train_accuracy'][idx]:<15.4f} {rank:<6}"
        )

    print("=" * 110)

    # Best configuration
    best_idx = sorted_indices[0]
    print(f"\n★ BEST CONFIGURATION:")
    print(
        f"  Architecture: {results['n_layers'][best_idx]} layer(s) × {results['n_nodes'][best_idx]} nodes"
    )
    print(f"  Regularization: λ = {results['best_lambda'][best_idx]:.2e}")
    print(f"  Learning rate: η = {results['best_eta'][best_idx]:.2e}")
    print(f"  Validation accuracy: {results['val_accuracy'][best_idx]:.4f}")
    print(f"  Training accuracy: {results['train_accuracy'][best_idx]:.4f}")
    print("=" * 110 + "\n")

    return best_idx


def stage2_architecture_comparison(
    stage1_results,
    X_train,
    y_train,
    y_train_encoded,
    X_val,
    y_val,
    y_val_encoded,
    X_test,
    y_test,
    y_test_encoded,
    activation="LeakyReLU",
    optimizer_name="RMSprop",
    epochs=100,
    batch_size=256,
    patience=10,
    seed=42,
):
    """
    Stage 2: Train all architectures with their optimal hyperparameters
    on the full dataset and evaluate on train/val/test

    Returns:
    --------
    stage2_results : dict
        Dictionary with train/val/test accuracies for each architecture
    """
    print("\n" + "=" * 70)
    print("STAGE 2: ARCHITECTURE COMPARISON ON FULL DATASET")
    print("=" * 70)
    print(
        f"Training all {len(stage1_results['n_layers'])} architectures with optimal hyperparameters..."
    )
    print(f"  Dataset size: {len(X_train)} training samples")
    print("=" * 70 + "\n")

    stage2_results = {
        "n_layers": [],
        "n_nodes": [],
        "lambda": [],
        "eta": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
    }

    for idx in range(len(stage1_results["n_layers"])):
        n_layers = stage1_results["n_layers"][idx]
        n_nodes = stage1_results["n_nodes"][idx]
        best_lambda = stage1_results["best_lambda"][idx]
        best_eta = stage1_results["best_eta"][idx]

        print(
            f"\n[{idx + 1}/{len(stage1_results['n_layers'])}] Training: {n_layers} layer(s) × {n_nodes} nodes"
        )
        print(f"  Using: λ={best_lambda:.2e}, η={best_eta:.2e}")

        # Create and train model
        layer_sizes, activations = create_network_architecture(
            n_layers, n_nodes, activation
        )

        model = NeuralNetwork(
            network_input_size=784,
            layer_output_sizes=layer_sizes,
            activations=activations,
            loss=CrossEntropy(),
            seed=seed,
            lambda_reg=best_lambda,
            reg_type="l2" if best_lambda > 0 else None,
        )

        if optimizer_name == "Adam":
            optimizer = Adam(eta=best_eta)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(eta=best_eta)

        train(
            nn=model,
            X_train=X_train,
            y_train=y_train_encoded,
            X_val=X_val,
            y_val=y_val_encoded,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            stochastic=True,
            task="classification",
            early_stopping=True,
            patience=patience,
            verbose=False,
            seed=seed,
        )

        # Evaluate on all sets
        y_train_pred_proba = model.predict(X_train)
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)
        train_acc = accuracy(y_train, y_train_pred)

        y_val_pred_proba = model.predict(X_val)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
        val_acc = accuracy(y_val, y_val_pred)

        y_test_pred_proba = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        test_acc = accuracy(y_test, y_test_pred)

        # Store results
        stage2_results["n_layers"].append(n_layers)
        stage2_results["n_nodes"].append(n_nodes)
        stage2_results["lambda"].append(best_lambda)
        stage2_results["eta"].append(best_eta)
        stage2_results["train_accuracy"].append(train_acc)
        stage2_results["val_accuracy"].append(val_acc)
        stage2_results["test_accuracy"].append(test_acc)

        print(
            f"  ✓ Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

    print("\n" + "=" * 70)
    print("STAGE 2: ARCHITECTURE COMPARISON COMPLETE")
    print("=" * 70)

    return stage2_results


def print_stage2_results_table(stage2_results):
    """Print Stage 2 architecture comparison results with train/val/test accuracies"""
    print("\n" + "=" * 130)
    print("STAGE 2: ARCHITECTURE COMPARISON RESULTS (Full Dataset)")
    print("=" * 130)
    print(
        f"{'#':<4} {'Layers':<8} {'Nodes':<8} {'λ':<12} {'η':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Rank':<6}"
    )
    print("-" * 130)

    # Sort by validation accuracy
    sorted_indices = np.argsort(stage2_results["val_accuracy"])[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        marker = "★" if rank == 1 else " "
        print(
            f"{marker} {idx + 1:<2} {stage2_results['n_layers'][idx]:<8} "
            f"{stage2_results['n_nodes'][idx]:<8} {stage2_results['lambda'][idx]:<12.2e} "
            f"{stage2_results['eta'][idx]:<12.2e} {stage2_results['train_accuracy'][idx]:<12.4f} "
            f"{stage2_results['val_accuracy'][idx]:<12.4f} {stage2_results['test_accuracy'][idx]:<12.4f} {rank:<6}"
        )

    print("=" * 130)

    # Best configuration
    best_idx = sorted_indices[0]
    print(f"\n★ BEST ARCHITECTURE (Stage 2):")
    print(
        f"  Configuration: {stage2_results['n_layers'][best_idx]} layer(s) × {stage2_results['n_nodes'][best_idx]} nodes"
    )
    print(
        f"  Hyperparameters: λ = {stage2_results['lambda'][best_idx]:.2e}, η = {stage2_results['eta'][best_idx]:.2e}"
    )
    print(f"  Train Accuracy: {stage2_results['train_accuracy'][best_idx]:.4f}")
    print(f"  Validation Accuracy: {stage2_results['val_accuracy'][best_idx]:.4f}")
    print(f"  Test Accuracy: {stage2_results['test_accuracy'][best_idx]:.4f}")
    print("=" * 130 + "\n")

    return best_idx


def stage3_best_model_evaluation(
    best_config,
    X_train_full,
    y_train_full,
    y_train_full_encoded,
    X_val_full,
    y_val_full,
    y_val_full_encoded,
    X_test_full,
    y_test_full,
    y_test_full_encoded,
    activation="LeakyReLU",
    optimizer_name="RMSprop",
    epochs=100,
    batch_size=256,
    patience=10,
    seed=42,
):
    """
    Stage 3: Train best model on full dataset and evaluate (for final comparison and confusion matrix)
    """
    print("\n" + "=" * 70)
    print("STAGE 3: BEST MODEL FINAL EVALUATION")
    print("=" * 70)
    print(f"Training best configuration for final evaluation...")
    print(
        f"  Architecture: {best_config['n_layers']} layer(s) × {best_config['n_nodes']} nodes"
    )
    print(f"  Regularization: $\lambda$ = {best_config['lambda']:.2e}")
    print(f"  Learning rate: $\eta$ = {best_config['eta']:.2e}")
    print(f"  Dataset size: {len(X_train_full)} training samples")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patience: {patience}")
    print("=" * 70 + "\n")

    # Train model
    layer_sizes, activations = create_network_architecture(
        best_config["n_layers"], best_config["n_nodes"], activation
    )

    model = NeuralNetwork(
        network_input_size=784,
        layer_output_sizes=layer_sizes,
        activations=activations,
        loss=CrossEntropy(),
        seed=seed,
        lambda_reg=best_config["lambda"],
        reg_type="l2" if best_config["lambda"] > 0 else None,
    )

    if optimizer_name == "Adam":
        optimizer = Adam(eta=best_config["eta"])
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(eta=best_config["eta"])

    print("Training neural network...")
    train(
        nn=model,
        X_train=X_train_full,
        y_train=y_train_full_encoded,
        X_val=X_val_full,
        y_val=y_val_full_encoded,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        stochastic=True,
        task="classification",
        early_stopping=True,
        patience=patience,
        verbose=True,
        seed=seed,
    )

    # Evaluate on validation and test sets
    print("\nEvaluating...")

    # Validation set
    y_val_pred_proba = model.predict(X_val_full)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    val_accuracy = accuracy(y_val_full, y_val_pred)

    # Test set
    y_test_pred_proba = model.predict(X_test_full)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    test_accuracy = accuracy(y_test_full, y_test_pred)

    print(f"\n{'=' * 70}")
    print(f"NEURAL NETWORK RESULTS (Full Dataset)")
    print(f"{'=' * 70}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"{'=' * 70}\n")

    return model, y_test_pred, val_accuracy, test_accuracy


def compare_with_sklearn(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Compare neural network with sklearn LogisticRegression"""
    print("\n" + "=" * 70)
    print("COMPARISON WITH SKLEARN LOGISTIC REGRESSION")
    print("=" * 70)

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        solver="saga",
        multi_class="multinomial",
        max_iter=100,
        random_state=seed,
        verbose=1,
    )

    lr_model.fit(X_train, y_train)

    # Predictions
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n{'=' * 70}")
    print(f"LOGISTIC REGRESSION RESULTS")
    print(f"{'=' * 70}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"{'=' * 70}\n")

    return val_acc, test_acc


def main_f(
    # Data parameters
    seed=42,
    use_subset_for_grid_search=True,
    subset_size_for_grid_search=10000,
    use_full_dataset_for_stage2=True,
    # Training parameters
    epochs=100,
    batch_size_stage1=100,
    batch_size_stage2=256,
    patience=10,
    # Hyperparameter search space
    lambda_values=None,
    eta_values=None,
    n_layers=None,
    n_nodes=None,
    activation="LeakyReLU",
    optimizer="RMSprop",
):
    """
    Main orchestrator function for Part F experiments (MNIST Classification)

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    use_subset_for_grid_search : bool
        Use subset for Stage 1 grid search
    subset_size_for_grid_search : int
        Size of subset for grid search
    use_full_dataset_for_stage2 : bool
        Use full dataset for Stage 2 and 3
    epochs : int
        Number of training epochs
    batch_size_stage1 : int
        Batch size for Stage 1 (grid search)
    batch_size_stage2 : int
        Batch size for Stage 2 and 3
    patience : int
        Early stopping patience
    lambda_values : array-like or None
        L2 regularization values. If None, defaults to logspace(-5, -1, 5)
    eta_values : array-like or None
        Learning rate values. If None, defaults to logspace(-5, -1, 5)
    n_layers : list or None
        Number of hidden layers to try. If None, defaults to [1, 2]
    n_nodes : list or None
        Nodes per hidden layer to try. If None, defaults to [100, 150, 200]
    activation : str
        Activation function: 'Sigmoid', 'ReLU', or 'LeakyReLU'
    optimizer : str
        Optimizer: 'Adam' or 'RMSprop'

    Returns:
    --------
    stage1_results : dict
        Results from Stage 1 (lambda-eta grid search)
    stage2_results : dict or None
        Results from Stage 2 (architecture comparison on full dataset)
    best_config : dict or None
        Best configuration from Stage 2
    """
    # Set defaults
    if lambda_values is None:
        lambda_values = np.logspace(-5, -1, 5)
    if eta_values is None:
        eta_values = np.logspace(-5, -1, 5)
    if n_layers is None:
        n_layers = [1, 2]
    if n_nodes is None:
        n_nodes = [100, 150, 200]

    # Set random seed
    np.random.seed(seed)

    print("=" * 70)
    print("MNIST CLASSIFICATION - THREE-STAGE ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Activation: {activation}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Architectures: {n_layers} layers × {n_nodes} nodes")
    print(f"  Lambda values: {len(lambda_values)}")
    print(f"  Eta values: {len(eta_values)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size (Stage 1): {batch_size_stage1}")
    print(f"  Batch size (Stage 2&3): {batch_size_stage2}")
    print(f"  Patience: {patience}")

    print(f"\nStage 1: Lambda-Eta Grid Search on {subset_size_for_grid_search} samples")
    print(f"  Total configurations: {len(n_layers) * len(n_nodes)}")
    print(
        f"  Total model trainings: {len(n_layers) * len(n_nodes) * len(lambda_values) * len(eta_values)}"
    )

    if use_full_dataset_for_stage2:
        print(f"\nStage 2: Architecture Comparison on full dataset (70,000 samples)")
        print(f"\nStage 3: Best Model Final Evaluation on full dataset")
    else:
        print(f"\nStage 2 & 3: Disabled")

    print("=" * 70 + "\n")

    # ========================================================================
    # STAGE 1: LAMBDA-ETA GRID SEARCH
    # ========================================================================

    print("\n" + "=" * 70)
    print("STAGE 1: LAMBDA-ETA GRID SEARCH")
    print("=" * 70)

    (
        X_train_stage1,
        y_train_stage1,
        y_train_encoded_stage1,
        X_val_stage1,
        y_val_stage1,
        y_val_encoded_stage1,
        X_test_stage1,
        y_test_stage1,
        y_test_encoded_stage1,
        scaler,
    ) = load_mnist_data(
        use_subset=use_subset_for_grid_search,
        subset_size=subset_size_for_grid_search,
        seed=seed,
    )

    # Run grid search to find best hyperparameters for each architecture
    stage1_results = run_architecture_comparison(
        X_train_stage1,
        y_train_stage1,
        y_train_encoded_stage1,
        X_val_stage1,
        y_val_stage1,
        y_val_encoded_stage1,
        n_layers,
        n_nodes,
        lambda_values,
        eta_values,
        batch_size_stage1,
        activation,
        optimizer,
        epochs,
        patience,
        seed,
    )

    # Print Stage 1 results table
    print_stage1_results_table(stage1_results)

    # Plot lambda-eta heatmaps (combined subplot only)
    print("\nGenerating lambda-eta grid search heatmaps...")
    plot_lambda_eta_heatmaps(
        stage1_results, lambda_values, eta_values, activation, optimizer
    )

    # ========================================================================
    # STAGE 2: ARCHITECTURE COMPARISON ON FULL DATASET
    # ========================================================================

    stage2_results = None
    best_config = None

    if use_full_dataset_for_stage2:
        print("\n" + "=" * 70)
        print("STAGE 2: LOADING DATA FOR ARCHITECTURE COMPARISON")
        print("=" * 70)

        # Load full dataset for Stage 2
        (
            X_train,
            y_train,
            y_train_encoded,
            X_val,
            y_val,
            y_val_encoded,
            X_test,
            y_test,
            y_test_encoded,
            _,
        ) = load_mnist_data(use_subset=False, seed=seed)

        # Train all architectures with their optimal hyperparameters
        stage2_results = stage2_architecture_comparison(
            stage1_results,
            X_train,
            y_train,
            y_train_encoded,
            X_val,
            y_val,
            y_val_encoded,
            X_test,
            y_test,
            y_test_encoded,
            activation,
            optimizer,
            epochs,
            batch_size_stage2,
            patience,
            seed,
        )

        # Print Stage 2 results table with train/val/test
        best_idx = print_stage2_results_table(stage2_results)

        # Plot architecture comparison heatmaps (using Stage 2 results)
        print("\nGenerating architecture comparison heatmaps (Train vs Val)...")
        plot_architecture_comparison_heatmaps(stage2_results, activation, optimizer)

        # Extract best configuration from Stage 2
        best_config = {
            "n_layers": stage2_results["n_layers"][best_idx],
            "n_nodes": stage2_results["n_nodes"][best_idx],
            "lambda": stage2_results["lambda"][best_idx],
            "eta": stage2_results["eta"][best_idx],
        }

        # ========================================================================
        # STAGE 3: BEST MODEL FINAL EVALUATION (with sklearn comparison)
        # ========================================================================

        # Train best model one more time for confusion matrix and sklearn comparison
        nn_model, y_test_pred_nn, nn_val_acc, nn_test_acc = (
            stage3_best_model_evaluation(
                best_config,
                X_train,
                y_train,
                y_train_encoded,
                X_val,
                y_val,
                y_val_encoded,
                X_test,
                y_test,
                y_test_encoded,
                activation,
                optimizer,
                epochs,
                batch_size_stage2,
                patience,
                seed,
            )
        )

        # Confusion matrix for best neural network
        print("\nGenerating confusion matrix for best model...")
        plot_confusion_matrix(
            y_test,
            y_test_pred_nn,
            activation,
            optimizer,
            n_layers=best_config["n_layers"],
            n_nodes=best_config["n_nodes"],
        )

        # Compare with sklearn
        sklearn_val_acc, sklearn_test_acc = compare_with_sklearn(
            X_train, y_train, X_val, y_val, X_test, y_test, seed
        )

        # Final comparison
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"{'Model':<30} {'Val Accuracy':<15} {'Test Accuracy':<15}")
        print("-" * 70)
        print(f"{'Neural Network (Ours)':<30} {nn_val_acc:<15.4f} {nn_test_acc:<15.4f}")
        print(
            f"{'Logistic Regression (sklearn)':<30} {sklearn_val_acc:<15.4f} {sklearn_test_acc:<15.4f}"
        )
        print("-" * 70)
        print(
            f"{'Difference (NN - LR)':<30} {nn_val_acc - sklearn_val_acc:<15.4f} {nn_test_acc - sklearn_test_acc:<15.4f}"
        )
        print("=" * 70)

        if nn_test_acc > sklearn_test_acc:
            print(
                f"\n✓ Our neural network outperforms sklearn by {(nn_test_acc - sklearn_test_acc) * 100:.2f}%"
            )
        else:
            print(
                f"\n✓ sklearn outperforms our neural network by {(sklearn_test_acc - nn_test_acc) * 100:.2f}%"
            )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files (all PDFs in figs/ folder):")
    print(f"  - figs/lambda_eta_heatmaps_all_{activation}_{optimizer}.pdf (Stage 1)")
    if use_full_dataset_for_stage2:
        print(
            f"  - figs/architecture_comparison_heatmaps_{activation}_{optimizer}.pdf (Stage 2)"
        )
        print(f"  - figs/confusion_matrix_{activation}_{optimizer}.pdf (Stage 3)")
    print("=" * 70 + "\n")

    return stage1_results, stage2_results, best_config


# ============================================================================
#                           EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Add multiprocessing protection
    if sys.platform.startswith("win"):
        main_f(
            seed=42,
            use_subset_for_grid_search=True,
            subset_size_for_grid_search=10000,
            use_full_dataset_for_stage2=True,
            epochs=100,
            batch_size_stage1=100,
            batch_size_stage2=256,
            patience=10,
            n_layers=[1, 2],
            n_nodes=[100, 150, 200],
            activation="LeakyReLU",
            optimizer="RMSprop",
        )
    else:
        main_f(
            seed=42,
            use_subset_for_grid_search=True,
            subset_size_for_grid_search=10000,
            use_full_dataset_for_stage2=True,
            epochs=100,
            batch_size_stage1=100,
            batch_size_stage2=256,
            patience=10,
            n_layers=[1, 2],
            n_nodes=[100, 150, 200],
            activation="LeakyReLU",
            optimizer="RMSprop",
        )
