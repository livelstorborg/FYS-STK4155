import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


from src.neural_network import NeuralNetwork
from src.activations import LeakyReLU, ReLU, Sigmoid, Linear
from src.losses import MSE
from src.optimizers import GD, RMSprop, Adam
from src.training import train
from src.metrics import mse
from src.utils import runge, polynomial_features, scale_data, Ridge_parameters, split_scale_data, generate_data
from src.plotting import lambda_eta_heatmap, plot_combined_learning_curves



def ridge_example(sigma, datasize, seed):
    """Calculates Ridge scaled MSE metrics on a dataset with P=14."""
    np.random.seed(seed)  # Set seed for noise generation
    x = np.linspace(-1, 1, datasize)
    y_true = runge(x)
    y_noise = y_true + np.random.normal(0, sigma, datasize)

    X_poly = polynomial_features(x, p=14, intercept=False)

    test_ratio = 0.2
    val_ratio = 0.2
    train_ratio = 1 - test_ratio - val_ratio

    X_temp, X_test_poly, y_temp, y_test = train_test_split(
        X_poly, y_noise, test_size=test_ratio, random_state=seed
    )
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train_poly, X_val_poly, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=seed
    )

    # Use consistent scaling with y_std
    X_train_s, y_train_s, X_mean, X_std, y_mean, y_std = scale_data(
        X_train_poly, y_train.reshape(-1, 1)
    )
    X_val_s, y_val_s, _, _, _, _ = scale_data(
        X_val_poly, y_val.reshape(-1, 1), X_mean, X_std, y_mean, y_std
    )

    # Use lambda=0.01 for Ridge benchmark
    theta_ridge = Ridge_parameters(X_train_s, y_train_s, lam=0.01)
    y_pred_ridge_val = X_val_s @ theta_ridge

    ridge_val_mse = mse(y_val_s.reshape(-1, 1), y_pred_ridge_val.reshape(-1, 1))

    print(f"Ridge Benchmark (N={datasize}, P=14, Lambda=0.01, Seed={seed}): Val MSE (Scaled) = {ridge_val_mse:.6f}")

    return ridge_val_mse.flatten()[0] if ridge_val_mse.ndim > 1 else ridge_val_mse


def lasso_example(sigma, datasize, seed):
    """Calculates Lasso scaled Validation MSE as a benchmark (P=15, alpha=0.01)."""
    np.random.seed(seed)  # Set seed for noise generation
    x = np.linspace(-1, 1, datasize)
    y_true = runge(x)
    y_noise = y_true + np.random.normal(0, sigma, datasize)

    X_poly = polynomial_features(x, p=15, intercept=False)

    test_ratio = 0.2
    val_ratio = 0.2
    train_ratio = 1 - test_ratio - val_ratio

    X_temp, X_test_poly, y_temp, y_test = train_test_split(
        X_poly, y_noise, test_size=test_ratio, random_state=seed
    )
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train_poly, X_val_poly, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=seed
    )

    # Use consistent scaling with y_std
    X_train_s, y_train_s, X_mean, X_std, y_mean, y_std = scale_data(
        X_train_poly, y_train.reshape(-1, 1)
    )
    X_val_s, y_val_s, _, _, _, _ = scale_data(
        X_val_poly, y_val.reshape(-1, 1), X_mean, X_std, y_mean, y_std
    )

    lasso_model = Lasso(alpha=0.01, max_iter=10000, random_state=seed)
    lasso_model.fit(X_train_s, y_train_s.ravel())

    y_pred_lasso_val = lasso_model.predict(X_val_s)
    lasso_val_mse = mse(y_val_s.reshape(-1, 1), y_pred_lasso_val.reshape(-1, 1))

    print(f"Lasso Benchmark (N={datasize}, P=15, alpha=0.01, Seed={seed}): Val MSE (Scaled) = {lasso_val_mse:.6f}")

    return lasso_val_mse.flatten()[0] if lasso_val_mse.ndim > 1 else lasso_val_mse


def find_best_lambda_eta(config):
    """
    Find best learning rate (eta) and regularization strength (lambda).
    Model weights are NOT saved; the final state after early stopping is used.
    Each trial uses a DIFFERENT data split.
    """
    layer_sizes = config["layer_sizes"]
    activations = config["activations"]
    optimizer_class = config["optimizer_class"]
    reg_type = config["reg_type"]
    
    # Get raw data and split ratios
    X_raw = config["X_raw"]
    y_raw = config["y_raw"]
    TRAIN_SIZE = config["TRAIN_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]

    mse_grid = np.zeros((len(config["LAMBDA_VALS"]), len(config["ETA_VALS"])))

    optimizer_name = optimizer_class.__name__
    activation_name = config["activation_name"]

    print(
        f"\n--- Starting {optimizer_name} LR/Lambda Search ({activation_name}, {reg_type}, {config['LR_TRIALS']} trials with different splits) ---"
    )

    # Track the best model's final performance after search (for the Test MSE printout)
    best_val_mse_overall = np.inf
    final_test_mse_at_best_params = None

    for trial in range(config["LR_TRIALS"]):
        trial_seed = config["SEED"] + trial
        
        # Create NEW data split for this trial
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, _, _, _, _, _, _, _ = (
            split_scale_data(X_raw, y_raw, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=trial_seed)
        )
        NETWORK_INPUT_SIZE = X_train_scaled.shape[1]

        for i, lam in enumerate(config["LAMBDA_VALS"]):
            for j, eta in enumerate(config["ETA_VALS"]):

                # Model is initialized FRESH for every hyperparameter combination
                model = NeuralNetwork(
                    network_input_size=NETWORK_INPUT_SIZE,
                    layer_output_sizes=layer_sizes,
                    activations=activations,
                    loss=MSE(),
                    seed=trial_seed,
                    lambda_reg=lam,
                    reg_type=reg_type,
                )

                is_stochastic = optimizer_class is not GD
                batch_size_setting = (
                    config["BATCH_SIZE"] if is_stochastic else len(X_train_scaled)
                )

                # Train with early stopping (using this trial's split)
                train(
                    nn=model,
                    X_train=X_train_scaled,
                    y_train=y_train_scaled,
                    X_val=X_val_scaled,
                    y_val=y_val_scaled,
                    optimizer=optimizer_class(eta=eta),
                    epochs=config["EPOCHS_LR_SEARCH"],
                    batch_size=batch_size_setting,
                    stochastic=is_stochastic,
                    task="regression",
                    early_stopping=True,
                    patience=150,
                    verbose=False,
                    seed=trial_seed,
                )

                # Use the model's final state (after early stopping) on this trial's split
                y_val_pred_scaled = model.predict(X_val_scaled)
                val_mse_scaled = mse(y_val_scaled, y_val_pred_scaled)
                mse_grid[i, j] += val_mse_scaled

                # Track the best test MSE based on the final model state in the grid search
                if val_mse_scaled < best_val_mse_overall:
                    best_val_mse_overall = val_mse_scaled
                    y_test_pred_scaled = model.predict(X_test_scaled)
                    final_test_mse_at_best_params = mse(
                        y_test_scaled, y_test_pred_scaled
                    )

    avg_mse_grid = mse_grid / config["LR_TRIALS"]

    best_idx = np.unravel_index(np.argmin(avg_mse_grid), avg_mse_grid.shape)
    best_lam = config["LAMBDA_VALS"][best_idx[0]]
    best_eta = config["ETA_VALS"][best_idx[1]]
    best_avg_val_mse = avg_mse_grid[best_idx]

    print(
        f"Best (Avg Grid): eta={best_eta:.2e}, lambda={best_lam:.2e}, Val MSE={best_avg_val_mse:.6f}"
    )
    if final_test_mse_at_best_params is not None:
        print(
            f"Test MSE of best-performing model (final state) found: {final_test_mse_at_best_params:.6f}"
        )


    return best_eta, best_lam, avg_mse_grid, None, final_test_mse_at_best_params


def run_model_and_get_history(config, best_eta, best_lam, initial_params):
    """Trains model with optimal params to capture full learning curve.
    Uses a representative split with the base SEED for plotting.
    Also runs early stopping trials to get average ES epoch and MSE."""
    activation_name = config["activation_name"]
    reg_type = config["reg_type"]

    print(
        f"\n--- Full Training Run ({activation_name}, {reg_type}, $\\eta$={best_eta:.2e}, $\\lambda$={best_lam:.2e}) ---"
    )

    # Get raw data and split ratios
    X_raw = config["X_raw"]
    y_raw = config["y_raw"]
    TRAIN_SIZE = config["TRAIN_SIZE"]
    VAL_SIZE = config["VAL_SIZE"]
    
    # Determine epochs and patience based on optimizer
    is_stochastic = config["optimizer_class"] is not GD
    if config["optimizer_class"] is GD:
        epochs_to_use = 6000
        patience_es = 300
    else:
        epochs_to_use = config["EPOCHS"]
        patience_es = config["PATIENCE"]
    
    # --- Run early stopping trials to get average ES epoch and MSE ---
    all_es_epochs = []
    all_es_val_mses = []
    
    print(f"Running {config['LR_TRIALS']} early stopping trials...")
    for trial in range(config["LR_TRIALS"]):
        trial_seed = config["SEED"] + trial
        
        # Create split for this trial
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, _, _, _, _, _, _, _ = (
            split_scale_data(X_raw, y_raw, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=trial_seed)
        )
        
        # Train with early stopping
        model_es = NeuralNetwork(
            network_input_size=X_train_s.shape[1],
            layer_output_sizes=config["layer_sizes"],
            activations=config["activations"],
            loss=MSE(),
            seed=trial_seed,
            lambda_reg=best_lam,
            reg_type=reg_type,
        )
        
        batch_size_setting = config["BATCH_SIZE"] if is_stochastic else len(X_train_s)
        
        history_es = train(
            nn=model_es,
            X_train=X_train_s,
            y_train=y_train_s,
            X_val=X_val_s,
            y_val=y_val_s,
            optimizer=config["optimizer_class"](eta=best_eta),
            epochs=epochs_to_use,
            batch_size=batch_size_setting,
            stochastic=is_stochastic,
            task="regression",
            early_stopping=True,
            patience=patience_es,
            verbose=False,
            seed=trial_seed,
        )
        
        es_val_loss_history = history_es.get('val_loss', [])
        # Only record ES metrics if early stopping actually triggered
        if len(es_val_loss_history) > 0 and len(es_val_loss_history) < epochs_to_use:
            es_epoch = len(es_val_loss_history) - 1
            es_val_mse = es_val_loss_history[-1]
            all_es_epochs.append(es_epoch)
            all_es_val_mses.append(es_val_mse)
    
    # Calculate average ES metrics
    if all_es_epochs:
        avg_es_epoch = np.mean(all_es_epochs)
        avg_es_val_mse = np.mean(all_es_val_mses)
        print(f"Average Early Stopping Epoch: {avg_es_epoch:.0f}")
        print(f"Average Early Stopping Val MSE (Scaled): {avg_es_val_mse:.6f}")
    else:
        avg_es_epoch = -1
        avg_es_val_mse = 0
        print("Early stopping did not trigger in any run (models need more epochs)")
    
    # --- Run full training (no early stopping) for learning curve with base SEED ---
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, _, _, _, _, _, _, _ = (
        split_scale_data(X_raw, y_raw, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=config["SEED"])
    )
    
    model = NeuralNetwork(
        network_input_size=X_train_scaled.shape[1],
        layer_output_sizes=config["layer_sizes"],
        activations=config["activations"],
        loss=MSE(),
        seed=config["SEED"],
        lambda_reg=best_lam,
        reg_type=reg_type,
    )

    batch_size_setting = config["BATCH_SIZE"] if is_stochastic else len(X_train_scaled)

    # Train for the full EPOCHS without early stopping to get the full curve
    history = train(
        nn=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        optimizer=config["optimizer_class"](eta=best_eta),
        epochs=epochs_to_use,
        batch_size=batch_size_setting,
        stochastic=is_stochastic,
        task="regression",
        early_stopping=False,
        verbose=False,
        seed=config["SEED"],
    )

    y_test_pred_scaled = model.predict(X_test_scaled)
    final_test_mse = mse(y_test_scaled, y_test_pred_scaled)

    print(f"Final Test MSE (Scaled): {final_test_mse:.6f}")

    return history, avg_es_epoch, avg_es_val_mse


def main_e(
    # Data Parameters
    N=1000,
    sigma=0.3,
    seed=45,
    # Experiment Parameters
    optimizer="RMSprop",
    activation="Sigmoid",
    experiment=2,
    regularization_types=["L1", "L2"],
    # Training Hyperparameters
    epochs=10000,
    epochs_lr_search=500,
    patience=200,
    batch_size=20,
    eta_vals=None,
    lambda_vals=None,
    n_trials=1,
    # Data split ratios
    train_size=0.6,
    val_size=0.2,
    # Plotting options
    plot_early_stopping=False,
    include_final_mse=True,
):
    """
    Main orchestrator function for Part E experiments.
    
    Parameters:
    -----------
    N : int
        Number of samples
    sigma : float
        Noise level
    seed : int
        Random seed for reproducibility
    optimizer : str
        Optimizer type: 'GD', 'RMSprop', or 'Adam'
    activation : str
        Activation function: 'Sigmoid', 'ReLU', or 'LeakyReLU'
    experiment : int
        Experiment ID (1, 2, or 3) defining architecture
    regularization_types : list
        List of regularization types to compare (e.g., ['L1', 'L2'])
    epochs : int
        Number of training epochs
    epochs_lr_search : int
        Epochs for hyperparameter search
    patience : int
        Patience for early stopping
    batch_size : int
        Batch size for training
    eta_vals : array-like or None
        Learning rate values to search. If None, defaults to logspace(-5, -1, 5)
    lambda_vals : array-like or None
        Lambda values to search. If None, defaults to logspace(-5, -1, 5)
    n_trials : int
        Number of trials with different data splits
    train_size : float
        Training set ratio
    val_size : float
        Validation set ratio
    plot_early_stopping : bool
        Whether to plot early stopping vertical line
    include_final_mse : bool
        Whether to include final MSE in legend labels
    
    Returns:
    --------
    None (generates and saves plots)
    """
    
    if eta_vals is None:
        eta_vals = np.logspace(-5, -1, 5)
    if lambda_vals is None:
        lambda_vals = np.logspace(-5, -1, 5)
    
    # Set seed BEFORE any operations that might affect random state
    # This ensures deterministic behavior regardless of prior notebook state
    np.random.seed(seed)
    
    optimizer_map = {"GD": GD, "RMSprop": RMSprop, "Adam": Adam}
    optimizer_class = optimizer_map[optimizer]
    
    # Match temp_e.py behavior: create all three activations, then select
    # This ensures identical object creation sequence
    activation_instance = {
        "Sigmoid": Sigmoid(),
        "ReLU": ReLU(),
        "LeakyReLU": LeakyReLU(),
    }[activation]
    
    architecture_configs = {
        1: ([50, 1], "1 Hidden Layer & 50 Nodes"),
        2: ([100, 100, 1], "2 Hidden Layers & 100 Nodes Each"),
        3: ([100, 100, 100, 1], "3 Hidden Layers & 100 Nodes Each"),
    }
    architecture_sizes, architecture_title = architecture_configs[experiment]
    
    full_activations = [activation_instance] * (len(architecture_sizes) - 1) + [Linear()]
    
    print(f"\n{'='*70}")
    print(f"NN EXPERIMENT RUNNER (L1 vs L2 Comparison)")
    print(f"N={N}, σ={sigma}, Opt={optimizer}, Act={activation}, Exp={experiment}")
    print(f"Architecture: {architecture_title}")
    print(f"Trials: {n_trials} (each with different data splits)")
    print(f"{'='*70}\n")
    
    # Don't reset seed again - generate_data() will set it internally
    # The seed was already set at the start of main_e()
    X, y = generate_data(N, noise_level=sigma, seed=seed)
    

    common_config = {
        "EPOCHS": epochs,
        "PATIENCE": patience,
        "SEED": seed,
        "ETA_VALS": eta_vals,
        "LAMBDA_VALS": lambda_vals,
        "LR_TRIALS": n_trials,
        "EPOCHS_LR_SEARCH": epochs_lr_search,
        "BATCH_SIZE": batch_size if optimizer_class is not GD else len(X),
        "X_raw": X,
        "y_raw": y,
        "TRAIN_SIZE": train_size,
        "VAL_SIZE": val_size,
        "optimizer_class": optimizer_class,
        "layer_sizes": architecture_sizes,
        "sigma": sigma,
        "N_SELECTED": N,
        "SIGMA_SELECTED": sigma,
        "EXPERIMENT_SELECTED": experiment,
    }
    
    # 3. Run experiments for each regularization type
    results_l1_l2 = {}
    
    for reg_type in regularization_types:
        print(f"\n{'*'*50}")
        print(f"** Running {reg_type} Regularization **")
        print(f"{'*'*50}")
        
        # A. Calculate ML Benchmark - Averaged over trials
        print(f"\n--- Computing {reg_type} Baseline (Averaged over {n_trials} trials) ---")
        baseline_mses = []
        for trial in range(n_trials):
            trial_seed = seed + trial
            if reg_type == "L2":
                baseline_mse = ridge_example(sigma=sigma, datasize=N, seed=trial_seed)
            else:  # L1
                baseline_mse = lasso_example(sigma=sigma, datasize=N, seed=trial_seed)
            baseline_mses.append(baseline_mse)
        
        ml_baseline_mse = np.mean(baseline_mses)
        ml_baseline_std = np.std(baseline_mses)
        print(f"Average {reg_type} Baseline Validation MSE (Scaled, {n_trials} trials): {ml_baseline_mse:.6f} +/- {ml_baseline_std:.6f}")
        print(f"Using {reg_type} ML Baseline Validation MSE ({ml_baseline_mse:.6f}) as Learning Curve Baseline.")
        

        config_search = common_config.copy()
        config_search.update({
            "activations": full_activations,
            "activation_name": activation,
            "reg_type": reg_type,
        })
        

        best_eta, best_lam, mse_grid, best_model_params, final_test_mse_search = (
            find_best_lambda_eta(config_search)
        )
        

        lambda_eta_heatmap(
            mse_grid,
            eta_vals,
            lambda_vals,
            metric_name="Validation MSE",
            optimizer_name=optimizer,
            reg_type=reg_type,
            N=N,
            sigma=sigma,
            exp_id=experiment,
            activation_name=activation,
            n_trials=n_trials,
        )


        history, avg_es_epoch, avg_es_val_mse = run_model_and_get_history(
            config_search, best_eta, best_lam, best_model_params
        )
        
        results_l1_l2[reg_type] = {
            "history": history,
            "best_eta": best_eta,
            "best_lam": best_lam,
            "ml_baseline_mse": ml_baseline_mse,
            "final_test_mse_search": final_test_mse_search,
            "avg_es_epoch": avg_es_epoch,
            "avg_es_val_mse": avg_es_val_mse,
        }
    

    results_l1 = results_l1_l2["L1"]
    results_l2 = results_l1_l2["L2"]
    
    plot_combined_learning_curves(
        results_l1,
        results_l2,
        results_l1["ml_baseline_mse"],
        results_l2["ml_baseline_mse"],
        activation,
        architecture_title,
        common_config,
        plot_early_stopping=plot_early_stopping,
        include_final_mse=include_final_mse,
    )
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")




# ============================================================================
#                      EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    main_e(
        N=1000,
        sigma=0.3,
        optimizer="RMSprop",
        activation="Sigmoid",
        experiment=2,
        regularization_types=["L1", "L2"],
        epochs=10000,
        epochs_lr_search=500,
        patience=200,
        batch_size=20,
        n_trials=1,
        seed=45,
        plot_early_stopping=False,
        include_final_mse=True,
    )














