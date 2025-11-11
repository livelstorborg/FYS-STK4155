from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim


from src.neural_network import NeuralNetwork
from src.activations import LeakyReLU, ReLU, Sigmoid, Linear
from src.losses import MSE
from src.optimizers import GD, RMSprop, Adam
from src.training import train
from src.metrics import mse
from src.utils import runge, polynomial_features, scale_data, split_scale_data, OLS_parameters, generate_data
from src.plotting import plot_learning_curves_with_std_on_ax


def ols_example(sigma, datasize, seed, train_size=0.6, val_size=0.2):
    """Calculates OLS scaled MSE metrics on a dataset with fixed P=14."""
    np.random.seed(seed)
    x = np.linspace(-1, 1, datasize)
    y_true = runge(x)
    y_noise = y_true + np.random.normal(0, sigma, datasize)

    X_poly = polynomial_features(x, p=14, intercept=False)

    test_size = 1 - train_size - val_size

    X_temp, X_test_poly, y_temp, y_test = train_test_split(X_poly, y_noise, test_size=test_size, random_state=seed)
    val_ratio_adj = val_size / (train_size + val_size)
    X_train_poly, X_val_poly, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio_adj, random_state=seed)

    X_train_s, y_train_s, X_mean, X_std, y_mean, y_std = scale_data(X_train_poly, y_train.reshape(-1, 1))
    X_val_s, y_val_s, _, _, _, _ = scale_data(X_val_poly, y_val.reshape(-1, 1), X_mean, X_std, y_mean, y_std)

    theta_ols = OLS_parameters(X_train_s, y_train_s)
    y_pred_ols_val = X_val_s @ theta_ols
    
    ols_val_mse = mse(y_val_s.reshape(-1, 1), y_pred_ols_val.reshape(-1, 1))

    print(f"OLS Benchmark (N={datasize}, P=14, Sigma={sigma}, Seed={seed}): Val MSE (Scaled) = {ols_val_mse:.6f}")
    
    return ols_val_mse.flatten()[0] if ols_val_mse.ndim > 1 else ols_val_mse


# --------------------------------------------------------------------
# PyTorch Baseline Implementation
# --------------------------------------------------------------------

class RungeNN_PT_Dynamic(nn.Module):
    def __init__(self, layer_sizes, activation_fn=nn.Sigmoid):
        """Dynamically creates a feedforward network."""
        super(RungeNN_PT_Dynamic, self).__init__()
        
        input_size = 1
        layers = []
        
        for out_size in layer_sizes[:-1]:
            layers.append(nn.Linear(input_size, out_size))
            layers.append(activation_fn()) 
            input_size = out_size
            
        # Output layer (Linear activation)
        layers.append(nn.Linear(input_size, layer_sizes[-1]))
        
        self.network = nn.Sequential(*layers)
        
        if activation_fn == nn.ReLU or activation_fn == nn.LeakyReLU:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'linear'
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                if m is layers[-1]:  # Output layer
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Hidden layers
                    if nonlinearity == 'relu':
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def get_pytorch_mse_for_config(config, best_eta, X_train_s_np, y_train_s_np, X_val_s_np, y_val_s_np):
    """
    Trains a PyTorch model matching the given config and returns final Train and Validation 
    scaled MSE HISTORIES using a single, provided split.
    """
    
    torch.manual_seed(config['SEED'])
    
    X_train = torch.tensor(X_train_s_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_s_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_s_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_s_np, dtype=torch.float32)

    optimizer_class = config['optimizer_class']
    
    batch_size = len(X_train) if optimizer_class is GD else config['BATCH_SIZE']
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    activation_map = {
        'Sigmoid': nn.Sigmoid,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
    }
    
    custom_activation = config['activations'][0]
    custom_activation_name = custom_activation.__class__.__name__
    pt_activation = activation_map.get(custom_activation_name, nn.Sigmoid)
     
    model = RungeNN_PT_Dynamic(config['layer_sizes'], activation_fn=pt_activation)
    criterion = nn.MSELoss()
    
    pt_optimizer_name = optimizer_class.__name__
    if pt_optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_eta)
        epochs_to_use = config['EPOCHS']
    elif pt_optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=best_eta)
        epochs_to_use = config['EPOCHS']
    elif pt_optimizer_name == 'GD':
        optimizer = optim.SGD(model.parameters(), lr=best_eta)
        epochs_to_use = 2000  
    else:
        optimizer = optim.Adam(model.parameters(), lr=best_eta)
        epochs_to_use = config['EPOCHS']
            
    pt_train_loss_history = []
    pt_val_loss_history = []

    for _ in range(epochs_to_use):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train)
            train_mse_scaled = criterion(y_train_pred, y_train).item()
            pt_train_loss_history.append(train_mse_scaled)
            
            y_val_pred = model(X_val)
            val_mse_scaled = criterion(y_val_pred, y_val).item()
            pt_val_loss_history.append(val_mse_scaled)

    return pt_train_loss_history, pt_val_loss_history


def find_best_eta(config):
    """Find best learning rate using validation set, averaging over LR_TRIALS with DIFFERENT DATA SPLITS."""
    layer_sizes = config['layer_sizes']
    activations = config['activations']
    optimizer_class = config['optimizer_class']
    
    # Get raw data and split ratios
    X_raw = config['X_raw']
    y_raw = config['y_raw']
    TRAIN_SIZE = config['TRAIN_SIZE']
    VAL_SIZE = config['VAL_SIZE']
    
    val_mse_all_trials = []
    
    optimizer_name = optimizer_class.__name__
    act_name_print = config.get('act_name', optimizer_name)
    print(f"\n--- Starting {act_name_print} LR Search ({config['LR_TRIALS']} trials per eta, new split each trial) ---")
    print("Eta | Avg Val MSE")
    print("-------------------------")

    for trial in range(config['LR_TRIALS']):
        trial_seed = config['SEED'] + trial
        val_mse_list = []
        
        # Perform split and scale for this trial
        X_train_s, X_val_s, _, y_train_s, y_val_s, _, _, _, _, _, _, _, _ = split_scale_data(
            X_raw, y_raw, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=trial_seed
        )
        NETWORK_INPUT_SIZE = X_train_s.shape[1]
        
        for eta in config['ETA_VALS']:
            model = NeuralNetwork(
                network_input_size=NETWORK_INPUT_SIZE,
                layer_output_sizes=layer_sizes,
                activations=activations,
                loss=MSE(),
                seed=trial_seed,
                lambda_reg=0.0
            )
            
            # Use appropriate epochs and batch size for LR search
            train(
                nn=model, X_train=X_train_s, y_train=y_train_s, 
                X_val=X_val_s, y_val=y_val_s,
                optimizer=optimizer_class(eta=eta),
                epochs=config['EPOCHS_LR_SEARCH'],
                batch_size=(X_train_s.shape[0] if optimizer_class is GD else config['BATCH_SIZE']),
                stochastic=(False if optimizer_class is GD else True),
                task='regression', early_stopping=True, patience=150, verbose=False, seed=trial_seed
            )
            
            y_val_pred_scaled = model.predict(X_val_s)
            val_mse_scaled = mse(y_val_s, y_val_pred_scaled)
            val_mse_list.append(val_mse_scaled)
        
        val_mse_all_trials.append(val_mse_list)
    
    avg_val_mse = np.mean(val_mse_all_trials, axis=0)
    
    for eta, avg_mse in zip(config['ETA_VALS'], avg_val_mse):
        print(f"{eta:.2e} | {avg_mse:.6f}")
        
    best_idx = np.argmin(avg_val_mse)
    best_eta = config['ETA_VALS'][best_idx]
    
    results = {
        'eta_vals': config['ETA_VALS'], 'avg_val_mse': avg_val_mse,
        'best_eta': best_eta, 'best_val_mse': avg_val_mse[best_idx],
    }
    return best_eta, results


def run_multiple_runs_and_get_stats(config, best_eta):
    """
    Runs N_TRIALS models with the best eta found, and returns statistics for the 
    full-run loss curves and the average Early Stopping (ES) point, with DIFFERENT DATA SPLITS.
    """
    optimizer_name = config['optimizer_class'].__name__
    act_name_print = config.get('act_name', optimizer_name)
    num_runs = config['LR_TRIALS']
    print(f"\n--- Multi-Run Training ({act_name_print}, eta={best_eta:.2e}, {num_runs} runs with different splits) ---")

    X_raw = config['X_raw']
    y_raw = config['y_raw']
    TRAIN_SIZE = config['TRAIN_SIZE']
    VAL_SIZE = config['VAL_SIZE']

    if config['optimizer_class'] is GD:
        epochs_to_use = config['EPOCHS_GD']
        patience_es = 500
    else:
        epochs_to_use = config['EPOCHS']
        patience_es = 150
    
    all_train_loss = []
    all_val_loss = []
    final_val_mses = []
    
    # Early Stopping metrics lists
    all_es_epochs = []
    all_es_val_mses = []
    
    print(f"Running {num_runs} full runs (epochs={epochs_to_use}, no early stopping) AND early stopping runs...")

    for run in range(num_runs):
        trial_seed = config['SEED'] + run
        np.random.seed(trial_seed)
        
        # Perform split and scale for this run
        X_train_s, X_val_s, _, y_train_s, y_val_s, _, _, _, _, _, _, _, _ = split_scale_data(
            X_raw, y_raw, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=trial_seed
        )
        NETWORK_INPUT_SIZE = X_train_s.shape[1]
        
        # 1. Run with early stopping to get the ES point
        model_es = NeuralNetwork(
            network_input_size=NETWORK_INPUT_SIZE,
            layer_output_sizes=config['layer_sizes'],
            activations=config['activations'],
            loss=MSE(),
            seed=trial_seed,
            lambda_reg=0.0
        )

        history_es = train(
            nn=model_es,
            X_train=X_train_s, y_train=y_train_s,
            X_val=X_val_s, y_val=y_val_s,
            optimizer=config['optimizer_class'](eta=best_eta),
            epochs=epochs_to_use,
            batch_size=(X_train_s.shape[0] if config['optimizer_class'] is GD else config['BATCH_SIZE']),
            stochastic=(False if config['optimizer_class'] is GD else True),
            task='regression', 
            early_stopping=True, 
            patience=patience_es,
            verbose=False, 
            seed=trial_seed
        )
        
        es_val_loss_history = history_es.get('val_loss', [])
        if len(es_val_loss_history) > 0 and len(es_val_loss_history) < epochs_to_use:
            es_epoch = len(es_val_loss_history) - 1
            es_val_mse = es_val_loss_history[-1]
            all_es_epochs.append(es_epoch)
            all_es_val_mses.append(es_val_mse)


        model_full = NeuralNetwork(
            network_input_size=NETWORK_INPUT_SIZE,
            layer_output_sizes=config['layer_sizes'],
            activations=config['activations'],
            loss=MSE(),
            seed=trial_seed,
            lambda_reg=0.0
        )

        history_full = train(
            nn=model_full,
            X_train=X_train_s, y_train=y_train_s,
            X_val=X_val_s, y_val=y_val_s,
            optimizer=config['optimizer_class'](eta=best_eta),
            epochs=epochs_to_use,
            batch_size=(X_train_s.shape[0] if config['optimizer_class'] is GD else config['BATCH_SIZE']),
            stochastic=(False if config['optimizer_class'] is GD else True),
            task='regression', early_stopping=False, verbose=False, seed=trial_seed
        )
        

        current_length = len(history_full.get('train_loss', []))
        
        if current_length > 0:
            if current_length < epochs_to_use:
                last_train_loss = history_full['train_loss'][-1]
                pad_size = epochs_to_use - current_length
                padded_train_loss = np.pad(history_full['train_loss'], (0, pad_size), 'constant', constant_values=last_train_loss)
                all_train_loss.append(padded_train_loss)
                
                last_val_loss = history_full['val_loss'][-1]
                padded_val_loss = np.pad(history_full['val_loss'], (0, pad_size), 'constant', constant_values=last_val_loss)
                all_val_loss.append(padded_val_loss)
            else:
                all_train_loss.append(history_full['train_loss'])
                all_val_loss.append(history_full['val_loss'])
            
            y_val_pred_scaled = model_full.predict(X_val_s)
            final_val_mses.append(mse(y_val_s, y_val_pred_scaled))
        else:
            print(f"Warning: Full Run {run} failed to produce any history. Skipping run for full averaging.")

    if not all_train_loss:
        print("ERROR: No successful full runs completed.")
        return None
    
    # Full run metrics (Mean Curves)
    train_loss_array = np.array(all_train_loss)
    val_loss_array = np.array(all_val_loss)

    train_loss_mean = np.mean(train_loss_array, axis=0)
    train_loss_std = np.std(train_loss_array, axis=0)
    val_loss_mean = np.mean(val_loss_array, axis=0)
    val_loss_std = np.std(val_loss_array, axis=0)
    
    avg_final_val_mse = np.mean(final_val_mses)
    std_final_val_mse = np.std(final_val_mses)

    print(f"Average Final Validation MSE (Scaled, {num_runs} full runs): {avg_final_val_mse:.6f} +/- {std_final_val_mse:.6f}")
    
    # Early Stopping run metrics
    if all_es_epochs:
        avg_es_epoch = np.mean(all_es_epochs)
        std_es_epoch = np.std(all_es_epochs)
        avg_es_val_mse = np.mean(all_es_val_mses)
        std_es_val_mse = np.std(all_es_val_mses)
        print(f"Average Early Stopping Epoch: {avg_es_epoch:.0f} +/- {std_es_epoch:.0f}")
        print(f"Average Early Stopping Val MSE (Scaled): {avg_es_val_mse:.6f} +/- {std_es_val_mse:.6f}")
    else:
        avg_es_epoch, std_es_epoch, avg_es_val_mse, std_es_val_mse = -1, 0, 0, 0
        print("Early stopping did not trigger in any run (models need more epochs)")
    
    return {
        'train_loss_mean': train_loss_mean, 'train_loss_std': train_loss_std,
        'val_loss_mean': val_loss_mean, 'val_loss_std': val_loss_std,
        'avg_es_epoch': avg_es_epoch, 'std_es_epoch': std_es_epoch,
        'avg_es_val_mse': avg_es_val_mse, 'std_es_val_mse': std_es_val_mse,
    }


def execute_adam_rmsprop_experiments(N, SIGMA, OLS_BASELINE_MSE, X, y, COMMON_CONFIG_BASE, 
                                     COMPARE_WITH_PYTORCH, layer_sizes_1, activations_1,
                                     layer_sizes_2, activations_2, TRAIN_SIZE, VAL_SIZE, SEED):
    """Runs and plots the Adam and RMSprop experiments for a single N/SIGMA pair."""
    print(f"\n--- Running Adam & RMSprop Experiments for N={N}, SIGMA={SIGMA} ---")

    # Define common configs needed
    COMMON_CONFIG_ADAM = {**COMMON_CONFIG_BASE, 'optimizer_class': Adam}
    COMMON_CONFIG_RMS = {**COMMON_CONFIG_BASE, 'optimizer_class': RMSprop}
    
    # Define experiments to run
    EXPERIMENTS_ADAM_RMS = [
        (COMMON_CONFIG_ADAM, layer_sizes_1, activations_1, 'Adam, 1 Layer (50)'),
        (COMMON_CONFIG_ADAM, layer_sizes_2, activations_2, 'Adam, 2 Layers (100)'),
        (COMMON_CONFIG_RMS, layer_sizes_1, activations_1, 'RMSprop, 1 Layer (50)'),
        (COMMON_CONFIG_RMS, layer_sizes_2, activations_2, 'RMSprop, 2 Layers (100)'),
    ]

    # Split data for PyTorch baseline (uses a single, representative split)
    X_train_s_pt_rep = X_val_s_pt_rep = y_train_s_pt_rep = y_val_s_pt_rep = None
    if COMPARE_WITH_PYTORCH:
        X_train_s_pt_rep, X_val_s_pt_rep, _, y_train_s_pt_rep, y_val_s_pt_rep, _, _, _, _, _, _, _, _ = split_scale_data(
            X, y, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=SEED 
        )

    history_storage = {}
    
    for i, (base_config, sizes, acts, name) in enumerate(EXPERIMENTS_ADAM_RMS):
        config_search = base_config.copy()
        config_search.update({'layer_sizes': sizes, 'activations': acts, 'act_name': name})
        
        best_eta, _ = find_best_eta(config_search)
        stats = run_multiple_runs_and_get_stats(config_search, best_eta) 
        
        pt_train_history = None
        pt_val_history = None
        
        if COMPARE_WITH_PYTORCH:
            print(f"\n--- Calculating PyTorch Baseline History for {name} (Eta={best_eta:.2e}) ---")
            pt_train_history, pt_val_history = get_pytorch_mse_for_config(
                config_search, best_eta, X_train_s_pt_rep, y_train_s_pt_rep, X_val_s_pt_rep, y_val_s_pt_rep
            )
            print(f"PyTorch Final Train MSE: {pt_train_history[-1]:.6f}, PyTorch Final Val MSE: {pt_val_history[-1]:.6f}")
        
        history_storage[name] = {
            'stats': stats, 
            'best_eta': best_eta,
            'pt_train_history': pt_train_history,
            'pt_val_history': pt_val_history
        }


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    n_trials = COMMON_CONFIG_BASE['LR_TRIALS']
    fig.suptitle(f'Learning Curves for N={N}, $\\mathbf{{\sigma}}$={SIGMA} (Averaged Over {n_trials} Runs)', 
                 fontsize=20, fontweight='bold', y=1.0)
    
    fig.text(0.5, 0.92, '1 Hidden Layer & 50 Nodes', 
             fontsize=18, fontweight='bold', ha='center', va='bottom')
    fig.text(0.5, 0.46, '2 Hidden Layers & 100 Nodes Each', 
             fontsize=18, fontweight='bold', ha='center', va='bottom')

    h_adam_1 = history_storage['Adam, 1 Layer (50)']
    h_rms_1 = history_storage['RMSprop, 1 Layer (50)']
    
    plot_learning_curves_with_std_on_ax(
        axes[0, 0], 
        h_adam_1['stats']['train_loss_mean'], h_adam_1['stats']['train_loss_std'], 
        h_adam_1['stats']['val_loss_mean'], h_adam_1['stats']['val_loss_std'], 
        OLS_BASELINE_MSE, 
        h_adam_1['stats']['avg_es_epoch'], h_adam_1['stats']['avg_es_val_mse'],
        h_adam_1['pt_train_history'], h_adam_1['pt_val_history'],
        f'Adam: $\\eta$={h_adam_1["best_eta"]:.4f}', ylabel_enabled=True,
        current_N=N, current_SIGMA=SIGMA)
        
    plot_learning_curves_with_std_on_ax(
        axes[0, 1], 
        h_rms_1['stats']['train_loss_mean'], h_rms_1['stats']['train_loss_std'], 
        h_rms_1['stats']['val_loss_mean'], h_rms_1['stats']['val_loss_std'], 
        OLS_BASELINE_MSE, 
        h_rms_1['stats']['avg_es_epoch'], h_rms_1['stats']['avg_es_val_mse'],
        h_rms_1['pt_train_history'], h_rms_1['pt_val_history'],
        f'RMSprop: $\\eta$={h_rms_1["best_eta"]:.4f}', ylabel_enabled=False,
        current_N=N, current_SIGMA=SIGMA)
    
    h_adam_2 = history_storage['Adam, 2 Layers (100)']
    h_rms_2 = history_storage['RMSprop, 2 Layers (100)']

    plot_learning_curves_with_std_on_ax(
        axes[1, 0], 
        h_adam_2['stats']['train_loss_mean'], h_adam_2['stats']['train_loss_std'], 
        h_adam_2['stats']['val_loss_mean'], h_adam_2['stats']['val_loss_std'], 
        OLS_BASELINE_MSE, 
        h_adam_2['stats']['avg_es_epoch'], h_adam_2['stats']['avg_es_val_mse'],
        h_adam_2['pt_train_history'], h_adam_2['pt_val_history'],
        f'Adam: $\\eta$={h_adam_2["best_eta"]:.4f}', ylabel_enabled=True,
        current_N=N, current_SIGMA=SIGMA)
        
    plot_learning_curves_with_std_on_ax(
        axes[1, 1], 
        h_rms_2['stats']['train_loss_mean'], h_rms_2['stats']['train_loss_std'], 
        h_rms_2['stats']['val_loss_mean'], h_rms_2['stats']['val_loss_std'], 
        OLS_BASELINE_MSE, 
        h_rms_2['stats']['avg_es_epoch'], h_rms_2['stats']['avg_es_val_mse'],
        h_rms_2['pt_train_history'], h_rms_2['pt_val_history'],
        f'RMSprop: $\\eta$={h_rms_2["best_eta"]:.4f}', ylabel_enabled=False,
        current_N=N, current_SIGMA=SIGMA)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.90, hspace=0.35)
    
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(f'figs/lc_N{N}_sigma{SIGMA}.pdf', bbox_inches='tight', dpi=300)
    
    print(f"\nSaved combined averaged plot: figs/lc_N{N}_sigma{SIGMA}.pdf")
    return history_storage


def execute_gd_activation_comparison(N, SIGMA, OLS_BASELINE_MSE, X, y, COMMON_CONFIG_BASE, 
                                     COMPARE_WITH_PYTORCH, activation_classes_to_test,
                                     gd_act_architectures_configs, TRAIN_SIZE, VAL_SIZE, SEED, n_trials):
    """Runs and plots the GD activation comparison for a single N/SIGMA pair."""
    print(f"\n--- Running GD Activation Comparison Experiments for N={N}, SIGMA={SIGMA} ---")

    # Define common config needed
    COMMON_CONFIG_GD = {**COMMON_CONFIG_BASE, 'optimizer_class': GD, 'BATCH_SIZE': len(X)}
    
    # Split data for PyTorch baseline
    X_train_s_pt_rep = X_val_s_pt_rep = y_train_s_pt_rep = y_val_s_pt_rep = None
    if COMPARE_WITH_PYTORCH:
        X_train_s_pt_rep, X_val_s_pt_rep, _, y_train_s_pt_rep, y_val_s_pt_rep, _, _, _, _, _, _, _, _ = split_scale_data(
            X, y, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=SEED 
        )

    for arch_config in gd_act_architectures_configs:
        arch_name = arch_config['name']
        layer_sizes_to_use = arch_config['layer_sizes']
        num_hidden_layers = len(layer_sizes_to_use) - 1 
        
        activation_results = {}
        
        print(f"\n--- Running Averaged Activation Comparison for: {arch_name} (Using GD) ---")

        for act_name, activation_class in activation_classes_to_test.items():
            activations_list = [activation_class() for _ in range(num_hidden_layers)]
            activations_list.append(Linear())
            
            config_act_search = COMMON_CONFIG_GD.copy()
            config_act_search.update({
                'layer_sizes': layer_sizes_to_use,
                'activations': activations_list,
                'act_name': f'GD {act_name} ({arch_name})',
            })
            
            best_eta_act, _ = find_best_eta(config_act_search)
            stats_act = run_multiple_runs_and_get_stats(config_act_search, best_eta_act)
            
            pt_train_history = None
            pt_val_history = None
            
            if COMPARE_WITH_PYTORCH:
                print(f"\n--- Calculating PyTorch Baseline History for GD {act_name} (Eta={best_eta_act:.2e}) ---")
                pt_train_history, pt_val_history = get_pytorch_mse_for_config(
                    config_act_search, best_eta_act, X_train_s_pt_rep, y_train_s_pt_rep, X_val_s_pt_rep, y_val_s_pt_rep
                )
                print(f"PyTorch Final Train MSE: {pt_train_history[-1]:.6f}, PyTorch Final Val MSE: {pt_val_history[-1]:.6f}")
                
            activation_results[act_name] = {
                'best_eta': best_eta_act,
                'stats': stats_act,
                'pt_train_history': pt_train_history,
                'pt_val_history': pt_val_history
            }
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        fig.suptitle(f'Learning Curves with GD (N={N}, $\\mathbf{{\sigma}}$={SIGMA}) - Averaged Over {n_trials} Runs\n {arch_name}', 
                     fontsize=22, fontweight='bold', y=1.07)
        
        # Determine experiment number
        if '1 Hidden Layer' in arch_name:
            exp_num = 1
            show_es = False
        else:
            exp_num = 2
            show_es = True
        
        h_sig = activation_results['Sigmoid']
        plot_learning_curves_with_std_on_ax(
            axes[0], 
            h_sig['stats']['train_loss_mean'], h_sig['stats']['train_loss_std'], 
            h_sig['stats']['val_loss_mean'], h_sig['stats']['val_loss_std'],
            OLS_BASELINE_MSE,
            h_sig['stats']['avg_es_epoch'], h_sig['stats']['avg_es_val_mse'],
            pt_train_history=h_sig['pt_train_history'], pt_val_history=h_sig['pt_val_history'],
            title=f'Sigmoid: $\\eta$={h_sig["best_eta"]:.4f}',
            ylabel_enabled=True,
            current_N=N, current_SIGMA=SIGMA,
            show_es_line=show_es
        )
        
        h_relu = activation_results['ReLU']
        plot_learning_curves_with_std_on_ax(
            axes[1], 
            h_relu['stats']['train_loss_mean'], h_relu['stats']['train_loss_std'], 
            h_relu['stats']['val_loss_mean'], h_relu['stats']['val_loss_std'],
            OLS_BASELINE_MSE,
            h_relu['stats']['avg_es_epoch'], h_relu['stats']['avg_es_val_mse'],
            pt_train_history=h_relu['pt_train_history'], pt_val_history=h_relu['pt_val_history'],
            title=f'ReLU: $\\eta$={h_relu["best_eta"]:.4f}',
            ylabel_enabled=False,
            current_N=N, current_SIGMA=SIGMA,
            show_es_line=show_es
        )
        
        h_lrelu = activation_results['LeakyReLU']
        plot_learning_curves_with_std_on_ax(
            axes[2], 
            h_lrelu['stats']['train_loss_mean'], h_lrelu['stats']['train_loss_std'], 
            h_lrelu['stats']['val_loss_mean'], h_lrelu['stats']['val_loss_std'],
            OLS_BASELINE_MSE,
            h_lrelu['stats']['avg_es_epoch'], h_lrelu['stats']['avg_es_val_mse'],
            pt_train_history=h_lrelu['pt_train_history'], pt_val_history=h_lrelu['pt_val_history'],
            title=f'LeakyReLU: $\\eta$={h_lrelu["best_eta"]:.4f}',
            ylabel_enabled=False,
            current_N=N, current_SIGMA=SIGMA,
            show_es_line=show_es
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(top=0.90, hspace=0.35)
        
        Path("figs").mkdir(exist_ok=True)
        plt.savefig(f'figs/lc_gd_N{N}_sigma{SIGMA}_comparison_exp{exp_num}.pdf', bbox_inches='tight', dpi=300)
        
        print(f"\nSaved activation comparison plot: figs/lc_gd_N{N}_sigma{SIGMA}_comparison_exp{exp_num}.pdf")


def main_b(
    # Data parameters
    N=300,
    sigma=0.1,
    seed=42,
    # Experiment control flags
    run_adam_rmsprop=True,
    run_gd_activation_comparison=True,
    compare_with_pytorch=False,
    # Training hyperparameters
    epochs=3000,
    epochs_gd=10000,
    batch_size=10,
    eta_vals=None,
    n_trials=3,
    # Data split ratios
    train_size=0.6,
    val_size=0.2,
):
    """
    Main orchestrator function for Part B experiments (Neural Network Regression)
    
    Parameters:
    -----------
    N : int
        Number of samples
    sigma : float
        Noise level
    seed : int
        Random seed for reproducibility
    run_adam_rmsprop : bool
        Run Adam and RMSprop comparison (generates 2x2 plot)
    run_gd_activation_comparison : bool
        Run GD activation comparison (generates 1x3 plots)
    compare_with_pytorch : bool
        Include PyTorch baseline in plots
    epochs : int
        Max epochs for Adam/RMSprop
    epochs_gd : int
        Max epochs for GD
    batch_size : int
        Batch size for stochastic optimizers
    eta_vals : array-like or None
        Learning rates to search. If None, defaults to logspace(-4, -1, 10)
    n_trials : int
        Number of trials with different data splits
    train_size : float
        Training set ratio
    val_size : float
        Validation set ratio
    
    Returns:
    --------
    None (generates and saves plots)
    """
    
    # Set defaults
    if eta_vals is None:
        eta_vals = np.logspace(-4, -1, 10)
    
    # Set random seed
    np.random.seed(seed)
    
    # Define architectures
    layer_sizes_1 = [50, 1]
    activations_1 = [Sigmoid(), Linear()]
    
    layer_sizes_2 = [100, 100, 1]
    activations_2 = [Sigmoid(), Sigmoid(), Linear()]
    
    # Define activation classes to test (for GD comparison)
    activation_classes_to_test = {
        'Sigmoid': Sigmoid,
        'ReLU': ReLU,
        'LeakyReLU': LeakyReLU
    }
    
    # Define architectures for GD activation comparison
    gd_act_architectures_configs = [
        {
            'name': '1 Hidden Layer & 50 Nodes',
            'layer_sizes': layer_sizes_1,
        },
        {
            'name': '2 Hidden Layers & 100 Nodes Each',
            'layer_sizes': layer_sizes_2,
        }
    ]
    
    print(f"\nRunning experiments for N={N}, SIGMA={sigma} (Averaging over {n_trials} runs with DIFFERENT SPLITS)\n")
    
    # Calculate OLS baseline averaged over N_TRIALS with different seeds
    print(f"\n--- Computing OLS Baseline (Averaged over {n_trials} trials) ---")
    ols_baseline_mses = []
    for trial in range(n_trials):
        trial_seed = seed + trial
        ols_mse = ols_example(sigma=sigma, datasize=N, seed=trial_seed, 
                             train_size=train_size, val_size=val_size)
        ols_baseline_mses.append(ols_mse)
    
    OLS_BASELINE_MSE = np.mean(ols_baseline_mses)
    OLS_BASELINE_STD = np.std(ols_baseline_mses)
    print(f"Average OLS Validation MSE (Scaled, {n_trials} trials): {OLS_BASELINE_MSE:.6f} +/- {OLS_BASELINE_STD:.6f}")
    
    X, y = generate_data(N, noise_level=sigma, seed=seed)
    
    # Base config common to all subsequent experiments
    COMMON_CONFIG_BASE = {
        'SEED': seed, 'ETA_VALS': eta_vals, 'LR_TRIALS': n_trials,
        'EPOCHS': epochs, 'EPOCHS_LR_SEARCH': 300, 'BATCH_SIZE': batch_size,
        'EPOCHS_GD': epochs_gd,
        'X_raw': X, 'y_raw': y,
        'TRAIN_SIZE': train_size, 'VAL_SIZE': val_size,
    }
    
    if run_adam_rmsprop:
        execute_adam_rmsprop_experiments(
            N, sigma, OLS_BASELINE_MSE, X, y, COMMON_CONFIG_BASE, compare_with_pytorch,
            layer_sizes_1, activations_1, layer_sizes_2, activations_2, 
            train_size, val_size, seed
        )

    if run_gd_activation_comparison:
        execute_gd_activation_comparison(
            N, sigma, OLS_BASELINE_MSE, X, y, COMMON_CONFIG_BASE, compare_with_pytorch,
            activation_classes_to_test, gd_act_architectures_configs,
            train_size, val_size, seed, n_trials
        )

    print(f"\nAll requested experiments completed.\n")


# ============================================================================
#               EXAMPLE USAGE (for testing purposes only)
# ============================================================================

if __name__ == "__main__":
    main_b(
        N=300,
        sigma=0.1,
        seed=42,
        run_adam_rmsprop=True,
        run_gd_activation_comparison=True,
        compare_with_pytorch=False,
        epochs=3000,
        epochs_gd=10000,
        batch_size=10,
        n_trials=3
    )