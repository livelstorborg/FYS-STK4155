"""
Neural Network Model Complexity Analysis - Multi-Trial Parallel Version
========================================================================
Parallelized training for faster execution.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# Project imports
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, LeakyReLU, Linear
from src.losses import MSE
from src.optimizers import Adam, RMSprop
from src.training import train
from src.metrics import mse
from src.utils import runge, scale_data, inverse_scale_y, save_all_results, load_all_results


# =============================================================================
#                              CONFIGURATION
# =============================================================================

SEED = 42
N_POINTS = 1000
TEST_SIZE = 0.2
VAL_SIZE = 0.25
NOISE_LEVEL = 0.3

EPOCHS = 2000
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.01
OPTIMIZER = 'adam'

NETWORK_INPUT_SIZE = 1
N_TRIALS = 3

# Parallelization
N_WORKERS = cpu_count() - 1  # Leave 1 core free

RESULTS_DIR = 'results'
FIGS_DIR = 'figs'


# =============================================================================
#                          ARCHITECTURE DEFINITIONS
# =============================================================================

# NEURONS_CONFIG = {
#     1: [50, 100, 150, 200, 300, 400, 500], 
#     2: [50, 100, 150, 200, 300, 400, 500],
#     3: [50, 100, 150, 200, 300, 400, 500],
#     4: [50, 100, 150, 200, 300, 400, 500],
#     5: [50, 100, 150, 200, 300, 400, 500],
# }

NEURONS_CONFIG = {
    1: [10, 20, 30, 50, 60, 80, 100],  # 
    2: [10, 20, 30, 50, 60, 80, 100],
    3: [10, 20, 30, 50, 60, 80, 100],
    4: [10, 20, 30, 50, 60, 80, 100],
}

ACTIVATIONS_TO_TEST = [
    (Sigmoid(), "Sigmoid")
    # (ReLU(), "ReLU"),
    # (LeakyReLU(), "Leaky ReLU"),
]


# =============================================================================
#                              HELPER FUNCTIONS
# =============================================================================

def count_parameters(network_input_size, layer_output_sizes):
    total_params = 0
    prev_size = network_input_size
    for layer_size in layer_output_sizes:
        total_params += (prev_size * layer_size) + layer_size
        prev_size = layer_size
    return total_params


def get_adaptive_learning_rate(n_params, base_lr=0.001):
    return base_lr 


def prepare_data_for_trial(trial_seed):
    np.random.seed(trial_seed)
    x = np.linspace(-1, 1, N_POINTS)
    y_true = runge(x)
    y_noise = y_true + np.random.normal(0, NOISE_LEVEL, N_POINTS)
    
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        x.reshape(-1, 1), y_noise.reshape(-1, 1),
        test_size=TEST_SIZE, random_state=trial_seed
    )
    
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=trial_seed
    )
    
    X_train, y_train_scaled, X_mean, X_std, y_mean = scale_data(X_train_raw, y_train)
    X_val, y_val_scaled, _, _, _ = scale_data(X_val_raw, y_val, X_mean, X_std, y_mean)
    X_test, y_test_scaled, _, _, _ = scale_data(X_test_raw, y_test, X_mean, X_std, y_mean)
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_scaled': y_train_scaled, 'y_val_scaled': y_val_scaled, 'y_test_scaled': y_test_scaled,
        'y_train_original': y_train, 'y_val_original': y_val, 'y_test_original': y_test,
        'y_mean': y_mean
    }


# =============================================================================
#                    PARALLELIZABLE TRAINING FUNCTION
# =============================================================================

def train_single_config(args):
    """
    Train a single architecture for a single trial.
    This function will be called in parallel.
    """
    n_layers, n_neurons, activation, base_seed, trial = args
    
    trial_seed = base_seed + trial * 1000
    data = prepare_data_for_trial(trial_seed)
    
    layer_output_sizes = [n_neurons] * n_layers + [1]
    activations = [activation] * n_layers + [Linear()]
    n_params = count_parameters(NETWORK_INPUT_SIZE, layer_output_sizes)
    
    nn = NeuralNetwork(
        network_input_size=NETWORK_INPUT_SIZE,
        layer_output_sizes=layer_output_sizes,
        activations=activations,
        loss=MSE(),
        seed=trial_seed
    )
    
    learning_rate = get_adaptive_learning_rate(n_params, BASE_LEARNING_RATE)
    optimizer = Adam(eta=learning_rate) if OPTIMIZER == 'adam' else RMSprop(eta=learning_rate)
    
    train(
        nn=nn,
        X_train=data['X_train'],
        y_train=data['y_train_scaled'],
        X_val=data['X_val'],
        y_val=data['y_val_scaled'],
        optimizer=optimizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        stochastic=True,
        task='regression',
        early_stopping=False,
        patience=250,
        verbose=False,
        seed=trial_seed
    )
    
    # Make predictions
    y_train_pred = inverse_scale_y(nn.predict(data['X_train']), data['y_mean'])
    y_test_pred = inverse_scale_y(nn.predict(data['X_test']), data['y_mean'])
    
    # ========== ADD THIS SECTION ==========
    # Check for NaN or Inf in predictions
    if (np.any(np.isnan(y_train_pred)) or np.any(np.isinf(y_train_pred)) or
        np.any(np.isnan(y_test_pred)) or np.any(np.isinf(y_test_pred))):
        
        print(f"⚠️  WARNING: NaN/Inf in {n_layers}L×{n_neurons}N trial {trial}")
        
        # Return high penalty loss instead of NaN
        return {
            'n_layers': n_layers,
            'n_neurons': n_neurons,
            'trial': trial,
            'train_loss': 999.0,  # High penalty
            'test_loss': 999.0,
            'n_params': n_params
        }
    
    # Optional: Check for unreasonably high predictions (indicates divergence)
    if (np.abs(y_train_pred).max() > 100 or np.abs(y_test_pred).max() > 100):
        print(f"⚠️  WARNING: Extreme values in {n_layers}L×{n_neurons}N trial {trial}")
        print(f"    Train pred range: [{y_train_pred.min():.2f}, {y_train_pred.max():.2f}]")
        print(f"    Test pred range: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}]")
        
        return {
            'n_layers': n_layers,
            'n_neurons': n_neurons,
            'trial': trial,
            'train_loss': 999.0,
            'test_loss': 999.0,
            'n_params': n_params
        }
    # ========== END NEW SECTION ==========
    
    # Compute losses (only if predictions are valid)
    train_loss = mse(data['y_train_original'], y_train_pred)
    test_loss = mse(data['y_test_original'], y_test_pred)
    
    # Optional: Print progress (comment out if too verbose)
    # print(f"✓ {n_layers}L×{n_neurons}N trial {trial}: Train={train_loss:.6f}, Test={test_loss:.6f}")
    
    return {
        'n_layers': n_layers,
        'n_neurons': n_neurons,
        'trial': trial,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'n_params': n_params
    }


# =============================================================================
#                           ANALYSIS WITH PARALLELIZATION
# =============================================================================

def analyze_activation_parallel(activation, activation_name, base_seed):
    """Test all architectures using parallel processing."""
    
    print(f"\n{'='*80}")
    print(f"Testing {activation_name.upper()} ({N_TRIALS} trials per architecture)")
    print(f"Using {N_WORKERS} parallel workers")
    print(f"{'='*80}")
    
    # Build list of all jobs
    jobs = []
    for n_layers in sorted(NEURONS_CONFIG.keys()):
        for n_neurons in NEURONS_CONFIG[n_layers]:
            for trial in range(N_TRIALS):
                jobs.append((n_layers, n_neurons, activation, base_seed, trial))
    
    print(f"\nTotal jobs: {len(jobs)}")
    print("Training in parallel...")
    
    # Run all jobs in parallel
    with Pool(processes=N_WORKERS) as pool:
        all_trial_results = pool.map(train_single_config, jobs)
    
    print("✓ All training complete!")
    
    # Aggregate results by architecture
    results = {}
    for n_layers in sorted(NEURONS_CONFIG.keys()):
        results[n_layers] = []
        
        for n_neurons in NEURONS_CONFIG[n_layers]:
            # Get all trials for this architecture
            trials_for_arch = [r for r in all_trial_results 
                              if r['n_layers'] == n_layers and r['n_neurons'] == n_neurons]
            
            # Compute statistics
            train_losses = [t['train_loss'] for t in trials_for_arch]
            test_losses = [t['test_loss'] for t in trials_for_arch]
            n_params = trials_for_arch[0]['n_params']
            
            results[n_layers].append({
                'n_params': n_params,
                'train_loss': np.mean(train_losses),
                'test_loss': np.mean(test_losses),
                'train_loss_std': np.std(train_losses),
                'test_loss_std': np.std(test_losses),
                'n_layers': n_layers,
                'n_neurons': n_neurons
            })
    
    return results


# =============================================================================
#                        PLOTTING
# =============================================================================

def plot_combined_complexity_curve(results, activation_name, n_trials=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_results = []
    for n_layers in sorted(results.keys()):
        all_results.extend(results[n_layers])
    
    all_results.sort(key=lambda x: x['n_params'])
    
    n_params = np.array([r['n_params'] for r in all_results])
    train_losses = np.array([r['train_loss'] for r in all_results])
    test_losses = np.array([r['test_loss'] for r in all_results])
    
    has_std = 'train_loss_std' in all_results[0]
    n_params_k = n_params / 1000
    
    ax.plot(n_params_k, test_losses, color='royalblue', label='Test')
    ax.plot(n_params_k, train_losses, color='darkorange', label='Train')
    
    if has_std:
        train_stds = np.array([r['train_loss_std'] for r in all_results])
        test_stds = np.array([r['test_loss_std'] for r in all_results])
        
        ax.fill_between(n_params_k, test_losses - test_stds, test_losses + test_stds,
                       color='royalblue', alpha=0.2, label='Test ±1 std')
        ax.fill_between(n_params_k, train_losses - train_stds, train_losses + train_stds,
                       color='darkorange', alpha=0.15, label='Train ±1 std')

    ax.set_xlabel('Number of parameters/weights (×10³)', fontsize=16)
    ax.set_ylabel('Squared loss', fontsize=16)
    
    title = f'{activation_name} Activation - Model Complexity Analysis'
    if n_trials:
        title += f'\n(averaged over {n_trials} trials)'
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    return fig, ax


# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "="*80)
    print("MODEL COMPLEXITY ANALYSIS - PARALLEL VERSION")
    print("="*80)
    print(f"Data: {N_POINTS} points | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Base LR: {BASE_LEARNING_RATE} | Optimizer: {OPTIMIZER}")
    print(f"Trials per architecture: {N_TRIALS}")
    print(f"Parallel workers: {N_WORKERS}")
    print(f"Split: Train {int((1-TEST_SIZE)*(1-VAL_SIZE)*100)}% | "
          f"Val {int((1-TEST_SIZE)*VAL_SIZE*100)}% | Test {int(TEST_SIZE*100)}%")
    
    all_results = {}
    
    for activation, activation_name in ACTIVATIONS_TO_TEST:
        results = analyze_activation_parallel(activation, activation_name, SEED)
        all_results[activation_name] = results
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_all_results(all_results, save_dir=RESULTS_DIR)
    
    Path(FIGS_DIR).mkdir(exist_ok=True)
    
    for activation_name, results in all_results.items():
        print(f"\nPlotting {activation_name}...")
        fig, ax = plot_combined_complexity_curve(results, activation_name, n_trials=N_TRIALS)
        filename = f"complexity_curve_{activation_name.lower().replace(' ', '_')}.pdf"
        plt.savefig(f"{FIGS_DIR}/{filename}", dpi=300, bbox_inches='tight')
        print(f"  Saved: {FIGS_DIR}/{filename}")
        plt.show()
    
    print("\n" + "="*80)
    print("BEST ARCHITECTURES (averaged over trials)")
    print("="*80)
    
    for activation_name, results in all_results.items():
        all_archs = [r for layer_results in results.values() for r in layer_results]
        best = min(all_archs, key=lambda x: x['test_loss'])
        
        print(f"\n{activation_name}:")
        print(f"  {best['n_layers']} layers × {best['n_neurons']} neurons = {best['n_params']:,} params")
        print(f"  Train: {best['train_loss']:.6f} ± {best['train_loss_std']:.6f}")
        print(f"  Test: {best['test_loss']:.6f} ± {best['test_loss_std']:.6f}")
        print(f"  Gap: {best['test_loss'] - best['train_loss']:.6f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()