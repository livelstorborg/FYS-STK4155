import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Project imports
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, LeakyReLU, Linear
from src.losses import MSE
from src.optimizers import Adam, RMSprop
from src.training import train
from src.metrics import mse
from src.utils import runge, scale_data, inverse_scale_y, save_results_to_csv, save_all_results, load_all_results, print_results_summary
from src.plotting import plot_combined_complexity_curve




# =============================================================================
#                              CONFIGURATION
# =============================================================================

SEED = 42
N_POINTS = 1000
TEST_SIZE = 0.2
NOISE_LEVEL = 0.1

EPOCHS = 1000
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.001
OPTIMIZER = 'adam'

NETWORK_INPUT_SIZE = 1

# TOGGLE THIS FLAG: True for first run (compute), False to load saved results
RUN_EXPERIMENTS = False  # Set to True for first run, False to load saved results
SAVE_DIR = 'results'


# =============================================================================
#                          ARCHITECTURE DEFINITIONS
# =============================================================================

NEURONS_CONFIG = {
    1: [10, 20, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 250, 300],
    2: [20, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 250],
    3: [30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200],
    4: [40, 50, 60, 75, 90, 100, 120, 140, 160],
    5: [40, 50, 60, 75, 90, 100, 120],
}

ACTIVATIONS_TO_TEST = [
    (Sigmoid(), "Sigmoid"),
    (ReLU(), "ReLU"),
    (LeakyReLU(), "Leaky ReLU"),
]


# =============================================================================
#                              HELPER FUNCTIONS
# =============================================================================

def count_parameters(network_input_size, layer_output_sizes):
    """Count total trainable parameters (weights + biases)."""
    total_params = 0
    prev_size = network_input_size
    
    for layer_size in layer_output_sizes:
        total_params += (prev_size * layer_size) + layer_size
        prev_size = layer_size
    
    return total_params


def get_adaptive_learning_rate(n_params, base_lr=0.001):
    """Reduce learning rate for larger networks."""
    if n_params < 5000:
        return base_lr
    elif n_params < 20000:
        return base_lr * 0.5
    else:
        return base_lr * 0.3


def prepare_data():
    """Generate and split Runge function data."""
    np.random.seed(SEED)
    
    x = np.linspace(-1, 1, N_POINTS)
    y_true = runge(x)
    y_noise = y_true + np.random.normal(0, NOISE_LEVEL, N_POINTS)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        x.reshape(-1, 1), y_noise.reshape(-1, 1),
        test_size=TEST_SIZE, random_state=SEED
    )
    
    X_train, y_train_scaled, X_mean, X_std, y_mean = scale_data(X_train_raw, y_train)
    X_test, y_test_scaled, _, _, _ = scale_data(X_test_raw, y_test, X_mean, X_std, y_mean)
    
    return X_train, X_test, y_train_scaled, y_test_scaled, y_train, y_test, y_mean


# =============================================================================
#                           TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train_scaled, y_test_scaled,
                       y_train_original, y_test_original, y_mean,
                       n_layers, n_neurons, activation):
    """Train single architecture and compute metrics on original scale."""
    np.random.seed(SEED)
    
    layer_output_sizes = [n_neurons] * n_layers + [1]
    activations = [activation] * n_layers + [Linear()]
    n_params = count_parameters(NETWORK_INPUT_SIZE, layer_output_sizes)
    
    nn = NeuralNetwork(
        network_input_size=NETWORK_INPUT_SIZE,
        layer_output_sizes=layer_output_sizes,
        activations=activations,
        loss=MSE(),
        seed=SEED,
        weight_init_scale=0.1
    )
    
    learning_rate = get_adaptive_learning_rate(n_params, BASE_LEARNING_RATE)
    optimizer = Adam(eta=learning_rate) if OPTIMIZER == 'adam' else RMSprop(eta=learning_rate)
    
    train(
        nn=nn,
        X_train=X_train,
        y_train=y_train_scaled,
        X_val=X_test,
        y_val=y_test_scaled,
        optimizer=optimizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        stochastic=True,
        task='regression',
        early_stopping=True,
        patience=50,
        verbose=False,
        seed=SEED
    )
    
    y_train_pred = inverse_scale_y(nn.predict(X_train), y_mean)
    y_test_pred = inverse_scale_y(nn.predict(X_test), y_mean)
    
    return {
        'n_params': n_params,
        'train_loss': mse(y_train_original, y_train_pred),
        'test_loss': mse(y_test_original, y_test_pred),
        'n_layers': n_layers,
        'n_neurons': n_neurons
    }


def analyze_activation(activation, activation_name, data):
    """Test all architectures for one activation function."""
    X_train, X_test, y_train_scaled, y_test_scaled, y_train_orig, y_test_orig, y_mean = data
    
    print(f"\n{'='*80}")
    print(f"Testing {activation_name.upper()}")
    print(f"{'='*80}")
    
    results = {}
    
    for n_layers in sorted(NEURONS_CONFIG.keys()):
        results[n_layers] = []
        print(f"\n{n_layers} layer(s):", end=" ")
        
        for n_neurons in NEURONS_CONFIG[n_layers]:
            result = train_and_evaluate(
                X_train, X_test, y_train_scaled, y_test_scaled,
                y_train_orig, y_test_orig, y_mean,
                n_layers, n_neurons, activation
            )
            results[n_layers].append(result)
            print(".", end="", flush=True)
        
        print(f" ✓")
    
    return results


# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def main():
    
    print("\n" + "="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    
    if RUN_EXPERIMENTS:
        # ========== FIRST RUN: COMPUTE AND SAVE ==========
        print("Running experiments...")
        print(f"Data: {N_POINTS} points | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
        print(f"Base LR: {BASE_LEARNING_RATE} | Optimizer: {OPTIMIZER}")
        
        data = prepare_data()
        print(f"Split: {len(data[0])} train, {len(data[1])} test")
        
        all_results = {}
        
        for activation, activation_name in ACTIVATIONS_TO_TEST:
            results = analyze_activation(activation, activation_name, data)
            all_results[activation_name] = results
        
        # Save all results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        saved_files = save_all_results(all_results, save_dir=SAVE_DIR)
        
    else:
        # ========== SUBSEQUENT RUNS: LOAD FROM FILES ==========
        print("Loading saved results...")
        all_results = load_all_results(save_dir=SAVE_DIR)
        
        # Print summaries
        for activation_name, results in all_results.items():
            print_results_summary(results, activation_name)
    
    
    # ========== PLOTTING (works for both cases) ==========
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    for activation_name, results in all_results.items():
        print(f"\nPlotting {activation_name}...")
        plot_combined_complexity_curve(results, activation_name)
        plt.savefig(f"figs/complexity_curve_{activation_name.lower().replace(' ', '_')}.pdf", dpi=300)
        plt.show()
    
    
    # ========== SUMMARY OF BEST ARCHITECTURES ==========
    print("\n" + "="*80)
    print("BEST ARCHITECTURES")
    print("="*80)
    
    for activation_name, results in all_results.items():
        all_archs = [r for layer_results in results.values() for r in layer_results]
        best = min(all_archs, key=lambda x: x['test_loss'])
        
        print(f"\n{activation_name}:")
        print(f"  {best['n_layers']} layers × {best['n_neurons']} neurons = {best['n_params']:,} params")
        print(f"  Train: {best['train_loss']:.6f} | Test: {best['test_loss']:.6f} | Gap: {best['test_loss'] - best['train_loss']:.6f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()