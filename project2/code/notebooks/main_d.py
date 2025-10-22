import sys, os
sys.path.insert(0, '/Users/livestorborg/Desktop/FYS-STK4155/project2/code')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, LeakyReLU, Linear
from src.losses import MSE
from src.optimizers import Adam, RMSprop
from src.training import train
from src.metrics import mse
from src.utils import runge, scale_data, inverse_scale_y

# Setup
SEED = 42
np.random.seed(SEED)

N = 100
x = np.linspace(-1, 1, N)
y_true = runge(x)
y_noise = y_true + np.random.normal(0, 0.1, N)

# Split and scale data
X_train_raw, X_test_raw, y_train_nn, y_test_nn = train_test_split(
    x.reshape(-1, 1), y_noise.reshape(-1, 1), 
    test_size=0.2, random_state=SEED
)

X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train_raw, y_train_nn)
X_test_s, y_test_s, _, _, _ = scale_data(X_test_raw, y_test_nn, X_mean, X_std, y_mean)

# Compute real y values
y_train_real = inverse_scale_y(y_train_s, y_mean)
y_test_real = inverse_scale_y(y_test_s, y_mean)

# Define your architecture grid
n_layers_list = [1, 2, 3, 4]
n_neurons_list = [10, 25, 50, 100, 150]

# Map activation names to classes
activation_map = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'leaky_relu': LeakyReLU()
}

# Best hyperparameters from part b) - UPDATE THESE!
best_eta = 0.001  # Replace with your best from part b)
best_optimizer = 'adam'  # or 'rmsprop'

# Storage for results
results = {}
all_models = {}

# Loop through each activation function
for activation_name, activation_func in activation_map.items():
    print(f"\n{'='*60}")
    print(f"Training with {activation_name.upper()} activation")
    print(f"{'='*60}")
    
    # Create a matrix to store MSE values
    train_mse_matrix = np.zeros((len(n_layers_list), len(n_neurons_list)))
    test_mse_matrix = np.zeros((len(n_layers_list), len(n_neurons_list)))
    
    # Store models for this activation
    models_grid = [[None for _ in range(len(n_neurons_list))] 
                   for _ in range(len(n_layers_list))]
    
    # Loop through architectures
    for i, n_layers in enumerate(n_layers_list):
        for j, n_neurons in enumerate(n_neurons_list):
            
            # Define the architecture
            hidden_layers = [n_neurons] * n_layers
            
            # Create list of activations: hidden layers + output layer
            activations = [activation_func for _ in range(n_layers)] + [Linear()]
            
            # Initialize neural network
            model = NeuralNetwork(
                network_input_size=1,
                layer_output_sizes=hidden_layers + [1],
                activations=activations,
                loss=MSE(),
                seed=SEED,
                lambda_reg=0.0,  # No regularization in part d)
                reg_type=None,
                weight_init='xavier'
            )
            
            # Choose optimizer
            if best_optimizer == 'adam':
                optimizer = Adam(eta=best_eta)
            else:
                optimizer = RMSprop(eta=best_eta)
            
            # Train the network
            train(
                nn=model,
                X_train=X_train_s,
                y_train=y_train_s,
                X_val=X_test_s,
                y_val=y_test_s,
                optimizer=optimizer,
                epochs=500,
                batch_size=32,
                stochastic=True,
                task='regression',
                early_stopping=True,
                patience=50,
                verbose=False,
                seed=SEED
            )
            
            # Store model
            models_grid[i][j] = model
            
            # Evaluate on both train and test
            y_train_pred = inverse_scale_y(model.predict(X_train_s), y_mean)
            y_test_pred = inverse_scale_y(model.predict(X_test_s), y_mean)
            
            train_mse_val = mse(y_train_real, y_train_pred)
            test_mse_val = mse(y_test_real, y_test_pred)
            
            # Store results
            train_mse_matrix[i, j] = train_mse_val
            test_mse_matrix[i, j] = test_mse_val
            
            print(f"Layers: {n_layers}, Neurons: {n_neurons:3d} | "
                  f"Train MSE: {train_mse_val:.6f}, Test MSE: {test_mse_val:.6f}")
    
    # Store results for this activation function
    results[activation_name] = {
        'train_mse': train_mse_matrix,
        'test_mse': test_mse_matrix,
        'models': models_grid
    }
    all_models[activation_name] = models_grid

# Save results
np.save('architecture_search_results.npy', results)

# Create heatmaps for TEST MSE
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, activation_name in enumerate(['sigmoid', 'relu', 'leaky_relu']):
    ax = axes[idx]
    
    # Create heatmap
    sns.heatmap(
        results[activation_name]['test_mse'],
        annot=True,
        fmt='.4f',
        cmap='viridis_r',  # Darker = better (lower MSE)
        xticklabels=n_neurons_list,
        yticklabels=n_layers_list,
        ax=ax,
        cbar_kws={'label': 'Test MSE'}
    )
    
    ax.set_xlabel('Number of Neurons per Layer', fontsize=12)
    ax.set_ylabel('Number of Hidden Layers', fontsize=12)
    ax.set_title(f'{activation_name.upper()} Activation', fontsize=14)

plt.tight_layout()
plt.savefig('activation_architecture_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Find and print best architecture for each activation function
print("\n" + "="*60)
print("BEST ARCHITECTURES")
print("="*60)

for activation_name in ['sigmoid', 'relu', 'leaky_relu']:
    test_mse_matrix = results[activation_name]['test_mse']
    train_mse_matrix = results[activation_name]['train_mse']
    
    min_idx = np.unravel_index(np.argmin(test_mse_matrix), test_mse_matrix.shape)
    best_layers = n_layers_list[min_idx[0]]
    best_neurons = n_neurons_list[min_idx[1]]
    best_test_mse = test_mse_matrix[min_idx]
    best_train_mse = train_mse_matrix[min_idx]
    
    print(f"\n{activation_name.upper()}:")
    print(f"  Best architecture: {best_layers} layer(s) × {best_neurons} neurons")
    print(f"  Best train MSE: {best_train_mse:.6f}")
    print(f"  Best test MSE:  {best_test_mse:.6f}")
    print(f"  Difference (overfitting indicator): {best_train_mse - best_test_mse:.6f}")