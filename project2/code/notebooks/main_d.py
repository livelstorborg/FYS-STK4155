import sys
sys.path.insert(0, '/Users/livestorborg/Desktop/FYS-STK4155/project2/code')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, ReLU, LeakyReLU, Linear
from src.losses import MSE
from src.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# =========================================================
#          Prepare Data (1D Runge Function)
# =========================================================
x = np.linspace(-1, 1, 300).reshape(-1, 1)
y = 1 / (1 + 25 * x**2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("=" * 60)


# =========================================================
#          Define Architectures to Test
# =========================================================
# Strategy: Funnel pattern with varying depths
architectures = {
    'shallow_wide': [100],
    'medium_constant': [50, 50],
    'medium_funnel': [100, 50],
    'deep_funnel': [150, 100, 50],
    'very_deep': [200, 150, 100, 50, 25]
}


# =========================================================
#          Define Activation Functions to Test
# =========================================================
activation_configs = {
    'sigmoid': Sigmoid,
    'relu': ReLU,
    'leaky_relu': LeakyReLU
}


# =========================================================
#          Hyperparameters (keep constant across experiments)
# =========================================================
LEARNING_RATE = 0.001  # Adam typically uses smaller learning rates
EPOCHS = 500
BATCH_SIZE = 32


# =========================================================
#          Run Experiments
# =========================================================
results = []

for arch_name, hidden_layers in architectures.items():
    for act_name, ActivationClass in activation_configs.items():
        
        print(f"\nTesting: {arch_name} with {act_name}")
        print(f"Architecture: {hidden_layers}")
        
        # Create activation functions for each hidden layer + output layer
        activations = [ActivationClass() for _ in hidden_layers] + [Linear()]
        
        # Initialize neural network
        nn = NeuralNetwork(
            network_input_size=1,
            layer_output_sizes=hidden_layers + [1], 
            activations=activations,
            loss_fn=MSE(),
            seed=SEED
        )
        
        # Initialize optimizer (fresh for each experiment)
        optimizer = Adam(eta=LEARNING_RATE)
        
        # Train the network
        train_losses = []
        for epoch in range(EPOCHS):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Compute gradients and update using Adam
                gradients = nn.compute_gradient(X_batch, y_batch)
                optimizer.update(nn, gradients)
            
            # Track training loss every 100 epochs
            if epoch % 100 == 0:
                y_pred_train = nn.predict(X_train)
                train_loss = np.mean((y_train - y_pred_train)**2)
                train_losses.append(train_loss)
                print(f"  Epoch {epoch}: Train MSE = {train_loss:.6f}")
        
        # Evaluate on train and test sets
        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)
        
        mse_train = np.mean((y_train - y_pred_train)**2)
        mse_test = np.mean((y_test - y_pred_test)**2)
        overfit_gap = mse_test - mse_train
        
        print(f"  Final Train MSE: {mse_train:.6f}")
        print(f"  Final Test MSE:  {mse_test:.6f}")
        print(f"  Overfit gap:     {overfit_gap:.6f}")
        
        # Store results
        results.append({
            'architecture': arch_name,
            'num_layers': len(hidden_layers),
            'total_nodes': sum(hidden_layers),
            'layer_sizes': str(hidden_layers),
            'activation': act_name,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'overfit_gap': overfit_gap,
            'ratio_test_train': mse_test / mse_train if mse_train > 0 else np.inf
        })

print("\n" + "=" * 60)
print("All experiments completed!")
print("=" * 60)


# =========================================================
#          Analyze Results
# =========================================================
df_results = pd.DataFrame(results)

# Sort by test MSE
df_sorted = df_results.sort_values('mse_test')

print("\n" + "=" * 60)
print("RESULTS SUMMARY (sorted by test MSE)")
print("=" * 60)
print(df_sorted.to_string(index=False))

print("\n" + "=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)
best = df_sorted.iloc[0]
print(f"Architecture: {best['architecture']}")
print(f"Layers: {best['layer_sizes']}")
print(f"Activation: {best['activation']}")
print(f"Train MSE: {best['mse_train']:.6f}")
print(f"Test MSE:  {best['mse_test']:.6f}")
print(f"Overfit gap: {best['overfit_gap']:.6f}")

print("\n" + "=" * 60)
print("SIGNS OF OVERFITTING")
print("=" * 60)
# Flag configurations with significant overfitting (test/train ratio > 1.5)
overfitting = df_results[df_results['ratio_test_train'] > 1.5]
if len(overfitting) > 0:
    print("Configurations showing overfitting:")
    print(overfitting[['architecture', 'activation', 'mse_train', 'mse_test', 'ratio_test_train']].to_string(index=False))
else:
    print("No significant overfitting detected in any configuration.")


# =========================================================
#          Visualizations
# =========================================================

# 1. Heatmap: Architecture vs Activation (Test MSE)
plt.figure(figsize=(10, 6))
pivot = df_results.pivot_table(
    values='mse_test',
    index='architecture',
    columns='activation'
)
sns.heatmap(pivot, annot=True, fmt='.6f', cmap='viridis_r', cbar_kws={'label': 'Test MSE'})
plt.title('Test MSE by Architecture and Activation Function')
plt.tight_layout()
# plt.savefig('heatmap_architecture_activation.png', dpi=300)
plt.show()

