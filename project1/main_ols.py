import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.plotting import (
    mse_degree_ols,
    r2_degree_ols,
    theta_evolution_ols,
    mse_degree_multiple,
)
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


sample_sizes = [35, 50, 100, 200, 300, 350, 500, 1000]

all_results = {}

for N in sample_sizes:

    x = np.linspace(-1, 1, N)
    np.random.seed(42)
    random_noise = np.random.normal(0, 0.1, N)
    y_true = runge(x)
    y_noise = y_true + random_noise
    degrees = range(1, 16)

    # Split data once for all degrees to ensure consistency
    data_splits = {}  # Dictionary to store splits for each degree
    for deg in degrees:
        X = polynomial_features(x, deg)
        X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.3, random_state=42)
        x_train = X_train[:, 0] 
        x_test = X_test[:, 0] 

        # Scaling the training data and using the same parameters to scale the test data (to avoid data leakage)
        X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
        X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)

        data_splits[deg] = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]

    # Create instances for all degrees
    results_current = {
        "degrees": list(degrees),
        "train_mse": [],
        "test_mse": [],
        "train_r2": [],
        "test_r2": [],
        "instances": [],
    }

    # theta_evolution = []  # Initialize for all N, but only use for N=1000
    
    theta = []
    for deg in degrees:
        analysis = RegressionAnalysis(
            data_splits[deg], degree=deg, lam=0.0, eta=None, num_iters=None
        )

        analysis.fit(models='ols', opts='analytical')

        results_current["train_mse"].append(analysis.get_metric('ols', 'analytical', 'train_mse'))
        results_current["test_mse"].append(analysis.get_metric('ols', 'analytical', 'test_mse'))
        results_current["train_r2"].append(analysis.get_metric('ols', 'analytical', 'train_r2'))
        results_current["test_r2"].append(analysis.get_metric('ols', 'analytical', 'test_r2'))
        results_current["instances"].append(analysis)

        
        
        # theta = analysis.get_theta('ols', 'analytical')
        # theta_norm = np.linalg.norm(theta)
        # theta_evolution.append(theta_norm)
        # theta.append(analysis.get_theta('ols', 'analytical'))

    all_results[N] = results_current




    if N == 50:
        mse_degree_ols(all_results, sample_size=N)  # MSE for N=1000 (a)
        r2_degree_ols(results_current, sample_size=N)  # R2 for N=1000 (a)
        # theta_evolution_ols(
        #     degrees, theta_evolution, sample_size=N
        # )  # Evolution of theta1 for N=1000 (a)

# for t in theta:
#     plt.plot(t/np.linalg.norm(t))

# plt.show()





import seaborn as sns
import pandas as pd
import matplotlib.patches as patches

# Extract data from all_results into a DataFrame
heatmap_data = []
for N in sample_sizes:
    for i, deg in enumerate(all_results[N]["degrees"]):
        heatmap_data.append({
            'Sample Size': N,
            'Degree': deg,
            'Test MSE': all_results[N]["test_mse"][i]
        })

df = pd.DataFrame(heatmap_data)

# Pivot to get matrix format
heatmap_matrix = df.pivot(index='Sample Size', columns='Degree', values='Test MSE')

# Find min MSE position
min_val = heatmap_matrix.min().min()
min_pos = np.where(heatmap_matrix == min_val)
min_row, min_col = min_pos[0][0], min_pos[1][0]

# Create heatmap
fig, ax = plt.subplots(figsize=(16, 10))
hm = sns.heatmap(heatmap_matrix, 
            annot=True,
            fmt='.3f',
            cmap='plasma',
            cbar_kws={'label': 'Test MSE'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws={'fontsize': 16})  # Annotation fontsize

# Set colorbar label and tick fontsize
cbar = hm.collections[0].colorbar
cbar.set_label('Test MSE', fontsize=20)
cbar.ax.tick_params(labelsize=20)

# Set axis tick label fontsize
ax.tick_params(axis='both', labelsize=20)

rect = patches.Rectangle((min_col, min_row), 1, 1, 
                         linewidth=3, edgecolor='red', facecolor='none')
ax.add_patch(rect)

plt.xlabel('Polynomial Degree', fontsize=20)
plt.ylabel('Sample Size', fontsize=20)
plt.tight_layout()
plt.savefig('figs/mse_heatmap_ols.pdf')
plt.show()