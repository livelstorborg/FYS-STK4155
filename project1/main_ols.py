import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from src.plotting import mse_degree_ols, r2_degree_ols
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


# ============================================================================
# CONFIGURATION
# ============================================================================


SAMPLE_SIZES = [34, 50, 100, 200, 300, 350, 500, 1000]
DEGREES = range(1, 16)
PLOT_SAMPLE_SIZE = 50  # Sample size for individual plots


# ============================================================================
# DATA GENERATION AND MODEL FITTING
# ============================================================================
all_results = {}

for N in SAMPLE_SIZES:
    # Generate data
    np.random.seed(42)
    x = np.linspace(-1, 1, N)
    random_noise = np.random.normal(0, 0.1, N)
    y_true = runge(x)
    y_noise = y_true + random_noise
    
    # Prepare data splits for each degree
    data_splits = {}
    for deg in DEGREES:
        X = polynomial_features(x, deg)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_noise, test_size=0.25, random_state=42
        )
        
        # Scale data
        X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
        X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)
        
        x_train = X_train[:, 0]
        x_test = X_test[:, 0]
        data_splits[deg] = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]
    
    # Fit models for all degrees
    results_current = {
        "degrees": list(DEGREES),
        "train_mse": [],
        "test_mse": [],
        "train_r2": [],
        "test_r2": [],
        "instances": [],
    }
    
    for deg in DEGREES:
        analysis = RegressionAnalysis(
            data_splits[deg], degree=deg, lam=0.0, eta=None, num_iters=None
        )
        analysis.fit(models='ols', opts='analytical')
        
        results_current["train_mse"].append(analysis.get_metric('ols', 'analytical', 'train_mse'))
        results_current["test_mse"].append(analysis.get_metric('ols', 'analytical', 'test_mse'))
        results_current["train_r2"].append(analysis.get_metric('ols', 'analytical', 'train_r2'))
        results_current["test_r2"].append(analysis.get_metric('ols', 'analytical', 'test_r2'))
        results_current["instances"].append(analysis)
    

    all_results[N] = results_current
    



    if N == 50: 
        mse_degree_ols(results_current)
        r2_degree_ols(results_current)




# ============================================================================
# HEATMAP: MSE vs SAMPLE SIZE AND DEGREE
# ============================================================================

# Prepare data for heatmap
heatmap_data = []
for N in SAMPLE_SIZES:
    for i, deg in enumerate(all_results[N]["degrees"]):
        heatmap_data.append({
            'Sample Size': N,
            'Degree': deg,
            'Test MSE': all_results[N]["test_mse"][i]
        })

df = pd.DataFrame(heatmap_data)
heatmap_matrix = df.pivot(index='Sample Size', columns='Degree', values='Test MSE')

# Find minimum MSE position
min_val = heatmap_matrix.min().min()
min_pos = np.where(heatmap_matrix == min_val)
min_row, min_col = min_pos[0][0], min_pos[1][0]

# Create heatmap
fig, ax = plt.subplots(figsize=(16, 10))
hm = sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt='.3f',
    cmap='plasma',
    cbar_kws={'label': 'Test MSE'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax,
    annot_kws={'fontsize': 16}
)

# Format colorbar
cbar = hm.collections[0].colorbar
cbar.set_label('Test MSE', fontsize=20)
cbar.ax.tick_params(labelsize=20)

# Format axes
ax.tick_params(axis='both', labelsize=20)
plt.xlabel('Polynomial Degree', fontsize=20)
plt.ylabel('Sample Size', fontsize=20)

# Highlight minimum MSE
rect = patches.Rectangle(
    (min_col, min_row), 1, 1,
    linewidth=3, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)

plt.tight_layout()
plt.savefig('figs/mse_heatmap_ols.pdf')
plt.show()