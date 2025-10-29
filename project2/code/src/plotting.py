import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.patches import Rectangle


plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

os.makedirs("figs", exist_ok=True)

# b)
def plot_learning_curves(train_loss, val_loss, epochs=None, title='Learning Curves'):
    """
    Simple learning curve plot.
    
    Parameters:
    -----------
    train_loss : array-like
        Training loss values
    val_loss : array-like
        Validation loss values
    epochs : array-like, optional
        Epoch numbers. If None, uses 1, 2, 3, ...
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    if epochs is None:
        epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label='Train')
    ax.plot(epochs, val_loss, label='Validation')
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    return fig, ax


# e)
def lambda_eta_heatmap(metric_array, eta_vals, lambda_vals, 
                       metric_name='MSE', dataset='Train',
                       cmap='viridis', figsize=(10, 8), annot=True,
                       maximize=False):

    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        metric_array,
        annot=annot,
        fmt='.4f' if annot else None,
        cmap=cmap,
        ax=ax,
        xticklabels=[f'{int(np.log10(lam))}' for lam in lambda_vals],
        yticklabels=[f'{int(np.log10(eta))}' for eta in eta_vals],
        cbar_kws={'label': metric_name}
    )
    
    # Find best value location
    if maximize:
        best_idx = np.unravel_index(np.argmax(metric_array), metric_array.shape)
    else:
        best_idx = np.unravel_index(np.argmin(metric_array), metric_array.shape)
    
    i_best, j_best = best_idx
    
    # Add red box around best cell
    rect = Rectangle((j_best, i_best), 1, 1, 
                     linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title(f'{dataset} {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=16)
    ax.set_ylabel(r'$\log_{10}(\eta)$', fontsize=16)
    
    plt.tight_layout()
    
    return fig, ax


# b)
def N_eta_heatmap(metric_array, N_vals, eta_vals,
                  metric_name='MSE', dataset='Test',
                  cmap='viridis', figsize=(10, 8), annot=False, title='MSE'):
    """
    Create a heatmap showing MSE for different N and eta combinations.
    
    Parameters:
    -----------
    metric_array : 2D array (n_etas x n_Ns)
        MSE values to plot
    N_vals : array
        Sample sizes
    eta_vals : array
        Learning rates
    metric_name : str
        Name of metric for colorbar
    dataset : str
        'Test' or 'Train' for title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with grid lines
    im = sns.heatmap(
        metric_array,
        annot=annot,
        fmt='.4f' if annot else None,
        cmap=cmap,
        ax=ax,
        xticklabels=[f'{N}' for N in N_vals],
        yticklabels=[f'{eta:.1e}' for eta in eta_vals],
        cbar_kws={'label': metric_name},
        linewidths=0.2,
        linecolor=(1, 1, 1, 0.2)
    )
    
    # Modify colorbar tick label size
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    # Find best value location (minimum for MSE)
    best_idx = np.unravel_index(np.argmin(metric_array), metric_array.shape)
    i_best, j_best = best_idx
    
    # Add red box around best cell
    rect = Rectangle((j_best, i_best), 1, 1, 
                     facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    # Set title and labels with enhanced formatting
    ax.set_title(f'{dataset} {metric_name} ({title})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Sample Size (N)', fontsize=16)
    ax.set_ylabel(r'Learning Rate ($\eta$)', fontsize=16)
    
    # Set tick parameters
    ax.tick_params(axis='x', rotation=0, labelsize=16)
    ax.tick_params(axis='y', rotation=0, labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax

def plot_complexity_curve(results, activation_name, n_trials=None):
    """
    Plot train/test loss vs number of parameters for different depths (number of layers).
    Includes shaded regions for ±1 standard deviation if std data is available.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_activation()
    activation_name : str
        Name of the activation function
    n_trials : int, optional
        Number of trials (for title). If None, not shown.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different depths
    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange', 5: 'purple'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # Check if std data is available
    first_layer = list(results.keys())[0]
    has_std = 'train_loss_std' in results[first_layer][0]
    
    for n_layers in sorted(results.keys()):
        layer_results = results[n_layers]
        
        # Extract data
        n_params = np.array([r['n_params'] for r in layer_results])
        train_losses = np.array([r['train_loss'] for r in layer_results])
        test_losses = np.array([r['test_loss'] for r in layer_results])
        
        # Convert to thousands
        n_params_k = n_params / 1000
        
        color = colors.get(n_layers, 'black')
        marker = markers.get(n_layers, 'o')
        
        # Plot test line
        ax.plot(n_params_k, test_losses,
                color=color, marker=marker, linewidth=2.5, markersize=8,
                label=f'{n_layers} layer(s) - Test')
        
        # Plot train line
        ax.plot(n_params_k, train_losses, 
                color=color, marker=marker, linewidth=2.5, markersize=8,
                linestyle='--', alpha=0.7,
                label=f'{n_layers} layer(s) - Train')
        
        # Add shaded std regions if available
        if has_std:
            train_stds = np.array([r['train_loss_std'] for r in layer_results])
            test_stds = np.array([r['test_loss_std'] for r in layer_results])
            
            # Shaded region for test
            ax.fill_between(n_params_k,
                           test_losses - test_stds,
                           test_losses + test_stds,
                           color=color, alpha=0.15)
            
            # Shaded region for train
            ax.fill_between(n_params_k,
                           train_losses - train_stds,
                           train_losses + train_stds,
                           color=color, alpha=0.1)
    
    ax.set_xlabel('Number of Parameters (×10³)', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    
    # Title with trial info if available
    title = f'{activation_name} - Model Complexity vs Performance'
    if n_trials:
        title += f' (averaged over {n_trials} trials)'
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax




def plot_combined_complexity_curve(results, activation_name, n_trials=None):
    """
    Plotting train/test loss vs number of parameters, combining all depths.
    Includes shaded regions for ±1 standard deviation if std data is available.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_activation()
    activation_name : str
        Name of the activation function
    n_trials : int, optional
        Number of trials (for title). If None, not shown.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect ALL results across all depths
    all_results = []
    for n_layers in sorted(results.keys()):
        all_results.extend(results[n_layers])
    
    # Sort by number of parameters
    all_results.sort(key=lambda x: x['n_params'])
    
    # Extract data
    n_params = np.array([r['n_params'] for r in all_results])
    train_losses = np.array([r['train_loss'] for r in all_results])
    test_losses = np.array([r['test_loss'] for r in all_results])
    
    # Check if std data is available
    has_std = 'train_loss_std' in all_results[0]
    
    # Convert to thousands
    n_params_k = n_params / 1000
    
    # Plot test line
    ax.plot(n_params_k, test_losses, 
            marker='o', markersize=8, linewidth=2.5,
            color='royalblue', label='Test')
    
    # Plot train line
    ax.plot(n_params_k, train_losses, 
            marker='o', markersize=8, linewidth=2.5,
            color='darkorange', label='Train')
    
    # Add shaded std regions if available
    if has_std:
        train_stds = np.array([r['train_loss_std'] for r in all_results])
        test_stds = np.array([r['test_loss_std'] for r in all_results])
        
        # Shaded region for test (±1 std)
        ax.fill_between(n_params_k,
                       test_losses - test_stds,
                       test_losses + test_stds,
                       color='royalblue', alpha=0.2, label='Test ±1 std')
        
        # Shaded region for train (±1 std)
        ax.fill_between(n_params_k,
                       train_losses - train_stds,
                       train_losses + train_stds,
                       color='darkorange', alpha=0.15, label='Train ±1 std')

    ax.set_xlabel('Number of parameters/weights (×10³)', fontsize=16)
    ax.set_ylabel('Squared loss', fontsize=16)
    
    # Title with trial info if available
    title = f'{activation_name} Activation - Model Complexity Analysis'
    if n_trials:
        title += f'\n(averaged over {n_trials} trials)'
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax


def plot_combined_complexity_simple(results, activation_name):
    """
    Simple version without std shading (for single-trial results).
    
    Parameters:
    -----------
    results : dict
        Results from analyze_activation()
    activation_name : str
        Name of the activation function
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect ALL results across all depths
    all_results = []
    for n_layers in sorted(results.keys()):
        all_results.extend(results[n_layers])
    
    # Sort by number of parameters
    all_results.sort(key=lambda x: x['n_params'])
    
    # Extract data
    n_params_k = [r['n_params'] / 1000 for r in all_results]
    train_losses = [r['train_loss'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    
    # Plot (style similar to reference image)
    ax.plot(n_params_k, test_losses, 
            marker='o', markersize=8, linewidth=2.5,
            color='royalblue', label='Test')
    
    ax.plot(n_params_k, train_losses, 
            marker='o', markersize=8, linewidth=2.5,
            color='darkorange', label='Train')

    ax.set_xlabel('Number of parameters/weights (×10³)', fontsize=16)
    ax.set_ylabel('Squared loss', fontsize=16)
    ax.set_title(f'{activation_name} Activation - Model Complexity Analysis', 
                 fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax
