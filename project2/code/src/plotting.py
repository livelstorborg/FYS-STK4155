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
    if epochs is None:
        epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.show()


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

# d)
def plot_complexity_curve(results, activation_name):
    """
    Plot train/test loss vs number of parameters for different depths (number of layers).
    
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
    
    # Colors for different depths
    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange', 5: 'purple'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    for n_layers in sorted(results.keys()):
        layer_results = results[n_layers]
        
        # Extract data
        n_params = [r['n_params'] for r in layer_results]
        train_losses = [r['train_loss'] for r in layer_results]
        test_losses = [r['test_loss'] for r in layer_results]
        
        # Convert to thousands
        n_params_k = [p / 1000 for p in n_params]
        
        # Plot train and test
        ax.plot(n_params_k, train_losses, 
                color=colors.get(n_layers, 'black'), 
                marker=markers.get(n_layers, 'o'),
                label=f'{n_layers} layer(s) - Train')
        
        ax.plot(n_params_k, test_losses,
                color=colors.get(n_layers, 'black'), 
                marker=markers.get(n_layers, 'o'),
                label=f'{n_layers} layer(s) - Test')
    
    ax.set_xlabel('Number of Parameters (×10³)', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_title(f'{activation_name} - Model Complexity vs Performance', 
                 fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax

# d)
def plot_combined_complexity_curve(results, activation_name):
    """
    Plotting train/test loss vs number of parameters, combining all depths.
    
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
    n_params = [r['n_params'] for r in all_results]
    train_losses = [r['train_loss'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    
    # Convert to thousands
    n_params_k = [p / 1000 for p in n_params]
    
    # Plot (style similar to reference image)
    ax.plot(n_params_k, test_losses, 
            marker='o', markersize=8, linewidth=2,
            label='Test')
    
    ax.plot(n_params_k, train_losses, 
            marker='o', markersize=8, linewidth=2,
            label='Train')

    ax.set_xlabel('Number of parameters/weights (×10³)', fontsize=16)
    ax.set_ylabel('Squared loss', fontsize=16)
    ax.set_title(f'{activation_name} Activation - Model Complexity Analysis', 
                 fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    return fig, ax