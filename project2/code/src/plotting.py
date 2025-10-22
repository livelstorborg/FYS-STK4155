import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.patches import Rectangle

# Match Project 1 style
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

os.makedirs("figs", exist_ok=True)


def plot_learning_curves(train_loss, val_loss, epochs=None):
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
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# For exercise e)
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


# ========================================
# PART A) 
# ========================================

def plot_activations():
    """Plot activation functions and derivatives for Part a)."""
    from .activations import Sigmoid, ReLU, LeakyReLU
    
    z = np.linspace(-5, 5, 100)
    sigmoid = Sigmoid()
    relu = ReLU()
    leaky = LeakyReLU(alpha=0.01)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Sigmoid
    axes[0, 0].plot(z, sigmoid.forward(z), linewidth=2, color='darkviolet')
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(z, sigmoid.backward(z), linewidth=2, color='#D63290')
    axes[1, 0].set_title("Sigmoid'")
    axes[1, 0].grid(True, alpha=0.3)
    
    # ReLU
    axes[0, 1].plot(z, relu.forward(z), linewidth=2, color='darkviolet')
    axes[0, 1].set_title('ReLU')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(z, relu.backward(z), linewidth=2, color='#D63290')
    axes[1, 1].set_title("ReLU'")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Leaky ReLU
    axes[0, 2].plot(z, leaky.forward(z), linewidth=2, color='darkviolet')
    axes[0, 2].set_title('Leaky ReLU')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].plot(z, leaky.backward(z), linewidth=2, color='#D63290')
    axes[1, 2].set_title("Leaky ReLU'")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/activations.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# ========================================
# PART B) 
# ========================================

def plot_predictions(X_test, y_test, y_pred, title='NN Predictions'):
    """Plot NN predictions vs true values."""
    plt.figure(figsize=(10, 6))
    
    # True function
    x_smooth = np.linspace(-1, 1, 200)
    y_smooth = 1 / (1 + 25 * x_smooth**2)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='True function')
    
    # Predictions (sorted for smooth line)
    sort_idx = np.argsort(X_test.ravel())
    plt.plot(X_test[sort_idx], y_pred[sort_idx], linewidth=2.5, 
             color='darkviolet', label='NN predictions')
    
    # Test data
    plt.scatter(X_test, y_test, alpha=0.6, s=50, color='#F2B44D',
                edgecolors='black', linewidths=0.5, label='Test data')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], linewidth=2, color='darkviolet', label='Train')
    ax1.plot(history['val_loss'], linewidth=2, color='#D63290', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE
    ax2.plot(history['train_metric'], linewidth=2, color='darkviolet', label='Train')
    ax2.plot(history['val_metric'], linewidth=2, color='#D63290', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()