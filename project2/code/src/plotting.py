import matplotlib.pyplot as plt
import numpy as np
import os

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