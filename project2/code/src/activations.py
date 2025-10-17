# src/activations.py
import numpy as np

class Activation:
    """Base class for activation functions."""
    
    def forward(self, z):
        """Apply activation function."""
        raise NotImplementedError
    
    def backward(self, z):
        """Compute derivative of activation function."""
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class ReLU(Activation):
    def forward(self, z):
        return np.maximum(0, z)
    
    def backward(self, z):
        return (z > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)
    
    def backward(self, z):
        return np.where(z > 0, 1, self.alpha)


class Linear(Activation):
    """Identity activation for output layer in regression."""
    def forward(self, z):
        return z
    
    def backward(self, z):
        return np.ones_like(z)


class Softmax(Activation):
    """Softmax for multiclass classification output layer."""
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def backward(self, z):
        # For softmax + cross-entropy, derivative is handled in loss function
        return np.ones_like(z)