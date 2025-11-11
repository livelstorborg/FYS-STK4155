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
        """
        WARNING: Softmax.backward() should not be called directly when using CrossEntropy loss!
        
        The combined derivative of Softmax + CrossEntropy simplifies to (y_pred - y_true),
        which is handled automatically in the CrossEntropy.backward() method.
        
        If you need the Jacobian of Softmax alone, you must implement it separately.
        """
        raise NotImplementedError(
            "Softmax.backward() is not implemented. When using Softmax with CrossEntropy loss, "
            "the combined derivative is handled by CrossEntropy.backward(). "
            "Do not call this method directly."
        )


class BatchNorm(Activation):
    """
    Batch Normalization layer.

    Normalizes inputs across the batch dimension and applies learnable
    scale (gamma) and shift (beta) parameters.
    """

    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        Args:
            num_features: Number of features (neurons) to normalize
            momentum: Momentum for running statistics
            epsilon: Small constant for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Training mode flag
        self.training = True

        # Cache for backward pass
        self.cache = None

        # Gradient buffers (populated during backward)
        self.dgamma = None
        self.dbeta = None

    def forward(self, z):
        """
        Forward pass with different behavior for training vs inference.

        Args:
            z: Input of shape (batch_size, num_features)

        Returns:
            Normalized and scaled output
        """
        if self.training:
            # Training: use batch statistics
            batch_mean = np.mean(z, axis=0)
            batch_var = np.var(z, axis=0)

            # Normalize
            z_normalized = (z - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running statistics
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            # Cache for backward
            self.cache = (z, z_normalized, batch_mean, batch_var)
        else:
            # Inference: use running statistics
            z_normalized = (z - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )

        # Scale and shift
        out = self.gamma * z_normalized + self.beta
        return out

    def backward(self, dout):
        """
        Full backward pass for batch normalization.

        Computes gradients for gamma, beta, and the input.

        Args:
            dout: Gradient from next layer, shape (batch_size, num_features)

        Returns:
            dx: Gradient w.r.t. input, shape (batch_size, num_features)
        """
        if self.cache is None:
            raise RuntimeError("Must call forward() before backward()")

        x, x_norm, batch_mean, batch_var = self.cache
        m = x.shape[0]  # batch size

        # Gradients for learnable parameters
        self.dgamma = np.sum(dout * x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # Gradient w.r.t. normalized input
        dx_norm = dout * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_norm
            * (x - batch_mean)
            * -0.5
            * np.power(batch_var + self.epsilon, -1.5),
            axis=0,
        )

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(batch_var + self.epsilon), axis=0)
        dmean += dvar * np.sum(-2.0 * (x - batch_mean), axis=0) / m

        # Gradient w.r.t. input
        dx = dx_norm / np.sqrt(batch_var + self.epsilon)
        dx += dvar * 2.0 * (x - batch_mean) / m
        dx += dmean / m

        return dx
