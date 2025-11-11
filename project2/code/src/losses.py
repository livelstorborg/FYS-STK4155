import numpy as np


class Loss:
    """Base class for loss functions."""
    
    def forward(self, y_true, y_pred):
        """
        Compute loss value.
        
        Args:
            y_true: True values/labels
            y_pred: Predicted values/probabilities
        
        Returns:
            float: Loss value
        """
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t. predictions.
        This is the STARTING POINT for backpropagation.
        
        Args:
            y_true: True values/labels
            y_pred: Predicted values/probabilities
        
        Returns:
            ndarray: Gradient ∂L/∂y_pred (same shape as y_pred)
        """
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squared Error loss for regression.
    
    L = (1/n) * Σ(y_true - y_pred)²
    
    Gradient:
    ∂L/∂y_pred = -(2/n) * (y_true - y_pred)
    """
    
    def forward(self, y_true, y_pred):
        """Compute MSE loss."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of MSE.
        
        Returns the initial error signal that the NN class
        will propagate backward through the layers.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_samples = y_true.shape[0]
        
        # Gradient: ∂L/∂y_pred = -(2/n) * (y_true - y_pred)
        return -(2.0 / n_samples) * (y_true - y_pred)



class CrossEntropy(Loss):
    """
    Cross-Entropy loss for classification.
    
    Assumes y_pred is the output of Softmax (probabilities).
    
    L = -(1/n) * Σ_i Σ_k y_true[i,k] * log(y_pred[i,k])
    
    Gradient (when combined with Softmax):
    ∂L/∂y_pred = (1/n) * (y_pred - y_true)
    
    This is the beautiful simplification when using Softmax + CrossEntropy!
    """
    
    def forward(self, y_true, y_pred):
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: True labels, either:
                    - Class indices: shape (n_samples,)
                    - One-hot encoded: shape (n_samples, n_classes)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
        
        Returns:
            float: Cross-entropy loss
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        # Convert class indices to one-hot if needed
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_onehot
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute cross-entropy
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of cross-entropy.
        
        When using Softmax activation in the output layer,
        the combined gradient of Softmax + CrossEntropy simplifies to:
        
        ∂L/∂y_pred = (y_pred - y_true) / n_samples
        
        This is a beautiful result that makes backprop much simpler!
        
        Args:
            y_true: True labels (one-hot or indices)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
        
        Returns:
            ndarray: Gradient, shape (n_samples, n_classes)
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        # Convert class indices to one-hot if needed
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_onehot
        
        n_samples = y_pred.shape[0]
        
        # Gradient: (y_pred - y_true) / n
        return (y_pred - y_true) / n_samples