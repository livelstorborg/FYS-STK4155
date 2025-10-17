import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def update(self, nn, weight_grads, bias_grads):
        """
        Update network parameters.
        
        Args:
            nn: NeuralNetwork instance
            weight_grads: List of weight gradients from backprop
            bias_grads: List of bias gradients from backprop
        """
        raise NotImplementedError


class GD(Optimizer):
    """
    Plain Gradient Descent.
    
    Update rule: θ = θ - η * ∇θ
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, nn, weight_grads, bias_grads):
        """Update weights using plain gradient descent."""
        for i in range(nn.n_layers):
            nn.weights[i] -= self.learning_rate * weight_grads[i]
            nn.biases[i] -= self.learning_rate * bias_grads[i]


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    
    With momentum:
        v = β * v + ∇θ
        θ = θ - η * v
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Velocity terms (initialized on first update)
        self.velocity_w = None
        self.velocity_b = None
    
    def update(self, nn, weight_grads, bias_grads):
        """Update weights using SGD with momentum."""
        # Initialize velocity on first call
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in nn.weights]
            self.velocity_b = [np.zeros_like(b) for b in nn.biases]
        
        # Update with momentum
        for i in range(nn.n_layers):
            # Update velocity
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + weight_grads[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + bias_grads[i]
            
            # Update parameters
            nn.weights[i] -= self.learning_rate * self.velocity_w[i]
            nn.biases[i] -= self.learning_rate * self.velocity_b[i]


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Adapts learning rate based on moving average of squared gradients:
        s = β * s + (1-β) * ∇θ²
        θ = θ - η * ∇θ / (√s + ε)
    """
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        
        # Cache of squared gradients
        self.cache_w = None
        self.cache_b = None
    
    def update(self, nn, weight_grads, bias_grads):
        """Update weights using RMSprop."""
        # Initialize cache on first call
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in nn.weights]
            self.cache_b = [np.zeros_like(b) for b in nn.biases]
        
        for i in range(nn.n_layers):
            # Update cache (moving average of squared gradients)
            self.cache_w[i] = self.decay_rate * self.cache_w[i] + \
                             (1 - self.decay_rate) * weight_grads[i]**2
            self.cache_b[i] = self.decay_rate * self.cache_b[i] + \
                             (1 - self.decay_rate) * bias_grads[i]**2
            
            # Update parameters with adaptive learning rate
            nn.weights[i] -= self.learning_rate * weight_grads[i] / \
                            (np.sqrt(self.cache_w[i]) + self.epsilon)
            nn.biases[i] -= self.learning_rate * bias_grads[i] / \
                           (np.sqrt(self.cache_b[i]) + self.epsilon)


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Combines momentum and RMSprop:
        m = β₁ * m + (1-β₁) * ∇θ          # First moment (momentum)
        v = β₂ * v + (1-β₂) * ∇θ²         # Second moment (RMSprop)
        m̂ = m / (1 - β₁ᵗ)                 # Bias correction
        v̂ = v / (1 - β₂ᵗ)                 # Bias correction
        θ = θ - η * m̂ / (√v̂ + ε)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # First and second moments
        self.m_w = None
        self.m_b = None
        self.v_w = None
        self.v_b = None
        
        # Time step for bias correction
        self.t = 0
    
    def update(self, nn, weight_grads, bias_grads):
        """Update weights using Adam."""
        # Initialize moments on first call
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in nn.weights]
            self.m_b = [np.zeros_like(b) for b in nn.biases]
            self.v_w = [np.zeros_like(w) for w in nn.weights]
            self.v_b = [np.zeros_like(b) for b in nn.biases]
        
        self.t += 1
        
        for i in range(nn.n_layers):
            # Update first moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * weight_grads[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * bias_grads[i]
            
            # Update second moment (RMSprop)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * weight_grads[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * bias_grads[i]**2
            
            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            nn.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            nn.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)