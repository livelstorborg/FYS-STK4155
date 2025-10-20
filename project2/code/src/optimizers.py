import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def update(self, nn, layer_grads):
        """
        Update network parameters using gradients.
        
        Args:
            nn: NeuralNetwork instance
            layer_grads: List of (dW, db) tuples from compute_gradient
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset optimizer state between training runs."""
        pass


class GD(Optimizer):
    """Plain Gradient Descent (batch or mini-batch)."""
    
    def __init__(self, eta=0.01):
        self.eta = eta
    
    def update(self, nn, layer_grads):
        """Update using plain GD."""
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(nn.layers, layer_grads)):
            W -= self.eta * W_g
            b -= self.eta * b_g
            nn.layers[i] = (W, b)


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, eta=0.001, decay_rate=0.9, epsilon=1e-8):
        self.eta = eta
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, nn, layer_grads):
        """Update using RMSprop."""
        if self.cache is None:
            self.cache = [(np.zeros_like(W), np.zeros_like(b)) 
                         for W, b in nn.layers]
        
        for i, ((W, b), (W_g, b_g), (cache_W, cache_b)) in enumerate(
            zip(nn.layers, layer_grads, self.cache)):
            
            # Update cache
            cache_W = self.decay_rate * cache_W + (1 - self.decay_rate) * W_g**2
            cache_b = self.decay_rate * cache_b + (1 - self.decay_rate) * b_g**2
            
            # Update parameters
            W -= self.eta * W_g / (np.sqrt(cache_W) + self.epsilon)
            b -= self.eta * b_g / (np.sqrt(cache_b) + self.epsilon)

            self.cache[i] = (cache_W, cache_b)
            nn.layers[i] = (W, b)
    
    def reset(self):
        """Reset cache for new training run."""
        self.cache = None


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, nn, layer_grads):
        """Update using Adam."""
        if self.m is None:
            self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
            self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        
        self.t += 1
        
        for i, ((W, b), (W_g, b_g), (m_W, m_b), (v_W, v_b)) in enumerate(
            zip(nn.layers, layer_grads, self.m, self.v)):
            
            # Update moments
            m_W = self.beta1 * m_W + (1 - self.beta1) * W_g
            m_b = self.beta1 * m_b + (1 - self.beta1) * b_g
            v_W = self.beta2 * v_W + (1 - self.beta2) * W_g**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * b_g**2
            
            # Bias correction
            m_W_hat = m_W / (1 - self.beta1**self.t)
            m_b_hat = m_b / (1 - self.beta1**self.t)
            v_W_hat = v_W / (1 - self.beta2**self.t)
            v_b_hat = v_b / (1 - self.beta2**self.t)
            
            # Update parameters
            W -= self.eta * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            b -= self.eta * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            self.m[i] = (m_W, m_b)
            self.v[i] = (v_W, v_b)
            nn.layers[i] = (W, b)
    
    def reset(self):
        """Reset moments and timestep for new training run."""
        self.m = None
        self.v = None
        self.t = 0