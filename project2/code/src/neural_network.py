import numpy as np


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, 
                 activations, loss_fn, seed=None, lambda_reg=0.0, reg_type=None):
        """
        Initialize neural network (lecture style).
        
        Parameters:
        -----------
        network_input_size : int
            Number of input features
        layer_output_sizes : list of int
            Number of neurons in each layer (including output)
        activations : list of Activation objects
            e.g., [Sigmoid(), Linear()]
        loss_fn : Loss object
            e.g., MSE() or CrossEntropy()
        seed : int
            Random seed
        lambda_reg : float
            Regularization strength (default: 0.0, no regularization)
        reg_type : str or None
            Regularization type: 'l1', 'l2', or None (default: None)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activations = activations
        self.loss_fn = loss_fn
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        
        # Initialize layers as list of (W, b) tuples (lecture style)
        self.layers = self._create_layers()
        self.n_layers = len(self.layers)
    
    def _create_layers(self):
        """Create layers with random initialization."""
        layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size) * 0.01
            b = np.zeros(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers
    
    def predict(self, inputs):
        """Simple feed forward pass."""
        a = inputs
        for (W, b), activation in zip(self.layers, self.activations):
            z = a @ W.T + b
            a = activation.forward(z)
        return a
    
    def _feed_forward_saver(self, inputs):
        """Feed forward that saves values for backprop."""
        layer_inputs = []
        zs = []
        a = inputs
        
        for (W, b), activation in zip(self.layers, self.activations):
            layer_inputs.append(a)
            z = a @ W.T + b
            a = activation.forward(z)
            zs.append(z)
        
        return layer_inputs, zs, a
    
    def compute_gradient(self, inputs, targets):
        """Compute gradients using backpropagation with optional regularization."""
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]
        
        # Start with output layer error (from loss function)
        delta = self.loss_fn.backward(targets, predict)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer_input = layer_inputs[i]
            
            # Compute gradients for this layer
            W, b = self.layers[i]
            dW = delta.T @ layer_input
            db = np.sum(delta, axis=0)
            
            # Add regularization term to weight gradients
            if self.lambda_reg > 0 and self.reg_type is not None:
                if self.reg_type.lower() == 'l2':
                    dW += self.lambda_reg * W
                elif self.reg_type.lower() == 'l1':
                    dW += self.lambda_reg * np.sign(W)
            
            layer_grads[i] = (dW, db)
            
            # Propagate error to previous layer (if not at input)
            if i > 0:
                # Propagate through weights
                delta = delta @ W
                # Apply activation derivative of previous layer
                delta = delta * self.activations[i-1].backward(zs[i-1])
        
        return layer_grads
    
    def compute_regularization_loss(self):
        """Compute the regularization penalty term."""
        if self.lambda_reg == 0 or self.reg_type is None:
            return 0.0
        
        reg_loss = 0.0
        for W, b in self.layers:
            if self.reg_type.lower() == 'l2':
                reg_loss += np.sum(W ** 2)
            elif self.reg_type.lower() == 'l1':
                reg_loss += np.sum(np.abs(W))
        
        return self.lambda_reg * reg_loss


    # Add properties for optimizer compatibility
    @property
    def weights(self):
        """Get weights (for optimizer compatibility)."""
        return [W for W, b in self.layers]
    
    @property
    def biases(self):
        """Get biases (for optimizer compatibility)."""
        return [b for W, b in self.layers]
    
    def set_weights_biases(self, weights, biases):
        """Set weights and biases (for optimizer compatibility)."""
        self.layers = [(W, b) for W, b in zip(weights, biases)]