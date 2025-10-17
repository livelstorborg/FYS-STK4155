import numpy as np


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, 
                 activations, loss_fn, seed=None):
        """
        Now using activation and loss OBJECTS instead of functions.
        
        Parameters:
        -----------
        activations : list of Activation objects
            e.g., [Sigmoid(), ReLU(), Softmax()]
        loss_fn : Loss object
            e.g., MSE() or CrossEntropy()
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activations = activations  # Now objects!
        self.loss_fn = loss_fn  # Now an object!
        
        # Initialize weights
        self.weights = []
        self.biases = []
        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size) * 0.01
            b = np.zeros(layer_output_size)
            self.weights.append(W)
            self.biases.append(b)
            i_size = layer_output_size
        # Number of layers (useful for optimizers)
        self.n_layers = len(self.weights)
    
    def predict(self, inputs):
        """Feed forward to make predictions."""
        a = inputs
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            z = a @ W.T + b
            a = activation.forward(z)  # Use object method
        return a
    
    def forward(self, inputs):
        """
        Feed forward that saves values for backprop.
        Returns: (a_values, z_values)
        """
        a_values = [inputs]
        z_values = []
        a = inputs
        
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            z = a @ W.T + b
            a = activation.forward(z)
            z_values.append(z)
            a_values.append(a)
        
        return a_values, z_values
    
    def backward(self, inputs, targets):
        """
        Compute gradients using backpropagation.
        Returns: (weight_grads, bias_grads)
        """
        a_values, z_values = self.forward(inputs)
        
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        
        # Start with output layer error
        delta = self.loss_fn.backward(targets, a_values[-1])  # Use object method
        
        # Backpropagate
        for i in reversed(range(n_layers)):
            # Compute gradients
            weight_grads[i] = (delta.T @ a_values[i]) / inputs.shape[0]
            bias_grads[i] = np.sum(delta, axis=0) / inputs.shape[0]
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i]) * \
                        self.activations[i-1].backward(z_values[i-1])
        
        return weight_grads, bias_grads