# src/neural_network.py
import numpy as np


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, 
                 activations, loss, seed=None, lambda_reg=0.0, 
                 reg_type=None, weight_init_scale=None,
                 gradient_clip_norm=5.0, check_gradients=True,
                 use_batch_norm=False):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        network_input_size : int
            Number of input features
        layer_output_sizes : list of int
            Number of neurons in each layer (including output)
        activations : list of Activation objects
            e.g., [ReLU(), ReLU(), Softmax()]
        loss : Loss object
            e.g., MSE() or CrossEntropy()
        seed : int
            Random seed for reproducibility
        lambda_reg : float
            Regularization strength (default: 0.0, no regularization)
        reg_type : str or None
            Regularization type: 'l1', 'l2', or None
        weight_init_scale : float
            Weight initialization scale factor
        gradient_clip_norm : float
            Maximum gradient norm for clipping (default: 5.0)
        check_gradients : bool
            Whether to check for gradient issues during training
        use_batch_norm : bool
            Whether to use batch normalization after each hidden layer
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activations = activations
        self.loss = loss
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.weight_init_scale = weight_init_scale
        self.gradient_clip_norm = gradient_clip_norm
        self.check_gradients = check_gradients
        self.use_batch_norm = use_batch_norm
        
        # Training state tracking
        self.training_failed = False
        self.failure_reason = None
        self.gradient_history = []
        
        # Initialize layers as list of (W, b) tuples
        self.layers = self._create_layers()
        self.n_layers = len(self.layers)
        
        # Initialize batch norm layers if requested
        self.batch_norm_layers = []
        if self.use_batch_norm:
            from .activations import BatchNorm
            # Add batch norm for all hidden layers (not output layer)
            for idx in range(len(self.layer_output_sizes) - 1):
                bn = BatchNorm(self.layer_output_sizes[idx])
                self.batch_norm_layers.append(bn)

    def _create_layers(self):
        """Create layers and initialize weights and biases."""
        layers = []
        i_size = self.network_input_size
        
        for idx, layer_output_size in enumerate(self.layer_output_sizes):
            # Use activation-aware initialization
            activation_name = self.activations[idx].__class__.__name__
            
            if activation_name in ['ReLU', 'LeakyReLU']:
                std = np.sqrt(2.0 / i_size) # He initialization
            elif activation_name in ['Sigmoid']:
                std = np.sqrt(2.0 / (i_size + layer_output_size)) # Xavier initialization
            else:
                std = np.sqrt(1.0 / i_size) # Default for Linear
            
            if self.weight_init_scale is not None:
                std = self.weight_init_scale
            
            W = np.random.randn(layer_output_size, i_size) * std
            b = np.random.randn(layer_output_size) * 0.01   
 
            layers.append((W, b))
            i_size = layer_output_size

        return layers
    
    def check_gradient_health(self, layer_grads, 
                             threshold_vanish=1e-7, 
                             threshold_explode=1e3):
        """
        Check if gradients are healthy.
        
        Parameters:
        -----------
        layer_grads : list of (dW, db) tuples
            Gradients for each layer
        threshold_vanish : float
            Threshold below which gradients are considered vanishing
        threshold_explode : float
            Threshold above which gradients are considered exploding
        
        Returns:
        --------
        status : str
            'healthy', 'vanishing', 'exploding', 'nan', or 'inf'
        magnitude : float or np.nan
            Average gradient magnitude (NaN if unhealthy)
        """
        if not layer_grads:
            return 'no_gradients', 0.0
        
        avg_grads = []
        max_grads = []
        
        # Check each layer
        for dW, db in layer_grads:
            # Critical failures: NaN or Inf
            if np.any(np.isnan(dW)) or np.any(np.isnan(db)):
                return 'nan', np.nan
            if np.any(np.isinf(dW)) or np.any(np.isinf(db)):
                return 'inf', np.nan
            
            avg_grads.append(np.abs(dW).mean())
            max_grads.append(np.abs(dW).max())
        
        avg_grad = np.mean(avg_grads)
        max_grad = np.max(max_grads)
        
        # Check for vanishing or exploding
        if avg_grad < threshold_vanish:
            return 'vanishing', np.nan
        elif max_grad > threshold_explode:
            return 'exploding', np.nan
        else:
            return 'healthy', avg_grad
    
    def numerical_gradient_check(self, inputs, targets, epsilon=1e-7, tolerance=1e-5):
        """
        Verify backpropagation implementation using numerical gradients.
        
        Computes gradients numerically using finite differences and compares
        them to the analytical gradients from backpropagation.
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input data (small batch recommended, e.g., 10 samples)
        targets : np.ndarray
            Target values
        epsilon : float
            Small perturbation for finite differences
        tolerance : float
            Maximum relative error to consider gradients correct
        
        Returns:
        --------
        is_correct : bool
            True if analytical gradients match numerical gradients
        max_error : float
            Maximum relative error across all parameters
        details : dict
            Detailed comparison for each layer
        """
        # Get analytical gradients
        analytical_grads = self.compute_gradient(inputs, targets)
        
        details = {}
        max_relative_error = 0.0
        
        for layer_idx, ((W, b), (dW_analytical, db_analytical)) in enumerate(zip(self.layers, analytical_grads)):
            # Check weight gradients
            dW_numerical = np.zeros_like(W)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    # Perturb weight positively
                    W[i, j] += epsilon
                    pred_plus = self.predict(inputs)
                    loss_plus = self.loss.forward(targets, pred_plus) + self.compute_regularization_loss()
                    
                    # Perturb weight negatively
                    W[i, j] -= 2 * epsilon
                    pred_minus = self.predict(inputs)
                    loss_minus = self.loss.forward(targets, pred_minus) + self.compute_regularization_loss()
                    
                    # Restore original weight
                    W[i, j] += epsilon
                    
                    # Compute numerical gradient
                    dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Check bias gradients
            db_numerical = np.zeros_like(b)
            for i in range(b.shape[0]):
                # Perturb bias positively
                b[i] += epsilon
                pred_plus = self.predict(inputs)
                loss_plus = self.loss.forward(targets, pred_plus) + self.compute_regularization_loss()
                
                # Perturb bias negatively
                b[i] -= 2 * epsilon
                pred_minus = self.predict(inputs)
                loss_minus = self.loss.forward(targets, pred_minus) + self.compute_regularization_loss()
                
                # Restore original bias
                b[i] += epsilon
                
                # Compute numerical gradient
                db_numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Compute relative errors
            weight_error = np.abs(dW_analytical - dW_numerical) / (np.abs(dW_analytical) + np.abs(dW_numerical) + 1e-8)
            bias_error = np.abs(db_analytical - db_numerical) / (np.abs(db_analytical) + np.abs(db_numerical) + 1e-8)
            
            max_weight_error = np.max(weight_error)
            max_bias_error = np.max(bias_error)
            
            details[f'layer_{layer_idx}'] = {
                'max_weight_error': max_weight_error,
                'max_bias_error': max_bias_error,
                'weight_gradient_norm': np.linalg.norm(dW_analytical),
                'bias_gradient_norm': np.linalg.norm(db_analytical)
            }
            
            max_relative_error = max(max_relative_error, max_weight_error, max_bias_error)
        
        is_correct = max_relative_error < tolerance
        
        return is_correct, max_relative_error, details
    
    def clip_gradients(self, layer_grads):
        """
        Clip gradients by global norm.
        
        Parameters:
        -----------
        layer_grads : list of (dW, db) tuples
            Gradients for each layer
        
        Returns:
        --------
        clipped_grads : list of (dW, db) tuples
            Clipped gradients
        total_norm : float
            Total gradient norm before clipping
        """
        if self.gradient_clip_norm is None or self.gradient_clip_norm <= 0:
            # No clipping
            return layer_grads, None
        
        # Compute global norm across all layers
        total_norm = 0.0
        for dW, db in layer_grads:
            total_norm += np.sum(dW**2) + np.sum(db**2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if needed
        if total_norm > self.gradient_clip_norm:
            clip_coef = self.gradient_clip_norm / (total_norm + 1e-8)
            clipped_grads = [
                (dW * clip_coef, db * clip_coef)
                for dW, db in layer_grads
            ]
            return clipped_grads, total_norm
        else:
            return layer_grads, total_norm

    def summary(self):
        """
        Print model architecture summary (TensorFlow style).
        """
        print("=" * 65)
        print("Model Summary")
        print("=" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 65)
        
        total_params = 0
        trainable_params = 0
        
        # Input layer
        print(f"{'Input':<25} {f'({self.network_input_size},)':<20} {'0':<10}")
        print("-" * 65)
        
        current_input = self.network_input_size
        for i, ((W, b), activation) in enumerate(zip(self.layers, self.activations)):
            layer_name = f"Dense_{i+1} ({activation.__class__.__name__})"
            output_shape = f"({W.shape[0]},)"
            
            # Calculate parameters: weights + biases
            n_params = W.size + b.size
            
            # Add batch norm parameters if applicable
            if self.use_batch_norm and i < len(self.batch_norm_layers):
                bn = self.batch_norm_layers[i]
                n_params += bn.gamma.size + bn.beta.size
            
            total_params += n_params
            trainable_params += n_params
            
            print(f"{layer_name:<25} {output_shape:<20} {n_params:<10}")
            print("-" * 65)
            
            current_input = W.shape[0]
        
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: 0")
        
        # Additional info
        print("=" * 65)
        print(f"Loss function: {self.loss.__class__.__name__}")
        if self.lambda_reg > 0:
            print(f"Regularization: {self.reg_type.upper()} (λ={self.lambda_reg})")
        else:
            print("Regularization: None")
        if self.use_batch_norm:
            print(f"Batch Normalization: Enabled ({len(self.batch_norm_layers)} layers)")
        print(f"Gradient clipping: {'Enabled' if self.gradient_clip_norm else 'Disabled'}")
        if self.gradient_clip_norm:
            print(f"  Max norm: {self.gradient_clip_norm}")
        print("=" * 65)
    
    def predict(self, inputs):
        """Simple feed forward pass."""
        restore_training = False
        if self.use_batch_norm:
            # During inference, use running stats
            restore_training = any(getattr(bn, "training", True) for bn in self.batch_norm_layers)
            if restore_training:
                self.set_training_mode(False)
        try:
            a = inputs
            for idx, ((W, b), activation) in enumerate(zip(self.layers, self.activations)):
                z = a @ W.T + b

                # Apply batch normalization if enabled (only for hidden layers)
                if self.use_batch_norm and idx < len(self.batch_norm_layers):
                    z = self.batch_norm_layers[idx].forward(z)

                a = activation.forward(z)
            return a
        finally:
            if self.use_batch_norm and restore_training:
                self.set_training_mode(True)
    
    def _feed_forward_saver(self, inputs):
        """Feed forward that saves values for backprop."""
        layer_inputs = []
        zs = []
        a = inputs
        
        for idx, ((W, b), activation) in enumerate(zip(self.layers, self.activations)):
            layer_inputs.append(a)
            z = a @ W.T + b
            
            # Apply batch normalization if enabled (only for hidden layers)
            if self.use_batch_norm and idx < len(self.batch_norm_layers):
                z = self.batch_norm_layers[idx].forward(z)
            
            a = activation.forward(z)
            zs.append(z)
        
        return layer_inputs, zs, a
    
    def compute_gradient(self, inputs, targets):
        """Compute gradients using backpropagation with optional regularization."""
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]
        bn_grads = [None] * (len(self.batch_norm_layers)) if self.use_batch_norm else None
        
        # Start with output layer error (from loss function)
        delta = self.loss.backward(targets, predict)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer_input = layer_inputs[i]
            
            # Compute gradients for this layer
            W, b = self.layers[i]
            dW = delta.T @ layer_input
            db = np.sum(delta, axis=0)
            
            # Add regularization term to weight gradients (NOT biases)
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
                
                # Propagate through activation derivative of previous layer
                delta = delta * self.activations[i-1].backward(zs[i-1])
                
                # Propagate through batch norm if enabled
                if self.use_batch_norm and (i - 1) < len(self.batch_norm_layers):
                    bn_layer = self.batch_norm_layers[i - 1]
                    delta = bn_layer.backward(delta)
                    bn_grads[i - 1] = (bn_layer.dgamma, bn_layer.dbeta)
        
        if self.use_batch_norm:
            return layer_grads, bn_grads
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

    def set_training_mode(self, training=True):
        """
        Set training mode for batch normalization layers.
        
        Parameters:
        -----------
        training : bool
            If True, use training mode (batch statistics).
            If False, use inference mode (running statistics).
        """
        if self.use_batch_norm:
            for bn in self.batch_norm_layers:
                bn.training = training

    # Properties for optimizer compatibility
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
