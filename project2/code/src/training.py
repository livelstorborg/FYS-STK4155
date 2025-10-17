

########################################################
#                FROM EXERCISES WEEK 42                #
########################################################



# ---------- Training functions outside the class ----------
def train_gd(nn, inputs, targets, learning_rate=0.01, epochs=100):
    """Basic gradient descent (full batch, no momentum)"""
    accuracies = []
    
    for epoch in range(epochs):
        # Compute gradient on entire dataset
        grads = nn.compute_gradient(inputs, targets)
        
        # Update weights using basic gradient descent
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(nn.layers, grads)):
            W -= learning_rate * W_g
            b -= learning_rate * b_g
            nn.layers[i] = (W, b)
        
        # Track accuracy
        predictions = nn.predict(inputs)
        acc = accuracy(predictions, targets)
        accuracies.append(acc)
    
    return accuracies


def train_momentum(nn, inputs, targets, batch_size=32, learning_rate=0.01, 
                   momentum=0.9, epochs=100):
    """Train neural network with mini-batch SGD + momentum"""
    n_samples = inputs.shape[0]
    velocities = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
    accuracies = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_inputs = inputs_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]
            
            # Use the class method!
            grads = nn.compute_gradient(batch_inputs, batch_targets)
            
            # Momentum update
            for j, ((W, b), (W_g, b_g), (v_W, v_b)) in enumerate(zip(nn.layers, grads, velocities)):
                v_W = momentum * v_W - learning_rate * W_g
                v_b = momentum * v_b - learning_rate * b_g
                W += v_W
                b += v_b
                velocities[j] = (v_W, v_b)
                nn.layers[j] = (W, b)
        
        # Track accuracy
        predictions = nn.predict(inputs)
        acc = accuracy(predictions, targets)
        accuracies.append(acc)
    
    return accuracies


def train_adam():
    pass