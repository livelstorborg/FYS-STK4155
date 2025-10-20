import numpy as np
from .metrics import mse, accuracy


def train(nn, X_train, y_train, X_val, y_val, optimizer,
          epochs=100, batch_size=32, task='regression', 
          verbose=True, seed=None):
    """
    Training function (lecture style compatible).
    Works with any optimizer from optimizers.py
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = X_train.shape[0]
    metric_fn = mse if task == 'regression' else accuracy
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        n_batches = max(n_samples // batch_size, 1)
        
        # Mini-batches
        for batch in range(n_batches):
            start = batch * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Compute gradients (lecture style)
            layer_grads = nn.compute_gradient(X_batch, y_batch)
            
            # Update with optimizer
            optimizer.update(nn, layer_grads)
            
            # Track loss
            batch_pred = nn.predict(X_batch)
            batch_loss = nn.loss_fn.forward(y_batch, batch_pred)
            epoch_loss += batch_loss
        
        # Metrics
        avg_train_loss = epoch_loss / n_batches
        train_pred = nn.predict(X_train)
        train_metric = metric_fn(y_train, train_pred)
        
        val_pred = nn.predict(X_val)
        val_loss = nn.loss_fn.forward(y_val, val_pred)
        val_metric = metric_fn(y_val, val_pred)
        
        # Save
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_metric'].append(train_metric)
        history['val_metric'].append(val_metric)
        
        # Print
        if verbose and epoch % 10 == 0:
            metric_name = 'MSE' if task == 'regression' else 'Acc'
            print(f"Epoch {epoch:3d}/{epochs} - "
                  f"Loss: {avg_train_loss:.4f}, Val: {val_metric:.4f}")
    
    return history