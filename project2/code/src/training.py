import numpy as np
from .metrics import mse, accuracy


def check_convergence(epoch, history, tol_relative=1e-6, tol_absolute=1e-10, 
                     warmup=10, stochastic=True):
    """
    Check for convergence or divergence.
    
    Args:
        epoch: Current epoch
        history: Training history dict
        tol_relative: Relative change threshold for convergence
        tol_absolute: Absolute change threshold for convergence
        warmup: Number of epochs before checking convergence
        stochastic: If True, use stochastic convergence criteria
    
    Returns:
        (converged, diverged, message)
    """
    train_loss = history['train_loss']
    
    # 1. Check for NaN/Inf
    if np.isnan(train_loss[-1]) or np.isinf(train_loss[-1]):
        return False, True, f"Diverged (NaN/Inf) at epoch {epoch}"
    
    # 2. Catastrophic explosion (10× initial loss)
    if train_loss[-1] > 10 * train_loss[0]:
        return False, True, f"Diverged (exploded) at epoch {epoch}"
    
    # 3. Sustained upward trend after warmup
    if epoch >= warmup and len(train_loss) >= 5:
        last_5 = train_loss[-5:]
        all_increasing = all(last_5[i] > last_5[i-1] for i in range(1, 5))
        if all_increasing and train_loss[-1] > 1.5 * train_loss[0]:
            return False, True, f"Diverged (sustained increase) at epoch {epoch}"
    
    # 4. Check for convergence (after warmup)
    if epoch > 0:
        abs_change = np.abs(train_loss[-1] - train_loss[-2])
        rel_change = abs_change / train_loss[-2] if train_loss[-2] > 1e-10 else abs_change
        
        if rel_change < tol_relative or abs_change < tol_absolute:
            return True, False, f"Converged at epoch {epoch}"
    
    return False, False, None





def train(nn, X_train, y_train, X_val, y_val, optimizer,
          epochs=100, batch_size=32, stochastic=True, task='regression', 
          tol_relative=1e-6, tol_absolute=1e-10, early_stopping=True,
          verbose=True, seed=None):
    """
    Training function supporting both full-batch GD and mini-batch SGD.
    Works with any optimizer from optimizers.py (GD, RMSprop, Adam).
    
    Args:
        nn: NeuralNetwork instance (with optional regularization)
        X_train, y_train: Training data
        X_val, y_val: Validation data
        optimizer: Optimizer instance (GD, RMSprop, or Adam)
        epochs: Maximum number of training epochs
        batch_size: Mini-batch size (only used if stochastic=True)
        stochastic: If True, use mini-batch SGD; if False, use full-batch GD
        task: 'regression' or 'classification'
        tol_relative: Relative change threshold for convergence
        tol_absolute: Absolute change threshold for convergence
        early_stopping: If True, stop when converged or diverged
        verbose: Print progress every 10 epochs
        seed: Random seed for reproducibility
    
    Returns:
        history: Dictionary with training metrics and convergence info
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = X_train.shape[0]
    metric_fn = mse if task == 'regression' else accuracy
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': [],
        'converged': False,
        'diverged': False,
        'final_epoch': epochs - 1,
        'convergence_message': None
    }
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        if stochastic:
            # Mini-batch SGD
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            n_batches = max(n_samples // batch_size, 1)
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Compute gradients
                layer_grads = nn.compute_gradient(X_batch, y_batch)
                
                # Update parameters
                optimizer.update(nn, layer_grads)
                
                # Track batch loss (including regularization)
                batch_pred = nn.predict(X_batch)
                batch_loss = nn.loss_fn.forward(y_batch, batch_pred)
                epoch_loss += batch_loss
            
            avg_train_loss = epoch_loss / n_batches
        
        else:
            # Full-batch GD
            layer_grads = nn.compute_gradient(X_train, y_train)
            optimizer.update(nn, layer_grads)
            
            train_pred = nn.predict(X_train)
            avg_train_loss = nn.loss_fn.forward(y_train, train_pred)
        
        # Add regularization to training loss
        avg_train_loss += nn.compute_regularization_loss()
        
        # Compute metrics on full sets
        train_pred = nn.predict(X_train)
        train_metric = metric_fn(y_train, train_pred)
        
        val_pred = nn.predict(X_val)
        val_loss = nn.loss_fn.forward(y_val, val_pred) + nn.compute_regularization_loss()
        val_metric = metric_fn(y_val, val_pred)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_metric'].append(train_metric)
        history['val_metric'].append(val_metric)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            metric_name = 'MSE' if task == 'regression' else 'Acc'
            mode = 'SGD' if stochastic else 'GD'
            print(f"[{mode}] Epoch {epoch:3d}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val {metric_name}: {val_metric:.4f}")
        
        # Check convergence (after first epoch)
        if early_stopping and epoch > 0:
            converged, diverged, message = check_convergence(
                epoch, history, tol_relative, tol_absolute, 
                warmup=10, stochastic=stochastic
            )
            
            if converged or diverged:
                history['converged'] = converged
                history['diverged'] = diverged
                history['final_epoch'] = epoch
                history['convergence_message'] = message
                if verbose:
                    print(message)
                break
    
    return history