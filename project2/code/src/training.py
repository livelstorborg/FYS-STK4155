# src/train.py
import numpy as np
from .metrics import mse, accuracy

def detect_overfitting_trend(train_losses, val_losses, window=10):
    """
    Check if validation loss is consistently increasing while training decreases.
    
    Args:
        train_losses: Array of training losses (may contain NaN)
        val_losses: Array of validation losses (may contain NaN)
        window: Number of recent epochs to check for trend
        
    Returns:
        (is_overfitting: bool, epochs_past_min: int)
    """
    # Remove NaN values
    valid_val_idx = ~np.isnan(val_losses)
    val_clean = val_losses[valid_val_idx]
    
    valid_train_idx = ~np.isnan(train_losses)
    train_clean = train_losses[valid_train_idx]
    
    if len(val_clean) < window + 5 or len(train_clean) < window + 5:
        return False, 0
    
    # Get recent window
    recent_val = val_clean[-window:]
    recent_train = train_clean[-window:]
    
    # Fit linear trends
    x = np.arange(window)
    val_slope = np.polyfit(x, recent_val, 1)[0]
    train_slope = np.polyfit(x, recent_train, 1)[0]
    
    # Overfitting: validation increasing AND training decreasing/stable
    is_overfitting = val_slope > 0 and train_slope < 0.01 * np.mean(recent_train)
    
    # Calculate epochs since minimum
    min_idx = np.argmin(val_clean)
    epochs_past_min = len(val_clean) - 1 - min_idx
    
    return is_overfitting and epochs_past_min > window, epochs_past_min


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
    if np.isnan(train_loss[epoch]) or np.isinf(train_loss[epoch]):
        return False, True, f"Diverged (NaN/Inf) at epoch {epoch}"
    
    # 2. Catastrophic explosion (10× initial loss)
    if train_loss[epoch] > 10 * train_loss[0]:
        return False, True, f"Diverged (exploded) at epoch {epoch}"
    
    # 3. Sustained upward trend after warmup
    if epoch >= warmup and epoch >= 4:
        last_5 = train_loss[epoch-4:epoch+1]
        all_increasing = all(last_5[i] > last_5[i-1] for i in range(1, 5))
        if all_increasing and train_loss[epoch] > 1.5 * train_loss[0]:
            return False, True, f"Diverged (sustained increase) at epoch {epoch}"
    
    # 4. Check for convergence (sustained low change over N epochs)
    N_CONVERGENCE_EPOCHS = 10 
    if epoch >= N_CONVERGENCE_EPOCHS:
        # Calculate the average absolute change in loss per epoch over the last N 
        total_abs_change = np.abs(train_loss[epoch] - train_loss[epoch - N_CONVERGENCE_EPOCHS])
        avg_abs_change = total_abs_change / N_CONVERGENCE_EPOCHS

        # Calculate the approximate relative change per epoch
        start_loss = train_loss[epoch - N_CONVERGENCE_EPOCHS]
        avg_rel_change = total_abs_change / (start_loss * N_CONVERGENCE_EPOCHS) if start_loss > 1e-10 else avg_abs_change 
        
        if avg_rel_change < tol_relative or avg_abs_change < tol_absolute:
            return True, False, f"Converged at epoch {epoch} (avg over {N_CONVERGENCE_EPOCHS} epochs)"
    
    return False, False, None


def train(nn, X_train, y_train, X_val, y_val, optimizer,
          epochs=100, batch_size=32, stochastic=True, task='regression', 
          tol_relative=1e-6, tol_absolute=1e-10, early_stopping=True,
          patience=50, min_delta=1e-5, check_overfitting_trend=True,  # ← CHANGED
          verbose=True, seed=None, check_gradient_frequency=10):
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
        early_stopping: If True, stop when converged or validation doesn't improve
        patience: Number of epochs to wait for validation improvement
        min_delta: Minimum change in val_loss to count as improvement
        check_overfitting_trend: If True, stop when overfitting trend detected
        verbose: Print progress every 10 epochs
        seed: Random seed for reproducibility
        check_gradient_frequency: Check gradients every N batches (default: 10)
    
    Returns:
        history: Dictionary with training metrics and convergence info
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = X_train.shape[0]
    metric_fn = mse if task == 'regression' else accuracy
    
    # Pre-allocate arrays for performance (5-10% speedup)
    history = {
        'train_loss': np.full(epochs, np.nan),
        'val_loss': np.full(epochs, np.nan),
        'train_metric': np.full(epochs, np.nan),
        'val_metric': np.full(epochs, np.nan),
        'gradient_norms': [],  # Keep as list - variable length
        'converged': False,
        'diverged': False,
        'failed': False,
        'failure_reason': None,
        'final_epoch': epochs - 1,
        'convergence_message': None
    }
    
    # Early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    best_biases = None
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
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
                
                # DEFENSE 1: Check gradient health (periodic)
                if nn.check_gradients and batch_count % check_gradient_frequency == 0:
                    grad_status, grad_mag = nn.check_gradient_health(layer_grads)
                    
                    if grad_status in ['nan', 'inf']:
                        # Critical failure - cannot continue
                        nn.training_failed = True
                        nn.failure_reason = f'{grad_status}_gradients'
                        history['failed'] = True
                        history['failure_reason'] = nn.failure_reason
                        history['final_epoch'] = epoch
                        
                        if verbose:
                            print(f"⚠️  Training failed at epoch {epoch}, batch {batch}: {nn.failure_reason}")
                        
                        # Arrays already pre-filled with NaN
                        return history
                    
                    elif grad_status == 'vanishing' and verbose and epoch % 10 == 0:
                        print(f"⚠️  Vanishing gradients detected at epoch {epoch}")
                    
                    elif grad_status == 'exploding' and verbose:
                        print(f"⚠️  Large gradients detected: norm={grad_mag:.2e}, clipping...")
                
                # DEFENSE 2: Gradient clipping (always applied)
                layer_grads, grad_norm = nn.clip_gradients(layer_grads)
                
                if grad_norm is not None and batch_count == 0:
                    history['gradient_norms'].append(grad_norm)
                
                # Update parameters
                optimizer.update(nn, layer_grads)
                
                # Track batch loss (including regularization)
                batch_pred = nn.predict(X_batch)
                batch_loss = nn.loss.forward(y_batch, batch_pred)
                
                # Check for NaN in loss
                if np.isnan(batch_loss) or np.isinf(batch_loss):
                    nn.training_failed = True
                    nn.failure_reason = 'nan_loss'
                    history['failed'] = True
                    history['failure_reason'] = 'nan_loss'
                    history['final_epoch'] = epoch
                    
                    if verbose:
                        print(f"⚠️  NaN/Inf loss detected at epoch {epoch}, batch {batch}")
                    
                    # Arrays already pre-filled with NaN
                    return history
                
                epoch_loss += batch_loss
                batch_count += 1
            
            avg_train_loss = epoch_loss / n_batches
        
        else:
            # Full-batch GD
            layer_grads = nn.compute_gradient(X_train, y_train)
            
            # Check gradient health
            if nn.check_gradients:
                grad_status, grad_mag = nn.check_gradient_health(layer_grads)
                
                if grad_status in ['nan', 'inf']:
                    nn.training_failed = True
                    nn.failure_reason = f'{grad_status}_gradients'
                    history['failed'] = True
                    history['failure_reason'] = nn.failure_reason
                    history['final_epoch'] = epoch
                    
                    if verbose:
                        print(f"⚠️  Training failed at epoch {epoch}: {nn.failure_reason}")
                    
                    # Arrays already pre-filled with NaN
                    return history
            
            # Gradient clipping
            layer_grads, grad_norm = nn.clip_gradients(layer_grads)
            if grad_norm is not None:
                history['gradient_norms'].append(grad_norm)
            
            optimizer.update(nn, layer_grads)
            
            train_pred = nn.predict(X_train)
            avg_train_loss = nn.loss.forward(y_train, train_pred)
        
        # Add regularization to training loss
        avg_train_loss += nn.compute_regularization_loss()
        
        # Check for NaN in epoch loss
        if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
            nn.training_failed = True
            nn.failure_reason = 'nan_loss'
            history['failed'] = True
            history['failure_reason'] = 'nan_loss'
            history['final_epoch'] = epoch
            
            if verbose:
                print(f"⚠️  NaN/Inf loss detected at epoch {epoch}")
            
            # Arrays already pre-filled with NaN
            return history
        
        # Compute metrics on full sets
        train_pred = nn.predict(X_train)
        train_metric = metric_fn(y_train, train_pred)
        
        val_pred = nn.predict(X_val)
        val_loss = nn.loss.forward(y_val, val_pred) + nn.compute_regularization_loss()
        val_metric = metric_fn(y_val, val_pred)
        
        # Save history (using indexing instead of append for performance)
        history['train_loss'][epoch] = avg_train_loss
        history['val_loss'][epoch] = val_loss
        history['train_metric'][epoch] = train_metric
        history['val_metric'][epoch] = val_metric
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            metric_name = 'MSE' if task == 'regression' else 'Acc'
            mode = 'SGD' if stochastic else 'GD'
            grad_info = f" | Grad: {history['gradient_norms'][-1]:.2e}" if history['gradient_norms'] else ""
            print(f"[{mode}] Epoch {epoch:3d}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val {metric_name}: {val_metric:.4f}{grad_info}")
        
        # Early stopping on validation loss
        # Early stopping on validation loss
        if early_stopping:
            # Check if improvement exceeds min_delta threshold
            improvement = best_val_loss - val_loss
            
            if improvement > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = [W.copy() for W, b in nn.layers]
                best_biases = [b.copy() for W, b in nn.layers]
            else:
                patience_counter += 1
            
            # NEW: Check for overfitting trend
            stop_due_to_trend = False
            if check_overfitting_trend and epoch >= 20:
                is_overfitting, epochs_past_min = detect_overfitting_trend(
                    history['train_loss'][:epoch+1],
                    history['val_loss'][:epoch+1],
                    window=10
                )
                
                # Stop if overfitting for significant period
                if is_overfitting and epochs_past_min >= min(patience // 2, 25):
                    stop_due_to_trend = True
                    history['convergence_message'] = (
                        f"Early stopping: Overfitting trend detected at epoch {epoch} "
                        f"({epochs_past_min} epochs past minimum)"
                    )
            
            # Stop if patience exceeded OR overfitting trend detected
            if patience_counter >= patience or stop_due_to_trend:
                history['final_epoch'] = epoch
                history['early_stop_epoch'] = epoch  # Track where we stopped
                
                if not stop_due_to_trend:  # Set message if not already set
                    history['convergence_message'] = (
                        f"Early stopping at epoch {epoch} (patience={patience})"
                    )
                
                if verbose:
                    print(history['convergence_message'])
                
                # Restore best weights
                if best_weights is not None:
                    nn.set_weights_biases(best_weights, best_biases)
                
                # Trim arrays to actual epochs trained
                history['train_loss'] = history['train_loss'][:epoch+1]
                history['val_loss'] = history['val_loss'][:epoch+1]
                history['train_metric'] = history['train_metric'][:epoch+1]
                history['val_metric'] = history['val_metric'][:epoch+1]
                break
    
    # Trim arrays to actual epochs (in case training completed without early stopping)
    actual_epochs = history['final_epoch'] + 1
    history['train_loss'] = history['train_loss'][:actual_epochs]
    history['val_loss'] = history['val_loss'][:actual_epochs]
    history['train_metric'] = history['train_metric'][:actual_epochs]
    history['val_metric'] = history['val_metric'][:actual_epochs]
    
    return history