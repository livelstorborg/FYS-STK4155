import numpy as np


def mse(y_true, y_pred):
    """
    Mean Squared Error - required for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: MSE value
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    R² score - useful for comparing with Project 1 results.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: R² value
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def accuracy(y_true, y_pred):
    """
    Accuracy score - required for classification.
    Formula from project: Accuracy = Σ I(t_i = y_i) / n
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities (will use argmax if 2D)
    
    Returns:
        float: Accuracy between 0 and 1
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    
    # If predictions are probabilities (2D), convert to class labels
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = np.asarray(y_pred).ravel().astype(int)
    
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Confusion matrix - optional but mentioned in project.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        n_classes: Number of classes (auto-detected if None)
    
    Returns:
        np.ndarray: Confusion matrix (n_classes, n_classes)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = np.asarray(y_pred).ravel().astype(int)
    
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm