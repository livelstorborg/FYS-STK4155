import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler






# ===================================================================
#                       UTILITY FUNCTIONS
# ===================================================================

def runge(x):
    """Runge function: f(x) = 1 / (1 + 25x^2)"""
    return 1.0 / (1 + 25 * x**2)


def OLS_parameters(X, y):
    """OLS solution (No explicit intercept needed due to scaling)"""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Ridge_parameters(X, y, lam=0.0):
    """Calculates Ridge/OLS parameters. lam=0 for OLS."""
    n = X.shape[1]
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    return np.linalg.pinv(X.T @ X + lam * np.eye(n)) @ X.T @ y

def polynomial_features(x, p, intercept=False):
    """Creates polynomial feature matrix"""
    X = x.reshape(-1, 1)
    if intercept:
        X = np.hstack((np.ones((x.size, 1)), X))
    for i in range(2, p + 1):
        X = np.hstack((X, (x**i).reshape(-1, 1)))
    return X

def generate_data(n_samples=100, noise_level=0.1, seed=42):
    """
    Generate Runge function with noise.
    
    This function sets np.random.seed(seed) to ensure reproducible data generation.
    """
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n_samples, 1))
    y = 1 / (1 + 25 * X.flatten()**2) + np.random.normal(0, noise_level, n_samples)
    return X, y.reshape(-1, 1)


def scale_data(X, y, X_mean=None, X_std=None, y_mean=None, y_std=None):
    """Scaling function: Standardize X and y."""
    if X_mean is None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        y_mean = np.mean(y)
        y_std = np.std(y)
        if np.isscalar(y_std):
            y_std = max(y_std, 1e-8)
        else:
            y_std = np.where(y_std < 1e-8, 1.0, y_std)

    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean)
    
    return X_scaled, y_scaled, X_mean, X_std, y_mean, y_std


def split_scale_data(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Split data into train/val/test and scale using training statistics."""
    test_ratio = 1 - train_ratio - val_ratio

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=seed
    )

    X_train_s, y_train_s, X_mean, X_std, y_mean, y_std = scale_data(X_train, y_train)
    X_val_s, y_val_s, _, _, _, _ = scale_data(
        X_val, y_val, X_mean, X_std, y_mean, y_std
    )
    X_test_s, y_test_s, _, _, _, _ = scale_data(
        X_test, y_test, X_mean, X_std, y_mean, y_std
    )

    return (
        X_train_s,
        X_val_s,
        X_test_s,
        y_train_s,
        y_val_s,
        y_test_s,
        y_train,
        y_val,
        y_test,
        X_mean,
        X_std,
        y_mean,
        y_std,
    )


def load_mnist_data(use_subset=True, subset_size=10000, seed=42):
    """
    Load and prepare MNIST dataset
    
    NOTE: This function uses sklearn's train_test_split with random_state=seed
    for reproducible splits, but does NOT set np.random.seed() to avoid
    interfering with the calling code's random state. The subset selection
    will use the current numpy random state.

    Returns:
    --------
    tuple: (X_train, y_train, y_train_encoded,
            X_val, y_val, y_val_encoded,
            X_test, y_test, y_test_encoded, scaler)
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data
    y = mnist.target.astype(int)

    # Use subset for faster experiments
    # NOTE: Uses current numpy random state, does not reset seed
    if use_subset and subset_size:
        indices = np.random.choice(len(X), min(subset_size, len(X)), replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Using subset: {len(X)} samples")
    else:
        print(f"Using full dataset: {len(X)} samples")

    # Split: 60% train, 20% validation, 20% test
    # Use random_state for reproducible splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Scale pixel values to [0, 1] then standardize
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # One-hot encode labels
    def one_hot_encode(y, n_classes=10):
        n_samples = len(y)
        y_encoded = np.zeros((n_samples, n_classes))
        y_encoded[np.arange(n_samples), y] = 1
        return y_encoded

    y_train_encoded = one_hot_encode(y_train)
    y_val_encoded = one_hot_encode(y_val)
    y_test_encoded = one_hot_encode(y_test)

    return (
        X_train,
        y_train,
        y_train_encoded,
        X_val,
        y_val,
        y_val_encoded,
        X_test,
        y_test,
        y_test_encoded,
        scaler,
    )
