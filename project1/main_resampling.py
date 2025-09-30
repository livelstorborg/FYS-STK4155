import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.linear_model import Ridge, Lasso

from src.utils import polynomial_features, scale_data, runge, OLS_parameters
from src.plotting import setup_plot_formatting

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
N = 300
degrees = range(1, 16)
n_bootstrap = 100
n_cv_folds = 5
lambda_ridge = 1e-5

# Generate data
x = np.linspace(-1, 1, N)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)
y_noise = y_true + random_noise
sigma_squared = 0.01  # Known noise variance (0.1^2)

print(f"Generated {N} data points with noise std = 0.1")
print(f"True noise variance σ² = {sigma_squared}")

# Split data into train/test for consistent evaluation
X_temp = polynomial_features(x, 1)  # Just for splitting
X_train_idx, X_test_idx, _, _ = train_test_split(
    X_temp, y_noise, test_size=0.2, random_state=42
)

# Get the actual indices for splitting
train_indices = np.arange(len(x))[: len(X_train_idx)]
test_indices = np.arange(len(x))[len(X_train_idx) :]

# Split the original data
x_train = x[train_indices]
x_test = x[test_indices]
y_train = y_noise[train_indices]
y_test = y_noise[test_indices]

print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

# Storage for results
# Part 1: MSE vs degree (like Hastie Fig 2.11)
mse_train = np.zeros(len(degrees))
mse_test = np.zeros(len(degrees))

# Part 2: Bias-variance decomposition
bias_squared = np.zeros(len(degrees))
variance = np.zeros(len(degrees))
total_error = np.zeros(len(degrees))
observed_test_error = np.zeros(len(degrees))

# Part 3: Cross-validation
cv_errors = np.zeros(len(degrees))

print("\nStarting analysis...")
print("=" * 50)

# Main analysis loop
for i, degree in enumerate(degrees):
    print(f"Processing polynomial degree {degree}...")

    # Create polynomial features for this degree
    X_train_deg = polynomial_features(x_train, degree)
    X_test_deg = polynomial_features(x_test, degree)

    # === PART 1: Basic MSE Analysis (Hastie Fig 2.11 style) ===
    # Scale data
    X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train_deg, y_train)
    X_test_s, y_test_s, _, _, _ = scale_data(X_test_deg, y_test, X_mean, X_std, y_mean)

    # Fit OLS
    theta = OLS_parameters(X_train_s, y_train_s)

    # Make predictions (unscale)
    y_pred_train = X_train_s @ theta + y_mean
    y_pred_test = X_test_s @ theta + y_mean

    # Compute MSE
    mse_train[i] = np.mean((y_train - y_pred_train) ** 2)
    mse_test[i] = np.mean((y_test - y_pred_test) ** 2)

    # === PART 2: Bootstrap Bias-Variance Analysis ===
    # Store predictions from each bootstrap sample
    bootstrap_predictions = np.zeros((n_bootstrap, len(x_test)))

    for b in range(n_bootstrap):
        # Create bootstrap sample from training data
        boot_indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
        X_boot = X_train_deg[boot_indices]
        y_boot = y_train[boot_indices]

        # Scale bootstrap data
        X_boot_s, y_boot_s, X_mean_boot, X_std_boot, y_mean_boot = scale_data(
            X_boot, y_boot
        )

        # Scale test data using bootstrap scaling parameters
        X_test_s_boot = (X_test_deg - X_mean_boot) / X_std_boot

        # Fit OLS on bootstrap sample
        try:
            theta_boot = OLS_parameters(X_boot_s, y_boot_s)
            # Make predictions on test set
            bootstrap_predictions[b, :] = X_test_s_boot @ theta_boot + y_mean_boot
        except np.linalg.LinAlgError:
            # Handle singular matrix (rare for reasonable degrees)
            print(f"  Warning: Singular matrix for degree {degree}, bootstrap {b}")
            bootstrap_predictions[b, :] = np.mean(y_boot)  # Fallback to mean

    # Compute bias-variance decomposition
    y_true_test_points = runge(x_test)  # True function values at test points
    mean_prediction = np.mean(bootstrap_predictions, axis=0)

    # Bias² = (E[ỹ] - f(x))²
    bias_squared[i] = np.mean((mean_prediction - y_true_test_points) ** 2)

    # Variance = E[(ỹ - E[ỹ])²]
    variance[i] = np.mean(np.var(bootstrap_predictions, axis=0))

    # Total expected error = Bias² + Variance + σ²
    total_error[i] = bias_squared[i] + variance[i] + sigma_squared

    # Observed test error (should approximately match total_error)
    observed_test_error[i] = np.mean((mean_prediction - y_test) ** 2)

    # === PART 3: Cross-Validation ===
    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_train_deg):
        X_cv_train, X_cv_val = X_train_deg[train_idx], X_train_deg[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        # Scale CV data
        X_cv_train_s, y_cv_train_s, X_mean_cv, X_std_cv, y_mean_cv = scale_data(
            X_cv_train, y_cv_train
        )
        X_cv_val_s = (X_cv_val - X_mean_cv) / X_std_cv

        # Fit OLS
        try:
            theta_cv = OLS_parameters(X_cv_train_s, y_cv_train_s)
            y_pred_cv = X_cv_val_s @ theta_cv + y_mean_cv
            cv_scores.append(np.mean((y_cv_val - y_pred_cv) ** 2))
        except np.linalg.LinAlgError:
            cv_scores.append(np.inf)  # Penalize singular matrices

    cv_errors[i] = np.mean(cv_scores)

    print(f"  MSE Train: {mse_train[i]:.4f}, Test: {mse_test[i]:.4f}")
    print(
        f"  Bias²: {bias_squared[i]:.4f}, Variance: {variance[i]:.4f}, Total: {total_error[i]:.4f}"
    )
    print(f"  CV Error: {cv_errors[i]:.4f}")

print("\n" + "=" * 50)
print("Analysis complete! Creating plots...")

# ================================================================================================
#                                           PLOTTING
# ================================================================================================

# Plot 1: MSE vs Degree (like Hastie Fig 2.11)
plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train, "o-", label="Training MSE", linewidth=2, markersize=6)
plt.plot(degrees, mse_test, "o-", label="Test MSE", linewidth=2, markersize=6)
plt.xlabel("Polynomial Degree", fontsize=16)
plt.ylabel("Mean Squared Error", fontsize=16)
plt.title("Training and Test MSE vs Model Complexity", fontsize=16)
plt.yscale("log")
setup_plot_formatting()
plt.savefig("figs/mse_vs_degree_bias_variance.pdf")
plt.show()

# Plot 2: Bias-Variance Decomposition
plt.figure(figsize=(12, 8))
plt.plot(degrees, bias_squared, "o-", label="Bias²", linewidth=2, markersize=6)
plt.plot(degrees, variance, "s-", label="Variance", linewidth=2, markersize=6)
plt.plot(
    degrees,
    total_error,
    "^-",
    label="Total Error (Bias² + Var + σ²)",
    linewidth=2,
    markersize=6,
)
plt.plot(
    degrees,
    observed_test_error,
    "x-",
    label="Observed Test Error",
    linewidth=2,
    markersize=6,
)
plt.axhline(
    y=sigma_squared,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Irreducible Error (σ² = {sigma_squared})",
)

plt.xlabel("Polynomial Degree", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.title("Bias-Variance Decomposition for OLS Regression", fontsize=16)
plt.yscale("log")
setup_plot_formatting()
plt.savefig("figs/bias_variance_decomposition.pdf")
plt.show()

# Plot 3: Components of bias-variance on linear scale (better for interpretation)
plt.figure(figsize=(12, 8))
plt.plot(degrees, bias_squared, "o-", label="Bias²", linewidth=2, markersize=6)
plt.plot(degrees, variance, "s-", label="Variance", linewidth=2, markersize=6)
plt.axhline(
    y=sigma_squared,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Irreducible Error (σ² = {sigma_squared})",
)

plt.xlabel("Polynomial Degree", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.title("Bias² and Variance vs Model Complexity (Linear Scale)", fontsize=16)
setup_plot_formatting()
plt.savefig("figs/bias_variance_components_linear.pdf")
plt.show()

# Plot 4: Cross-validation comparison
plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_test, "o-", label="Test MSE", linewidth=2, markersize=6)
plt.plot(
    degrees, cv_errors, "s-", label="Cross-Validation MSE", linewidth=2, markersize=6
)
plt.plot(
    degrees,
    total_error,
    "^-",
    label="Expected Error (Bias-Var)",
    linewidth=2,
    markersize=6,
)

plt.xlabel("Polynomial Degree", fontsize=16)
plt.ylabel("Mean Squared Error", fontsize=16)
plt.title("Test MSE vs Cross-Validation vs Expected Error", fontsize=16)
plt.yscale("log")
setup_plot_formatting()
plt.savefig("figs/cv_vs_test_error.pdf")
plt.show()

# ================================================================================================
#                                      RESULTS SUMMARY
# ================================================================================================

print("\n" + "=" * 60)
print("BIAS-VARIANCE ANALYSIS SUMMARY")
print("=" * 60)

# Find optimal degree for different criteria
min_test_idx = np.argmin(mse_test)
min_cv_idx = np.argmin(cv_errors)
min_total_idx = np.argmin(total_error)

print(
    f"Optimal polynomial degree (Test MSE): {degrees[min_test_idx]} (MSE = {mse_test[min_test_idx]:.4f})"
)
print(
    f"Optimal polynomial degree (CV): {degrees[min_cv_idx]} (MSE = {cv_errors[min_cv_idx]:.4f})"
)
print(
    f"Optimal polynomial degree (Expected Error): {degrees[min_total_idx]} (Error = {total_error[min_total_idx]:.4f})"
)

print(f"\nAt optimal degree {degrees[min_total_idx]}:")
print(f"  Bias² = {bias_squared[min_total_idx]:.4f}")
print(f"  Variance = {variance[min_total_idx]:.4f}")
print(f"  Irreducible Error = {sigma_squared:.4f}")
print(f"  Total = {total_error[min_total_idx]:.4f}")

# Verify bias-variance decomposition
print(f"\nDecomposition verification at degree {degrees[min_total_idx]}:")
expected_sum = bias_squared[min_total_idx] + variance[min_total_idx] + sigma_squared
print(f"  Bias² + Variance + σ² = {expected_sum:.4f}")
print(f"  Total Error = {total_error[min_total_idx]:.4f}")
print(f"  Difference = {abs(expected_sum - total_error[min_total_idx]):.6f}")

# Analysis of bias-variance trade-off
print(f"\nBias-Variance Trade-off Analysis:")
print(f"  Low degrees (underfitting): High bias, low variance")
print(f"    Degree 1: Bias² = {bias_squared[0]:.4f}, Variance = {variance[0]:.4f}")
print(f"  High degrees (overfitting): Low bias, high variance")
print(
    f"    Degree {max(degrees)}: Bias² = {bias_squared[-1]:.4f}, Variance = {variance[-1]:.4f}"
)

print("\n" + "=" * 60)
print("All plots saved to figs/ directory")
print("=" * 60)
