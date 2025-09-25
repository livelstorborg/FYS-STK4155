import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.plotting import *
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


# Lasso Analysis Parameters
N = 500  # sample size
degree = 8  # polynomial degree (high degree to show sparsity effect)
eta = 5e-4  # learning rate for GD (smaller for Lasso stability)
num_iters = 5000

# Multiple lambda values to compare sparsity vs. performance trade-off
lambda_values = np.logspace(-4, 1, 10)  # From 0.0001 to 10
print(f"Testing lambda values: {lambda_values}")

# Generate data
np.random.seed(42)
x = np.linspace(-1, 1, N)
y_true = runge(x) + np.random.normal(0, 0.1, N)

# Create polynomial features
X = polynomial_features(x, degree)
X_norm, y_centered, y_mean = scale_data(X, y_true)

# Train/test split
X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
    X_norm, y_centered, x, test_size=0.2, random_state=42
)
data = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]

# Storage for results
results = {
    "lambda": [],
    "train_mse": [],
    "test_mse": [],
    "train_r2": [],
    "test_r2": [],
    "n_nonzero_coefs": [],
    "theta_values": [],
    "instances": [],
}

print("=" * 60)
print("LASSO REGRESSION ANALYSIS")
print("=" * 60)
print(f"Dataset: N={N}, degree={degree}")
print(f"Gradient descent: eta={eta}, iterations={num_iters}")
print("=" * 60)

# Analyze different lambda values
for i, lam in enumerate(lambda_values):
    print(f"\nProcessing λ = {lam:.4f} ({i + 1}/{len(lambda_values)})...")

    # Create analysis instance
    analysis = RegressionAnalysis(
        data, degree=degree, lam=lam, eta=eta, num_iters=num_iters
    )

    # Fit all methods for comparison
    analysis.fit_one('ols', 'analytical')  # OLS 
    analysis.fit_one('ridge', 'analytical')  # Ridge
    analysis.fit_one('lasso', 'gd')  # Lasso

    # Store results
    results["lambda"].append(lam)
    results["train_mse"].append(analysis.get_metric('lasso', 'gd', 'train_mse'))
    results["test_mse"].append(analysis.get_metric('lasso', 'gd', 'test_mse'))
    results["train_r2"].append(analysis.get_metric('lasso', 'gd', 'train_r2'))
    results["test_r2"].append(analysis.get_metric('lasso', 'gd', 'test_r2'))

    # Count non-zero coefficients (sparsity measure)
    theta_lasso = analysis.get_theta('lasso', 'gd')
    n_nonzero = np.sum(np.abs(theta_lasso) > 1e-6)
    results["n_nonzero_coefs"].append(n_nonzero)
    results["theta_values"].append(theta_lasso.copy())
    results["instances"].append(analysis)

    print(f"  Train MSE: {analysis.get_metric('lasso', 'gd', 'train_mse'):.6f}")
    print(f"  Test MSE:  {analysis.get_metric('lasso', 'gd', 'test_mse'):.6f}")
    print(f"  Test R²:   {analysis.get_metric('lasso', 'gd', 'test_r2'):.6f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{degree}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Find optimal lambda
optimal_idx = np.argmin(results["test_mse"])
optimal_lambda = results["lambda"][optimal_idx]
optimal_analysis = results["instances"][optimal_idx]

print(f"Optimal λ = {optimal_lambda:.4f}")
print(f"Best Test MSE: {results['test_mse'][optimal_idx]:.6f}")
print(f"Best Test R²:  {results['test_r2'][optimal_idx]:.6f}")
print(
    f"Sparsity: {results['n_nonzero_coefs'][optimal_idx]}/{degree} non-zero coefficients"
)

# Compare with OLS and Ridge at optimal lambda
print(f"\nComparison at λ = {optimal_lambda:.4f}:")
print(
    f"{'Method':<15} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<12} {'Non-zero':<10}"
)
print("-" * 65)

methods_to_compare = ["ols", "ridge", "lasso"]
method_names = ["OLS", "Ridge", "Lasso"]
optimizers = ["analytical", "analytical", "gd"]

for method, name, opt in zip(methods_to_compare, method_names, optimizers):
    train_mse = optimal_analysis.get_metric(method, opt, 'train_mse')
    test_mse = optimal_analysis.get_metric(method, opt, 'test_mse')
    test_r2 = optimal_analysis.get_metric(method, opt, 'test_r2')
    theta = optimal_analysis.get_theta(method, opt)

    n_nonzero = np.sum(np.abs(theta) > 1e-6)
    print(
        f"{name:<15} {train_mse:<12.6f} {test_mse:<12.6f} {test_r2:<12.6f} {n_nonzero:<10}"
    )

print("\n" + "=" * 60)
print("SPARSITY ANALYSIS")
print("=" * 60)

# Show coefficient evolution with lambda
print("Coefficient sparsity vs regularization:")
print(f"{'Lambda':<12} {'Non-zero coefs':<15} {'Sparsity %':<12}")
print("-" * 40)
for lam, n_nonzero in zip(results["lambda"], results["n_nonzero_coefs"]):
    sparsity_pct = (1 - n_nonzero / degree) * 100
    print(f"{lam:<12.4f} {n_nonzero:<15} {sparsity_pct:<12.1f}%")

# Generate plots
print("\nGenerating plots...")

# Plot 1: MSE vs Lambda
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.semilogx(
    results["lambda"], results["train_mse"], "o-", label="Train MSE", color="blue"
)
plt.semilogx(
    results["lambda"], results["test_mse"], "s-", label="Test MSE", color="red"
)
plt.axvline(
    optimal_lambda,
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"Optimal λ={optimal_lambda:.4f}",
)
plt.xlabel("Lambda (λ)")
plt.ylabel("MSE")
plt.title("Lasso: MSE vs Regularization")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Sparsity vs Lambda
plt.subplot(2, 2, 2)
plt.semilogx(results["lambda"], results["n_nonzero_coefs"], "o-", color="purple")
plt.axvline(optimal_lambda, color="green", linestyle="--", alpha=0.7)
plt.xlabel("Lambda (λ)")
plt.ylabel("Number of Non-zero Coefficients")
plt.title("Lasso: Sparsity vs Regularization")
plt.grid(True, alpha=0.3)

# Plot 3: R² vs Lambda
plt.subplot(2, 2, 3)
plt.semilogx(
    results["lambda"], results["train_r2"], "o-", label="Train R²", color="blue"
)
plt.semilogx(results["lambda"], results["test_r2"], "s-", label="Test R²", color="red")
plt.axvline(optimal_lambda, color="green", linestyle="--", alpha=0.7)
plt.xlabel("Lambda (λ)")
plt.ylabel("R² Score")
plt.title("Lasso: R² vs Regularization")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Coefficient paths
plt.subplot(2, 2, 4)
theta_matrix = np.array(results["theta_values"]).T  # Shape: (n_features, n_lambdas)
for i in range(min(5, theta_matrix.shape[0])):  # Show first 5 coefficients
    plt.semilogx(
        results["lambda"], theta_matrix[i], "o-", label=f"θ_{i + 1}", alpha=0.7
    )
plt.axvline(optimal_lambda, color="green", linestyle="--", alpha=0.7)
plt.xlabel("Lambda (λ)")
plt.ylabel("Coefficient Value")
plt.title("Lasso: Coefficient Paths")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/lasso_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("Analysis complete! Plot saved to 'figs/lasso_analysis.png'")
print("Key findings:")
print(f"- Optimal λ = {optimal_lambda:.4f} gives best test performance")
print(f"- Lasso achieved {results['n_nonzero_coefs'][optimal_idx]}/{degree} sparsity")
print("- As λ increases, more coefficients become exactly zero (feature selection)")
print("- Trade-off between model complexity and prediction accuracy")
