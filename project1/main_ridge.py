import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

from src.plotting import mse_degree_ridge, r2_degree_ridge, plot_theta
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZES = [
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
]
LAMBDAS = np.logspace(-5, 2, 8)
DEGREES = range(1, 16)
"""
Note: To produce MSE and R2 plots (exactly as in the report), use 
DEGREES = range(1, 36)
"""

# ============================================================================
# DATA GENERATION AND MODEL FITTING
# ============================================================================

all_results = {}
theta_list = []

for N in SAMPLE_SIZES:
    results_for_N = {}

    for lam in LAMBDAS:
        # Generate data
        np.random.seed(2018)
        x = np.linspace(-1, 1, N)
        random_noise = np.random.normal(0, 0.1, N)
        y_true = runge(x)
        y_noise = y_true + random_noise

        # Prepare data splits for each degree
        data_splits = {}
        for deg in DEGREES:
            X = polynomial_features(x, deg)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_noise, test_size=0.25, random_state=42
            )

            # Scale data
            X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
            X_test_s, y_test_s, _, _, _ = scale_data(
                X_test, y_test, X_mean, X_std, y_mean
            )

            x_train = X_train[:, 0]
            x_test = X_test[:, 0]
            data_splits[deg] = [
                X_train_s,
                X_test_s,
                y_train_s,
                y_test_s,
                x_train,
                x_test,
                y_mean,
            ]

        # Fit models for all degrees
        results_current = {
            "lambda": lam,
            "degrees": list(DEGREES),
            "train_mse": [],
            "test_mse": [],
            "train_r2": [],
            "test_r2": [],
            "instances": [],
        }

        for deg in DEGREES:
            analysis = RegressionAnalysis(
                data_splits[deg], degree=deg, lam=lam, eta=None, num_iters=None
            )
            analysis.fit(models="ridge", opts="analytical")

            if N == 50 and lam == 0.01:
                theta = analysis.get_theta("ridge", "analytical")
                theta_list.append(theta)

            results_current["train_mse"].append(
                analysis.get_metric("ridge", "analytical", "train_mse")
            )
            results_current["test_mse"].append(
                analysis.get_metric("ridge", "analytical", "test_mse")
            )
            results_current["train_r2"].append(
                analysis.get_metric("ridge", "analytical", "train_r2")
            )
            results_current["test_r2"].append(
                analysis.get_metric("ridge", "analytical", "test_r2")
            )
            results_current["instances"].append(analysis)

        results_for_N[lam] = results_current

    all_results[N] = results_for_N

    if N == 50:
        results = results_for_N[0.01]
        mse_degree_ridge(results)
        r2_degree_ridge(results)


plot_theta(theta_list, method="Ridge")
# ============================================================================
# PREPARE DATA AND FIND GLOBAL OPTIMUM
# ============================================================================

# Prepare data for all visualizations
all_data = []
for N in SAMPLE_SIZES:
    for lam in LAMBDAS:
        for i, deg in enumerate(all_results[N][lam]["degrees"]):
            all_data.append(
                {
                    "Sample Size": N,
                    "Lambda": lam,
                    "log10(Lambda)": np.log10(lam),
                    "Degree": deg,
                    "Test MSE": all_results[N][lam]["test_mse"][i],
                }
            )

df_all = pd.DataFrame(all_data)

# Find global minimum
global_min = df_all["Test MSE"].min()
min_row = df_all[df_all["Test MSE"] == global_min].iloc[0]
global_min_N = min_row["Sample Size"]
global_min_lambda = min_row["Lambda"]
global_min_degree = min_row["Degree"]

print(f"\n{'=' * 60}")
print(f"GLOBAL OPTIMAL PARAMETERS FOR RIDGE REGRESSION")
print(f"{'=' * 60}")
print(f"Sample Size (N):      {global_min_N}")
print(
    f"Lambda (λ):           {global_min_lambda:.2e} (log₁₀(λ) = {np.log10(global_min_lambda):.1f})"
)
print(f"Polynomial Degree:    {global_min_degree}")
print(f"Minimum Test MSE:     {global_min:.6f}")
print(f"{'=' * 60}\n")


# ============================================================================
# 3D SCATTER PLOT WITH MATPLOTLIB (FOR PDF EXPORT)
# ============================================================================

# Create figure
fig_3d = plt.figure(figsize=(16, 10))
ax_3d = fig_3d.add_subplot(111, projection="3d")

# Create scatter plot
scatter = ax_3d.scatter(
    df_all["log10(Lambda)"],
    df_all["Degree"],
    df_all["Sample Size"],
    c=df_all["Test MSE"],
    cmap="plasma",
    s=100,
)

# Highlight global minimum
min_point = df_all[df_all["Test MSE"] == global_min].iloc[0]
ax_3d.scatter(
    [min_point["log10(Lambda)"]],
    [min_point["Degree"]],
    [min_point["Sample Size"]],
    c="#F2B44D",
    edgecolors="black",
    s=1200,
    marker="*",
    depthshade=False,
    zorder=10,  # ensures it's drawn on top
    label=f"Min(MSE) = {global_min:.6f}\n"
    + r"$\lambda$"
    + f"= {global_min_lambda}\n"
    + f"N = {global_min_N}\n"
    + f"Degree = {global_min_degree}",
)


# Add colorbar
cbar = fig_3d.colorbar(scatter, ax=ax_3d, shrink=0.8)
cbar.set_label("MSE", fontsize=16)
cbar.ax.tick_params(labelsize=16)

# Set labels
ax_3d.set_xlabel("log₁₀(λ)", fontsize=16, labelpad=10)
ax_3d.set_ylabel("Polynomial Degree", fontsize=16, labelpad=10)
ax_3d.set_zlabel("Sample Size", fontsize=16, labelpad=10)

ax_3d.tick_params(axis="x", labelsize=16)
ax_3d.tick_params(axis="y", labelsize=16)
ax_3d.tick_params(axis="z", labelsize=16)

# Adjust viewing angle
ax_3d.view_init(elev=20, azim=45)

# Add legend
ax_3d.legend(loc="upper left", fontsize=16)

# Improve tick label size
ax_3d.tick_params(axis="both", which="major", labelsize=16)

plt.tight_layout()
ax_3d.set_box_aspect([1, 1, 1])  # Equal aspect ratio
plt.savefig("figs/mse_3d_scatter_ridge.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figs/mse_3d_scatter_ridge.png", dpi=300, bbox_inches="tight")
plt.show()
