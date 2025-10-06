import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

os.makedirs("figs", exist_ok=True)


def setup_plot_formatting():
    """Apply standard formatting to current plot."""
    plt.grid(True, alpha=0.6)

    # Only add legend if there are labeled artists
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:  # Only create legend if there are labeled items
        plt.legend(fontsize=16, framealpha=0.6)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()


# Exercise 1a)
def mse_degree_ols(results_dict, sample_size):
    """
    Plot MSE vs polynomial degree for training and test data.

    Parameters
    ----------
    results_dict : dict
        Dictionary where keys are sample sizes and values are results dictionaries
    sample_size : int
        Sample size to plot
    """
    results = results_dict[sample_size]

    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_mse"],
        "o-",
        label="MSE (train)",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        results["degrees"],
        results["test_mse"],
        "o-",
        label="MSE (test)",
        linewidth=2,
        markersize=6,
    )

    min_test_mse = min(results["test_mse"])
    min_degree = results["degrees"][results["test_mse"].index(min_test_mse)]
    plt.plot(
        min_degree,
        min_test_mse,
        "rx",
        markersize=12,
        markeredgewidth=2,
        label=f"Min Test MSE: {min_test_mse:.2f}",
    )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"MSE", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/mse_vs_degree_ols.pdf")
    plt.show()


def r2_degree_ols(results, sample_size):
    """
    Plot R² vs polynomial degree for training and test sets.

    Parameters
    ----------
    results : dict
        Results dictionary containing 'degrees', 'train_r2', 'test_r2'
    sample_size : int
        Sample size for labeling
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        results["degrees"],
        results["train_r2"],
        "o-",
        label="R² (train)",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        results["degrees"],
        results["test_r2"],
        "o-",
        label="R² (test)",
        linewidth=2,
        markersize=6,
    )

    max_test_r2 = max(results["test_r2"])
    max_degree = results["degrees"][results["test_r2"].index(max_test_r2)]

    plt.plot(
        max_degree,
        max_test_r2,
        "rx",
        markersize=12,
        markeredgewidth=2,
        label=f"Max Test R²: {max_test_r2:.2f}",
    )
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"R²", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/r2_vs_degree_ols.pdf")
    plt.show()


def theta_evolution_ols(degrees, theta_norms, sample_size):
    """
    Plot the evolution of the first parameter (theta1) across polynomial degrees.

    Parameters
    ----------
    degrees : list or range
        Polynomial degrees
    theta1_evolution : list
        List of theta1 values for each degree
    sample_size : int
        Sample size for labeling
    """
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, theta_norms, "o-", linewidth=2, markersize=6)
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(r"$\|\theta\|_2$", fontsize=16)
    plt.yscale("log")  # Usually helpful since norms can grow exponentially
    setup_plot_formatting()
    plt.savefig("figs/theta_vs_degree_ols.pdf")
    plt.show()


def mse_degree_multiple(results_dict, sample_sizes):
    """
    Plot test MSE vs polynomial degree for multiple sample sizes.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each sample size.
    sample_sizes : list
        List of sample sizes to include in the plot.
    """
    plt.figure(figsize=(10, 6))

    for N in sample_sizes:
        results = results_dict[N]
        plt.plot(
            results["degrees"],
            results["test_mse"],
            label=f"Datasize = {N}",
            linewidth=2,
            markersize=6,
        )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel("MSE (Test)", fontsize=16)
    # plt.yscale("log")  # often useful for error plots
    setup_plot_formatting()
    plt.savefig("figs/mse_vs_degree_multiple_samples_ols.pdf")
    plt.show()


def mse_degree_lambdas(results_dict, lambda_values, sample_size):
    """Plot MSE vs polynomial degree for different lambda values."""
    plt.figure(figsize=(10, 6))

    results_by_lambda = results_dict[sample_size]

    for lam in lambda_values:
        results = results_by_lambda[lam]
        plt.plot(
            results["degrees"],
            results["test_mse"],
            "o-",
            label=f"λ = {lam:.1e}",
            linewidth=2,
            markersize=4,
        )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"MSE (Test), N = {sample_size}", fontsize=16)
    plt.yscale("log")
    setup_plot_formatting()
    plt.savefig(f"figs/mse_vs_degree_lambdas_N{sample_size}.pdf")
    plt.show()


def r2_degree_lambdas(results_dict, lambda_values, sample_size):
    """Plot R² vs polynomial degree for different lambda values."""
    plt.figure(figsize=(10, 6))

    results_by_lambda = results_dict[sample_size]

    for lam in lambda_values:
        results = results_by_lambda[lam]
        plt.plot(
            results["degrees"],
            results["test_r2"],
            "o-",
            label=f"λ = {lam:.1e}",
            linewidth=2,
            markersize=4,
        )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"R² (Test), N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/r2_vs_degree_lambdas_N{sample_size}.pdf")
    plt.show()


def theta_evolution_lambdas(
    all_results, lambda_values, degrees, sample_size, include_ols=True
):
    """Plot theta1 evolution vs polynomial degree for different lambda values."""
    plt.figure(figsize=(12, 8))

    results_by_lambda = all_results[sample_size]

    for lam in lambda_values:
        results = results_by_lambda[lam]
        theta1_values = [
            instance.get_theta("ridge", "analytical")[0]
            for instance in results["instances"]
        ]
        plt.plot(
            degrees,
            theta1_values,
            "o-",
            label=f"Ridge λ = {lam:.1e}",
            linewidth=2,
            markersize=4,
        )

    if include_ols:
        first_results = results_by_lambda[lambda_values[0]]
        ols_theta1_values = [
            instance.get_theta("ols", "analytical")[0]
            for instance in first_results["instances"]
        ]
        plt.plot(
            degrees,
            ols_theta1_values,
            "--",
            label="OLS (λ = 0)",
            linewidth=3,
            markersize=6,
        )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"Evolution of $\\theta_1$, N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/theta1_evolution_lambdas_N{sample_size}.pdf")
    plt.show()


def compare(
    x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False
):
    """
    Plot true function and model predictions (analytical/sklearn + GD).

    Parameters
    ----------
    type : str
        Type of regression: 'lasso' for Lasso regression, otherwise treats as OLS/Ridge
    test : bool
        If True, highlights test data points in a different color
    """
    # Use type parameter to determine what baseline to expect
    if type == "lasso":
        y_scikit, y_gd, x_plotting = solutions
        lasso = True
    else:
        y_analytical, y_gd, x_plotting = solutions
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")

    if test:
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask

        plt.scatter(
            x[train_mask],
            y_noise[train_mask],
            color="lightgray",
            alpha=0.5,
            s=50,
            label=r"$y_{noise}$ (train)",
        )
        plt.scatter(
            x[test_mask],
            y_noise[test_mask],
            color="dimgray",
            alpha=0.7,
            s=50,
            label=r"$y_{noise}$ (test)",
        )
    else:
        plt.scatter(
            x, y_noise, color="lightgray", alpha=0.7, s=50, label=r"$y_{noise}$"
        )

    if lasso:
        plt.plot(x_plotting, y_scikit, label=f"Scikit-Learn, λ={lam:.0e}")
    else:
        plt.plot(x_plotting, y_analytical, label=f"Analytical, λ={lam:.0e}")

    plt.plot(x_plotting, y_gd, label=f"GD, λ={lam:.0e}")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=16)

    title = (
        f'Test split - {type if type else "Comparison"}'
        if test
        else f'Full dataset - {type if type else "Comparison"}'
    )
    plt.title(title, fontsize=16)
    plt.show()


def compare_gd(
    x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False
):
    """
    Plotting solutions using different methods for computing the optimal parameters (gradient descent).

    Parameters
    ----------
    test : bool
        If True, highlights test data points in a different color
    """

    if type == "lasso":
        y_scikit, y_gd, y_momentum, y_adagrad, y_rmsprop, y_adam, x_plotting = solutions
        lasso = True
    else:
        y_analytical, y_gd, y_momentum, y_adagrad, y_rmsprop, y_adam, x_plotting = (
            solutions
        )
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")

    if test:
        # Find which points from the full dataset are NOT in the test set (i.e., training points)
        # Create a mask for test points
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask

        # Plot training points in light gray
        plt.scatter(
            x[train_mask],
            y_noise[train_mask],
            color="lightgray",
            alpha=0.5,
            s=50,
            label=r"$y_{noise}$ (train)",
        )
        # Plot test points in a different color to highlight them
        plt.scatter(
            x[test_mask],
            y_noise[test_mask],
            color="dimgray",
            alpha=0.7,
            s=50,
            label=r"$y_{noise}$ (test)",
        )
    else:
        # Plot all data points in one color
        plt.scatter(
            x, y_noise, color="lightgray", alpha=0.7, s=50, label=r"$y_{noise}$"
        )

    if lasso:
        plt.plot(x_plotting, y_scikit, label="Scikit-Learn")
    else:
        plt.plot(x_plotting, y_analytical, label="Analytical")

    plt.plot(x_plotting, y_gd, label="Gradient Descent")
    plt.plot(x_plotting, y_momentum, label="Momentum")
    plt.plot(x_plotting, y_adagrad, label="AdaGrad")
    plt.plot(x_plotting, y_rmsprop, label="RMSProp")
    plt.plot(x_plotting, y_adam, label="Adam")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=14)
    title = (
        f'Test split - GD Methods {type if type else ""}'
        if test
        else f'Full dataset - GD Methods {type if type else ""}'
    )
    plt.title(title, fontsize=16)
    plt.show()


def compare_sgd(
    x, y_noise, y_true, solutions, sample_size, degree, lam, type=None, test=False
):
    """
    Plot true function and model predictions (analytical/sklearn + GD + SGD).

    Parameters
    ----------
    type : str
        Type of regression: 'lasso' for Lasso regression, otherwise treats as OLS/Ridge
    test : bool
        If True, highlights test data points in a different color
    """
    # Use type parameter to determine what baseline to expect
    if type == "lasso":
        y_pred_sklearn, y_pred_gd, y_pred_sgd, x_plotting = solutions
        lasso = True
    else:
        y_pred_analytical, y_pred_gd, y_pred_sgd, x_plotting = solutions
        lasso = False

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")

    if test:
        test_mask = np.isin(x, x_plotting)
        train_mask = ~test_mask

        plt.scatter(
            x[train_mask],
            y_noise[train_mask],
            color="lightgray",
            alpha=0.5,
            s=50,
            label=r"$y_{noise}$ (train)",
        )
        plt.scatter(
            x[test_mask],
            y_noise[test_mask],
            color="dimgray",
            alpha=0.7,
            s=50,
            label=r"$y_{noise}$ (test)",
        )
    else:
        plt.scatter(
            x, y_noise, color="lightgray", alpha=0.7, s=50, label=r"$y_{noise}$"
        )

    if lasso:
        plt.plot(x_plotting, y_pred_sklearn, label=f"Scikit-Learn, λ={lam:.0e}")
    else:
        plt.plot(x_plotting, y_pred_analytical, label=f"Analytical, λ={lam:.0e}")

    plt.plot(x_plotting, y_pred_gd, label=f"GD, λ={lam:.0e}")
    plt.plot(x_plotting, y_pred_sgd, label=f"SGD, λ={lam:.0e}")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=14)

    title = (
        f'Test split - SGD Methods {type if type else ""}'
        if test
        else f'Full dataset - SGD Methods {type if type else ""}'
    )
    plt.title(title, fontsize=16)
    plt.show()


## Bootstrap plots
def plot_diagonal_comparison(results, diagonal_points, n_bootstraps=2000):
    """
    Plot model fits along the diagonal with individual bootstrap samples.

    Shows multiple bootstrap predictions to illustrate variance through visual density:
    - Low complexity: predictions cluster together (dense overlap)
    - High complexity: predictions vary wildly (sparse spread)

    Parameters
    ----------
    results : dict
        Results from bootstrap_bias_variance_analysis
    diagonal_points : list of tuples
        List of (n, degree) pairs to plot
        Example: [(200, 3), (200, 20)]
    """
    n_plots = len(diagonal_points)

    # Calculate grid layout: 2 columns, multiple rows
    ncols = 2
    nrows = (n_plots + 1) // 2  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))

    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Hide unused subplots if odd number of plots
    if n_plots % 2 == 1:
        axes[-1].set_visible(False)

    for idx, (n, degree) in enumerate(diagonal_points):
        ax = axes[idx]

        # Get saved fit data
        fit_data = results["diagonal_fits"][n][degree]
        x_plot = fit_data["x_plot"]
        x_test = fit_data["x_test"]
        y_test = fit_data["y_test"]

        bootstrap_predictions = fit_data["bootstrap_predictions"]

        # Plot individual bootstrap predictions (uniform color, transparency shows density)
        for i, y_pred in enumerate(bootstrap_predictions):
            label = (
                f"{len(bootstrap_predictions)} bootstrap samples" if i == 0 else None
            )
            ax.plot(
                x_plot,
                y_pred,
                color="darkviolet",
                linewidth=0.5,
                alpha=0.1,
                label=label,
                zorder=4,
            )

        # Plot training data (low opacity)
        ax.scatter(
            fit_data["x_train"],
            fit_data["y_train"],
            c="#1E88E5",
            s=30,
            alpha=0.6,
            label="Training data",
            zorder=2,
            edgecolors="none",
        )

        # Plot test data (higher opacity)
        ax.scatter(
            x_test,
            y_test,
            c="#F2B44D",
            s=30,
            alpha=0.7,
            label="Test data",
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add text with bias-variance values at y = 0, x = 0 in data coords
        textstr = f"{n_bootstraps} bootstraps avg.:\n"
        textstr += f"Bias²: {fit_data['bias_squared']:.4f}\n"
        textstr += f"Var: {fit_data['variance']:.4f}\n"
        textstr += f"Error: {fit_data['error']:.4f}"
        ax.text(
            0.50,
            0.15,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            zorder=15,
        )

        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title(
            f"Data-size {n} with degree={degree}",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_ylim([-0.3, 1.2])

    plt.tight_layout(pad=3.0)
    plt.savefig("figs/diagonal_comparison_bootstrap_samples.pdf", dpi=300)
    plt.show()


def plot_bias_variance_decomposition(results, sample_size_idx=0):
    """Plot bias-variance decomposition for a specific sample size."""
    sample_size = results["sample_sizes"][sample_size_idx]
    degrees = results["degrees"]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axhline(
        results["noise_var"],
        color="black",
        linestyle="--",
        label="Irreducible error",
        linewidth=2.5,
        alpha=0.7,
    )

    ax.plot(
        degrees,
        results["error_matrix"][sample_size_idx, :],
        "-",
        label="Test Error (MSE)",
        linewidth=2.5,
        markersize=7,
        color="#D63290",
        zorder=5,
    )

    ax.plot(
        degrees,
        results["expected_error_matrix"][sample_size_idx, :],
        "-o",
        label="Bias² + Variance",
        linewidth=2,
        markersize=6,
        alpha=1,
        color="#F2B44D",
    )

    ax.plot(
        degrees,
        results["bias_squared_matrix"][sample_size_idx, :],
        "-",
        label="Bias²",
        linewidth=2.5,
        markersize=7,
        color="darkviolet",
    )

    ax.plot(
        degrees,
        results["variance_matrix"][sample_size_idx, :],
        "-",
        label="Variance",
        linewidth=2.5,
        markersize=7,
        color="#1E88E5",
    )

    ax.set_xlabel("Polynomial Degree", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_title(
        f"Bias-Variance Decomposition (Sample-size: {sample_size})",
        fontsize=18,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    # Y-axis limit to focus on relevant range
    ax.set_ylim(bottom=0, top=np.max(results["error_matrix"][sample_size_idx, :]) * 1.1)
    plt.tight_layout()
    plt.savefig("figs/bias_variance_decomposition.pdf")
    plt.show()


def plot_heatmaps(results):
    """Create three subplots for Error, Bias², and Variance heatmaps."""
    sample_sizes = results["sample_sizes"]
    degrees = results["degrees"]

    # Create figure with 3 subplots (no sharey due to seaborn compatibility issues)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Error heatmap
    sns.heatmap(
        results["error_matrix"],
        ax=axes[0],
        cmap="plasma",
        xticklabels=degrees,
        yticklabels=sample_sizes,
        cbar_kws={"label": "MSE"},
        annot=False,
    )

    # Find and mark the best model complexity (lowest MSE)
    min_row, min_col = np.unravel_index(
        np.argmin(results["error_matrix"]), results["error_matrix"].shape
    )
    rect = plt.Rectangle(
        (min_col, min_row), 1, 1, facecolor="none", edgecolor="red", linewidth=2
    )
    axes[0].add_patch(rect)

    axes[0].set_title("Test Error (MSE)", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Polynomial Degree", fontsize=14)
    axes[0].set_ylabel("Sample Size (n)", fontsize=14)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0)

    # Bias² heatmap
    sns.heatmap(
        results["bias_squared_matrix"],
        ax=axes[1],
        cmap="plasma",
        xticklabels=degrees,
        yticklabels=sample_sizes,  # Keep the same labels for proper alignment
        cbar_kws={"label": "Bias²"},
        annot=False,
    )
    axes[1].set_title("Bias²", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Polynomial Degree", fontsize=14)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)
    # Hide y-axis labels and ticks for middle plot but keep the same scale
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])

    # Variance heatmap
    sns.heatmap(
        results["variance_matrix"],
        ax=axes[2],
        cmap="plasma",
        xticklabels=degrees,
        yticklabels=sample_sizes,  # Keep the same labels for proper alignment
        cbar_kws={"label": "Variance"},
        annot=False,
    )
    axes[2].set_title("Variance", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("Polynomial Degree", fontsize=14)
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].tick_params(axis="y", rotation=0)
    # Hide y-axis labels and ticks for right plot but keep the same scale
    axes[2].set_ylabel("")
    axes[2].set_yticklabels([])

    # Manually ensure all plots have the same y-limits
    y_min = min([ax.get_ylim()[0] for ax in axes])
    y_max = max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout(pad=3.0)
    plt.savefig("figs/bias_variance_heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# cross validation plots
def plot_cv_vs_bootstrap_comparison(cv_results, bootstrap_results, sample_size_idx=0):
    """
    Compare cross-validation and bootstrap MSE results for OLS.

    Parameters
    ----------
    cv_results : dict
        Results from cross_validation_analysis
    bootstrap_results : dict
        Results from bootstrap_bias_variance_analysis
    sample_size_idx : int
        Index of sample size to plot
    """
    sample_size = cv_results["sample_sizes"][sample_size_idx]
    degrees = cv_results["degrees"]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Bootstrap results
    ax.plot(
        degrees,
        bootstrap_results["error_matrix"][sample_size_idx, :],
        "o-",
        label="Bootstrap MSE",
        linewidth=2.5,
        markersize=7,
        color="#D63290",
        alpha=0.8,
    )

    # Cross-validation results
    cv_mse = cv_results["ols"]["mse_matrix"][sample_size_idx, :]
    cv_std = cv_results["ols"]["mse_std"][sample_size_idx, :]

    ax.plot(
        degrees,
        cv_mse,
        "s-",
        label=f'{cv_results["k_folds"]}-Fold CV MSE',
        linewidth=2.5,
        markersize=7,
        color="darkviolet",
        alpha=0.8,
    )

    # Add error bars for CV
    ax.fill_between(
        degrees,
        cv_mse - cv_std,
        cv_mse + cv_std,
        alpha=0.2,
        color="darkviolet",
        label="CV MSE ± 1 std",
    )

    ax.set_xlabel("Polynomial Degree", fontsize=16)
    ax.set_ylabel("MSE", fontsize=16)
    ax.set_title(
        f"Bootstrap vs Cross-Validation (OLS, n={sample_size})",
        fontsize=18,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    ax.set_ylim(
        bottom=0,
        top=np.max(bootstrap_results["error_matrix"][sample_size_idx, :]) * 1.1,
    )

    plt.tight_layout()
    plt.savefig("figs/bootstrap_vs_cv_comparison.pdf")
    plt.show()


def plot_cv_heatmaps_multi_noise(
    noise_results_dict,
    sample_sizes,
    degrees,
    k_folds=5,
    filename="figs/cv_mse_multi_noise.pdf",
):
    """
    Create 3×3 subplot of CV MSE heatmaps: 3 noise levels × 3 models.

    Parameters
    ----------
    noise_results_dict : dict
        Dictionary with structure:
        {
            "low": {"cv_results": ..., "σ": 0.1},
            "medium": {...},
            "high": {...}
        }
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    noise_order = ["low", "medium", "high"]
    noise_labels = {"low": "σ=0.1", "medium": "σ=0.2", "high": "σ=0.3"}
    model_names = ["OLS", "Ridge (Best λ)", "Lasso (Best λ)"]

    for row, noise_level in enumerate(noise_order):
        cv_results = noise_results_dict[noise_level]["cv_results"]
        lambda_values = cv_results["lambda_values"]

        # OLS (column 0)
        ax = axes[row, 0]
        ols_mse = cv_results["ols"]["mse_matrix"]

        sns.heatmap(
            ols_mse,
            cmap="plasma",
            xticklabels=degrees,
            yticklabels=sample_sizes,
            cbar_kws={"label": "MSE"},
            annot=False,
            ax=ax,
        )

        # Mark minimum
        min_idx = np.unravel_index(np.argmin(ols_mse), ols_mse.shape)
        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (min_idx[1], min_idx[0]),
            1,
            1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.set_title(
            f"{model_names[0]} | {noise_labels[noise_level]}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Sample Size (n)", fontsize=11)
        if row == 2:
            ax.set_xlabel("Polynomial Degree", fontsize=11)
        else:
            ax.set_xlabel("")

        # Ridge (column 1) - best lambda per cell
        ax = axes[row, 1]
        ridge_cube = np.stack(
            [cv_results["ridge"][lam]["mse_matrix"] for lam in lambda_values], axis=2
        )
        ridge_best_mse = ridge_cube.min(axis=2)

        sns.heatmap(
            ridge_best_mse,
            cmap="plasma",
            xticklabels=degrees,
            yticklabels=sample_sizes,
            cbar_kws={"label": "MSE"},
            annot=False,
            ax=ax,
        )

        min_idx = np.unravel_index(np.argmin(ridge_best_mse), ridge_best_mse.shape)
        rect = patches.Rectangle(
            (min_idx[1], min_idx[0]),
            1,
            1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.set_title(
            f"{model_names[1]} | {noise_labels[noise_level]}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("")
        if row == 2:
            ax.set_xlabel("Polynomial Degree", fontsize=11)
        else:
            ax.set_xlabel("")

        # Lasso (column 2) - best lambda per cell
        ax = axes[row, 2]
        lasso_cube = np.stack(
            [cv_results["lasso"][lam]["mse_matrix"] for lam in lambda_values], axis=2
        )
        lasso_best_mse = lasso_cube.min(axis=2)

        sns.heatmap(
            lasso_best_mse,
            cmap="plasma",
            xticklabels=degrees,
            yticklabels=sample_sizes,
            cbar_kws={"label": "MSE"},
            annot=False,
            ax=ax,
        )

        min_idx = np.unravel_index(np.argmin(lasso_best_mse), lasso_best_mse.shape)
        rect = patches.Rectangle(
            (min_idx[1], min_idx[0]),
            1,
            1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.set_title(
            f"{model_names[2]} | {noise_labels[noise_level]}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("")
        if row == 2:
            ax.set_xlabel("Polynomial Degree", fontsize=11)
        else:
            ax.set_xlabel("")

    fig.suptitle(
        f"Cross-Validation MSE Across Noise Levels ({k_folds}-Fold CV)\n"
        "Red box = minimum MSE",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")


def plot_delta_heatmap_multi_noise(
    noise_results_dict,
    sample_sizes,
    degrees,
    filename="figs/delta_heatmap_multi_noise.pdf",
):
    """
    Create 3×2 subplot of *relative* delta heatmaps for all noise levels.
    Color encodes mean signed relative Δ across seeds:
      Δ_rel = (OLS − Model@λ*) / OLS, shown in percent.
    Positive (green): regularization improves over OLS; Negative (red): hurts.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    noise_order = ["low", "medium", "high"]
    noise_labels = {
        "low": "Low Noise (σ=0.1)",
        "medium": "Medium Noise (σ=0.2)",
        "high": "High Noise (σ=0.3)",
    }

    # Compute a global symmetric color range (in %) to make rows comparable
    all_rel = []
    for nl in noise_order:
        d = noise_results_dict[nl]
        all_rel.append(d["delta_ridge"])
        all_rel.append(d["delta_lasso"])
    all_rel = np.concatenate([x.flatten() for x in all_rel])
    # Use a robust limit (e.g., 98th percentile of |Δ_rel|), then make symmetric
    vmax_abs = np.quantile(np.abs(all_rel), 0.98) if all_rel.size > 0 else 0.05
    vmin, vmax = -100 * vmax_abs, 100 * vmax_abs  # convert to %

    for row, noise_level in enumerate(noise_order):
        data = noise_results_dict[noise_level]

        # Convert to percent for display
        ridge_rel_pct = 100.0 * data["delta_ridge"]
        lasso_rel_pct = 100.0 * data["delta_lasso"]

        # Ridge heatmap (left column)
        ax_ridge = axes[row, 0]
        sns.heatmap(
            ridge_rel_pct,
            cmap="RdYlGn",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=degrees,
            yticklabels=sample_sizes,
            cbar_kws={"label": "Mean relative Δ (%)"},
            annot=False,
            linewidths=0.5,
            linecolor="gray",
            ax=ax_ridge,
        )

        # Mark maximum improvement cell (most positive Δ_rel), optional
        max_idx = np.unravel_index(np.argmax(ridge_rel_pct), ridge_rel_pct.shape)
        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (max_idx[1], max_idx[0]),
            1,
            1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax_ridge.add_patch(rect)

        ax_ridge.set_title(
            f"Ridge | {noise_labels[noise_level]}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax_ridge.set_ylabel("Sample Size (n)", fontsize=12)
        if row == 2:  # Bottom row
            ax_ridge.set_xlabel("Polynomial Degree", fontsize=12)
        else:
            ax_ridge.set_xlabel("")

        # Lasso heatmap (right column)
        ax_lasso = axes[row, 1]
        sns.heatmap(
            lasso_rel_pct,
            cmap="RdYlGn",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=degrees,
            yticklabels=sample_sizes,
            cbar_kws={"label": "Mean relative Δ (%)"},
            annot=False,
            linewidths=0.5,
            linecolor="gray",
            ax=ax_lasso,
        )

        # Mark maximum improvement cell for Lasso (optional)
        max_idx = np.unravel_index(np.argmax(lasso_rel_pct), lasso_rel_pct.shape)
        rect = patches.Rectangle(
            (max_idx[1], max_idx[0]),
            1,
            1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax_lasso.add_patch(rect)

        ax_lasso.set_title(
            f"Lasso | {noise_labels[noise_level]}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax_lasso.set_ylabel("")
        if row == 2:  # Bottom row
            ax_lasso.set_xlabel("Polynomial Degree", fontsize=12)
        else:
            ax_lasso.set_xlabel("")

    # Overall title
    fig.suptitle(
        "Regularization Effectiveness Across Noise Levels\n"
        "Color = Mean relative improvement over OLS (%, green = better)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")
