import matplotlib.pyplot as plt


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
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"MSE, N = {sample_size}", fontsize=16)
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
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"R², N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/r2_vs_degree_ols.pdf")
    plt.show()


def theta_evolution_ols(degrees, theta1_evolution, sample_size):
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
    plt.plot(degrees, theta1_evolution, "o-", linewidth=2, markersize=6)
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"Evolution of $\\theta_1$, N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig("figs/theta1_vs_degree_ols.pdf")
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
            "o-",
            label=f"N = {N}",
            linewidth=2,
            markersize=6,
        )

    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel("MSE (Test)", fontsize=16)
    plt.yscale("log")  # often useful for error plots
    setup_plot_formatting()
    plt.savefig("figs/mse_vs_degree_multiple_samples_ols.pdf")
    plt.show()


# Exercise 1b)
def mse_degree_ridge(results_dict, lam, sample_size):
    """Plot train vs test MSE for Ridge regression with specific lambda."""
    results = results_dict[lam]

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
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"MSE, N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/mse_vs_degree_ridge_lambda_{lam:.1e}_N{sample_size}.pdf")
    plt.show()


def r2_degree_ridge(results, sample_size):
    """Plot R² for Ridge regression."""
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
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"R², N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/r2_vs_degree_ridge_N{sample_size}.pdf")
    plt.show()


def theta_evolution_ridge(degrees, theta1_evolution, lam, sample_size):
    """Plot theta1 evolution for Ridge regression."""
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, theta1_evolution, "o-", linewidth=2, markersize=6)
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel(f"Evolution of $\\theta_1$, N = {sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.savefig(f"figs/theta1_evolution_ridge_lambda_{lam:.1e}_N{sample_size}.pdf")
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
            instance.get_theta('ridge', 'analytical')[0] for instance in results["instances"]
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
            instance.get_theta('ols', 'analytical')[0] for instance in first_results["instances"]
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


def solution_comparison(x, y_noise, y_true, solutions, sample_size, degree, lam, test=False):
    """
    Plot true function and model predictions from OLS and Ridge (analytical + GD).
    """

    (
        y_pred_ols_analytical,
        y_pred_ols_gd,
        y_pred_ridge_analytical,
        y_pred_ridge_gd,
        x_plotting
    ) = solutions

    # --- OLS ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    plt.scatter(x, y_noise, label="y_noise")
    if test:
        plt.scatter(x_plotting, y_pred_ols_analytical, label="OLS (Analytical)")
        plt.scatter(x_plotting, y_pred_ols_gd, label="OLS (GD)")
    else: 
        plt.plot(x_plotting, y_pred_ols_analytical, label="OLS (Analytical)")
        plt.plot(x_plotting, y_pred_ols_gd, label="OLS (GD)")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend()
    plt.show()

    # --- Ridge ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    plt.scatter(x, y_noise, label="y_noise")
    if test:
        plt.scatter(x_plotting, y_pred_ridge_analytical, label=f"Ridge (Analytical, λ={lam:.0e})")
        plt.scatter(x_plotting, y_pred_ridge_gd, label=f"Ridge (GD, λ={lam:.0e})")
    else:
        plt.plot(x_plotting, y_pred_ridge_analytical, label=f"Ridge (Analytical, λ={lam:.0e})")
        plt.plot(x_plotting, y_pred_ridge_gd, label=f"Ridge (GD, λ={lam:.0e})")
    plt.xlabel("x", fontsize=16)
    plt.ylabel(f"y(x), degree={degree}, N={sample_size}", fontsize=16)
    setup_plot_formatting()
    plt.legend(fontsize=16)
    plt.show()




def solution_comparison_gd(x, y_noise, y_true, solutions, sample_size, degree, lam, test=False):
    """
    Plotting solutions using different methods for computing the optimal parameters (gradient descent).
    """

    (
        y_analytical, 
        y_gd,
        y_momentum,
        y_adagrad,
        y_rmsprop,
        y_adam,
        x_plotting
    ) = solutions

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Runge function")
    plt.scatter(x, y_noise, label="y_noise")
    if test:
        plt.scatter(x_plotting, y_analytical, label="Analytical")
        plt.scatter(x_plotting, y_gd, label="Gradient Descent")
        plt.scatter(x_plotting, y_momentum, label="Momentum")
        plt.scatter(x_plotting, y_adagrad, label="AdaGrad")
        plt.scatter(x_plotting, y_rmsprop, label="RMSProp")
        plt.scatter(x_plotting, y_adam, label="Adam")
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
    plt.legend(fontsize=16)
    # plt.savefig()
    plt.show()


def optimizer_comparison():
    pass

def method_comparison():
    pass