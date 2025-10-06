import numpy as np
import concurrent.futures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

from src.utils import polynomial_features, scale_data, OLS_parameters, runge
from src.plotting import (
    plot_diagonal_comparison,
    plot_bias_variance_decomposition,
    plot_heatmaps,
    plot_cv_vs_bootstrap_comparison,
    plot_cv_heatmaps_multi_noise,
    plot_delta_heatmap_multi_noise,
    plot_delta_heatmap_zoom
)
import pandas as pd
import time
import os
import pickle
from datetime import datetime
import time

#######################
# Bootstrap resampling
#######################


def process_single_sample_size(args):
    """
    Process one sample size for bootstrap analysis.
    """
    (
        n,
        maxdegree,
        n_bootstraps,
        noise_std,
        test_size,
        random_state,
        save_fits,
        n_samples_to_save,
    ) = args

    print(f"Processing n={n}...")

    np.random.seed(random_state)

    # Initialize results for this sample size
    error_array = np.zeros(maxdegree)
    bias_squared_array = np.zeros(maxdegree)
    variance_array = np.zeros(maxdegree)
    expected_error_array = np.zeros(maxdegree)

    # Generate data
    x = np.linspace(-1, 1, n)
    epsilon = np.random.normal(0, noise_std, n)
    y_true = runge(x)
    y = y_true + epsilon

    # Split into train and test
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    n_test = len(test_idx)

    # Store fit data if requested
    saved_fits = {}

    # Loop over polynomial degrees
    for degree in range(1, maxdegree + 1):
        X_test = polynomial_features(x_test, degree)

        # Store predictions from each bootstrap sample
        y_pred_bootstraps = np.zeros((n_test, n_bootstraps))

        # Storage for individual bootstrap predictions (for plotting)
        bootstrap_predictions_plot = (
            [] if (save_fits is not None and degree in save_fits) else None
        )

        # Bootstrap loop
        for b in range(n_bootstraps):
            # Resample training data
            boot_indices = np.random.choice(
                len(x_train), size=len(x_train), replace=True
            )
            x_boot = x_train[boot_indices]
            y_boot = y_train[boot_indices]

            # Create polynomial features
            X_boot = polynomial_features(x_boot, degree)

            # Scale data
            X_boot_scaled, y_boot_scaled, X_mean, X_std, y_mean = scale_data(
                X_boot, y_boot
            )

            # Fit OLS
            theta = OLS_parameters(X_boot_scaled, y_boot_scaled)

            # Scale test data and predict
            X_test_scaled, _, _, _, _ = scale_data(
                X_test, y_test, X_mean, X_std, y_mean
            )
            y_pred_test = X_test_scaled @ theta + y_mean
            y_pred_bootstraps[:, b] = y_pred_test

            # Save individual bootstrap predictions for plotting (only first few)
            if bootstrap_predictions_plot is not None and b < n_samples_to_save:
                # Predict on fine grid for smooth curves
                x_plot = np.linspace(-1, 1, 1000)
                X_plot = polynomial_features(x_plot, degree)
                X_plot_scaled, _, _, _, _ = scale_data(
                    X_plot, np.zeros(len(x_plot)), X_mean, X_std, y_mean
                )
                y_pred_plot = X_plot_scaled @ theta + y_mean
                bootstrap_predictions_plot.append(y_pred_plot)

        # Calculate bias-variance decomposition
        y_pred_mean = np.mean(y_pred_bootstraps, axis=1)

        error_array[degree - 1] = np.mean(
            np.mean((y_test.reshape(-1, 1) - y_pred_bootstraps) ** 2, axis=1)
        )
        bias_squared_array[degree - 1] = np.mean((y_test - y_pred_mean) ** 2)
        variance_array[degree - 1] = np.mean(np.var(y_pred_bootstraps, axis=1))
        expected_error_array[degree - 1] = (
            bias_squared_array[degree - 1] + variance_array[degree - 1]
        )

        # Save fit data for specific degrees if requested
        if save_fits is not None and degree in save_fits:
            saved_fits[degree] = {
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "x_full": x,
                "y_true_full": y_true,
                "x_plot": np.linspace(-1, 1, 1000),
                "bootstrap_predictions": bootstrap_predictions_plot,
                "bias_squared": bias_squared_array[degree - 1],
                "variance": variance_array[degree - 1],
                "error": error_array[degree - 1],
            }
    print(f"Completed n={n}")

    return (
        error_array,
        bias_squared_array,
        variance_array,
        expected_error_array,
        saved_fits,
    )


def bootstrap_bias_variance_analysis(
    sample_sizes,
    maxdegree=15,
    n_bootstraps=500,
    noise_std=0.1,
    test_size=0.25,
    random_state=42,
    parallel=True,
    max_workers=4,
    save_diagonal_fits=None,
    n_bootstrap_samples_to_plot=5,
):
    """
    Bootstrap bias-variance analysis with parallelization over sample sizes.

    Parameters
    ----------
    sample_sizes : list
        List of sample sizes to analyze (each runs on separate core)
    maxdegree : int
        Maximum polynomial degree
    n_bootstraps : int
        Number of bootstrap samples
    noise_std : float
        Standard deviation of noise
    test_size : float
        Fraction for test set (evaluates on this held-out set)
    random_state : int
        Random seed
    parallel : bool
        If True, process different sample sizes in parallel
    max_workers : int
        Number of CPU cores to use (default 4)
    save_diagonal_fits : list of tuples, optional
        List of (n, degree) pairs to save fit data for plotting
        Example: [(200, 5), (350, 10), (500, 15)]
    n_bootstrap_samples_to_plot : int
        Number of individual bootstrap predictions to save for visualization
    """

    # Initialize matrices
    n_samples = len(sample_sizes)
    error_matrix = np.zeros((n_samples, maxdegree))
    bias_squared_matrix = np.zeros((n_samples, maxdegree))
    variance_matrix = np.zeros((n_samples, maxdegree))
    expected_error_matrix = np.zeros((n_samples, maxdegree))

    noise_var = noise_std**2

    # Track which degrees to save for each sample size
    save_dict = {}
    if save_diagonal_fits is not None:
        for n, deg in save_diagonal_fits:
            if n in sample_sizes:
                if n not in save_dict:
                    save_dict[n] = []
                save_dict[n].append(deg)

    # Storage for saved fits
    diagonal_fits = {}

    if parallel:
        print(f"Running calculations in parallel with {max_workers} cores...")
        print(f"Processing {len(sample_sizes)} sample sizes: {sample_sizes}")

        # Prepare arguments for each sample size
        args_list = [
            (
                n,
                maxdegree,
                n_bootstraps,
                noise_std,
                test_size,
                random_state,
                save_dict.get(n),
                n_bootstrap_samples_to_plot,
            )
            for n in sample_sizes
        ]

        # Process each sample size on a separate core
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            results = list(executor.map(process_single_sample_size, args_list))

        # Unpack results into matrices
        for i, result in enumerate(results):
            error_matrix[i, :] = result[0]
            bias_squared_matrix[i, :] = result[1]
            variance_matrix[i, :] = result[2]
            expected_error_matrix[i, :] = result[3]

            # Store saved fits
            n = sample_sizes[i]
            if result[4]:  # If there are saved fits
                diagonal_fits[n] = result[4]

    else:
        print("Running calculations serially...")
        for i, n in enumerate(sample_sizes):
            args = (
                n,
                maxdegree,
                n_bootstraps,
                noise_std,
                test_size,
                random_state,
                save_dict.get(n),
                n_bootstrap_samples_to_plot,
            )
            result = process_single_sample_size(args)
            error_matrix[i, :] = result[0]
            bias_squared_matrix[i, :] = result[1]
            variance_matrix[i, :] = result[2]
            expected_error_matrix[i, :] = result[3]

            # Store saved fits
            if result[4]:
                diagonal_fits[n] = result[4]

    print("Bootstrap analysis complete!")

    # Best model complexity
    best_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    best_n = sample_sizes[best_idx[0]]
    best_degree = best_idx[1] + 1  # +1 because degrees start at 1
    print(
        f"Best model complexity: n={best_n}, degree={best_degree} with MSE={error_matrix[best_idx]:.4f}"
    )

    return {
        "error_matrix": error_matrix,
        "bias_squared_matrix": bias_squared_matrix,
        "variance_matrix": variance_matrix,
        "expected_error_matrix": expected_error_matrix,
        "sample_sizes": sample_sizes,
        "degrees": np.arange(1, maxdegree + 1),
        "noise_var": noise_var,
        "diagonal_fits": diagonal_fits,
    }


######################
# Cross validation
#####################


def process_single_sample_size_cv(args):
    """
    Process cross-validation for one sample size.
    Uses k-fold CV on the FULL dataset.
    """
    (n, maxdegree, k_folds, lambda_values, noise_std, random_state) = args

    print(f"Processing CV for n={n}...")

    np.random.seed(random_state)

    # Initialize results for this sample size
    ols_results = {"mse": np.zeros(maxdegree), "std": np.zeros(maxdegree)}
    ridge_results = {
        lam: {"mse": np.zeros(maxdegree), "std": np.zeros(maxdegree)}
        for lam in lambda_values
    }
    lasso_results = {
        lam: {"mse": np.zeros(maxdegree), "std": np.zeros(maxdegree)}
        for lam in lambda_values
    }

    # Generate data
    x = np.linspace(-1, 1, n)
    epsilon = np.random.normal(0, noise_std, n)
    y_true = runge(x)
    y = y_true + epsilon

    # K-fold CV on the FULL dataset
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    for degree in range(1, maxdegree + 1):
        # Precompute polynomial features
        X_full = polynomial_features(x, degree)

        # Storage for fold MSEs (CV validation errors)
        ols_fold_mse = []
        ridge_fold_mse = {lam: [] for lam in lambda_values}
        lasso_fold_mse = {lam: [] for lam in lambda_values}

        #  K-fold CV loop
        for train_fold_idx, val_fold_idx in kf.split(X_full):
            X_train = X_full[train_fold_idx]
            X_val = X_full[val_fold_idx]
            y_train = y[train_fold_idx]
            y_val = y[val_fold_idx]

            # Scale
            X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
            X_val_s, y_val_s, _, _, _ = scale_data(X_val, y_val, X_mean, X_std, y_mean)
            y_val_unscaled = y_val_s + y_mean

            # OLS (CV validation error)
            theta_ols = OLS_parameters(X_train_s, y_train_s)
            y_pred_ols = X_val_s @ theta_ols + y_mean
            ols_fold_mse.append(mean_squared_error(y_val_unscaled, y_pred_ols))

            # Ridge (CV validation error per lambda)
            for lam in lambda_values:
                ridge_model = Ridge(alpha=lam, fit_intercept=False, solver="svd")
                ridge_model.fit(X_train_s, y_train_s)
                y_pred_ridge = ridge_model.predict(X_val_s) + y_mean
                ridge_fold_mse[lam].append(
                    mean_squared_error(y_val_unscaled, y_pred_ridge)
                )

            # Lasso (CV validation error per lambda) with warm-start
            n_fold = X_train_s.shape[0]
            alpha_max = np.max(np.abs(X_train_s.T @ y_train_s)) / n_fold
            n_path = max(10, len(lambda_values))
            lams_desc = np.geomspace(alpha_max, alpha_max * 1e-4, num=n_path)

            lasso_model = Lasso(
                alpha=lams_desc[0],
                fit_intercept=False,
                max_iter=100_000,
                tol=1e-3,
                warm_start=True,
                selection="cyclic",
                copy_X=False,
            )

            best_val = np.inf
            patience = 2
            bad = 0

            # mapping from my lambda grid to the closest point on the path
            idx_map = {
                lam: int(np.argmin(np.abs(lams_desc - lam))) for lam in lambda_values
            }

            for j, a in enumerate(lams_desc):
                lasso_model.set_params(alpha=a)
                lasso_model.fit(X_train_s, y_train_s)
                y_pred_lasso = lasso_model.predict(X_val_s) + y_mean
                mse_val = mean_squared_error(y_val_unscaled, y_pred_lasso)

                # simple early stopping on validation
                if mse_val + 1e-6 < best_val:
                    best_val = mse_val
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

                # record CV MSE for lambda on my original grid that map here
                for lam, j_closest in idx_map.items():
                    if j_closest == j:
                        lasso_fold_mse[lam].append(mse_val)

            # Fill any missing lambda entries for this fold with the best observed val-MSE
            for lam in lambda_values:
                if len(lasso_fold_mse[lam]) < len(ols_fold_mse):
                    lasso_fold_mse[lam].append(best_val)

        # Aggregate CV across folds
        ols_results["mse"][degree - 1] = np.mean(ols_fold_mse)
        ols_results["std"][degree - 1] = np.std(ols_fold_mse)

        for lam in lambda_values:
            ridge_results[lam]["mse"][degree - 1] = np.mean(ridge_fold_mse[lam])
            ridge_results[lam]["std"][degree - 1] = np.std(ridge_fold_mse[lam])

            lasso_results[lam]["mse"][degree - 1] = np.mean(lasso_fold_mse[lam])
            lasso_results[lam]["std"][degree - 1] = np.std(lasso_fold_mse[lam])

    print(f"Completed CV for n={n}")
    return (ols_results, ridge_results, lasso_results)


def cross_validation_analysis(
    sample_sizes,
    maxdegree=15,
    k_folds=10,
    lambda_values=np.logspace(-4, 2, 20),
    noise_std=0.1,
    random_state=42,
    parallel=True,
    max_workers=8,
):
    """
    Perform k-fold cross-validation for OLS, Ridge, and Lasso regression.
    Uses full dataset for k-fold CV (no separate test set).

    Parameters
    ----------
    sample_sizes : list
        List of sample sizes to analyze
    maxdegree : int
        Maximum polynomial degree
    k_folds : int
        Number of folds for cross-validation (used for stability estimation)
    lambda_values : array
        Array of regularization parameters for Ridge and Lasso
    noise_std : float
        Standard deviation of noise
    random_state : int
        Random seed
    parallel : bool
        If True, process different sample sizes in parallel
    max_workers : int
        Number of CPU cores to use

    Returns
    -------
    dict
        Dictionary containing test MSE results for OLS, Ridge, and Lasso
    """
    np.random.seed(random_state)

    # Initialize result storage
    results = {
        "ols": {
            "mse_matrix": np.zeros((len(sample_sizes), maxdegree)),
            "mse_std": np.zeros((len(sample_sizes), maxdegree)),
        },
        "ridge": {},
        "lasso": {},
    }

    # For Ridge and Lasso, store results for each lambda
    for lam in lambda_values:
        results["ridge"][lam] = {
            "mse_matrix": np.zeros((len(sample_sizes), maxdegree)),
            "mse_std": np.zeros((len(sample_sizes), maxdegree)),
        }
        results["lasso"][lam] = {
            "mse_matrix": np.zeros((len(sample_sizes), maxdegree)),
            "mse_std": np.zeros((len(sample_sizes), maxdegree)),
        }

    if parallel:
        print(f"Running {k_folds}-fold CV in parallel with {max_workers} cores...")
        print(f"Processing {len(sample_sizes)} sample sizes: {sample_sizes}")

        # Prepare arguments for each sample size
        args_list = [
            (n, maxdegree, k_folds, lambda_values, noise_std, random_state)
            for n in sample_sizes
        ]

        # Process each sample size on a separate core
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            cv_results = list(executor.map(process_single_sample_size_cv, args_list))

        # Unpack results into matrices
        for i, (ols_res, ridge_res, lasso_res) in enumerate(cv_results):
            results["ols"]["mse_matrix"][i, :] = ols_res["mse"]
            results["ols"]["mse_std"][i, :] = ols_res["std"]

            for lam in lambda_values:
                results["ridge"][lam]["mse_matrix"][i, :] = ridge_res[lam]["mse"]
                results["ridge"][lam]["mse_std"][i, :] = ridge_res[lam]["std"]
                results["lasso"][lam]["mse_matrix"][i, :] = lasso_res[lam]["mse"]
                results["lasso"][lam]["mse_std"][i, :] = lasso_res[lam]["std"]

    else:
        print(f"Running {k_folds}-fold CV serially...")
        for i, n in enumerate(sample_sizes):
            args = (n, maxdegree, k_folds, lambda_values, noise_std, random_state)
            ols_res, ridge_res, lasso_res = process_single_sample_size_cv(args)

            results["ols"]["mse_matrix"][i, :] = ols_res["mse"]
            results["ols"]["mse_std"][i, :] = ols_res["std"]

            for lam in lambda_values:
                results["ridge"][lam]["mse_matrix"][i, :] = ridge_res[lam]["mse"]
                results["ridge"][lam]["mse_std"][i, :] = ridge_res[lam]["std"]
                results["lasso"][lam]["mse_matrix"][i, :] = lasso_res[lam]["mse"]
                results["lasso"][lam]["mse_std"][i, :] = lasso_res[lam]["std"]

    results["sample_sizes"] = sample_sizes
    results["degrees"] = np.arange(1, maxdegree + 1)
    results["lambda_values"] = lambda_values
    results["k_folds"] = k_folds

    print(f"{k_folds}-fold cross-validation complete!")
    return results


################################################################################
# MAIN EXECUTION
################################################################################

if __name__ == "__main__":

    # ===========================
    # GLOBAL PARAMETERS
    # ===========================
    random_state = 42
    sample_sizes = [
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
        400,
        425,
        450,
        475,
        500,
        525,
        550,
        575,
    ]
    maxdegree = 20
    noise_std = 0.1
    test_size = 0.25
    parallel = True
    max_workers = 10

    # ===========================
    # PART G: BOOTSTRAP ANALYSIS
    # ===========================
    print("=" * 70)
    print("PART G: BOOTSTRAP BIAS-VARIANCE ANALYSIS")
    print("=" * 70)

    n_bootstraps = 2000
    n_bootstrap_samples_to_plot = 300
    diagonal_points = [
        (min(sample_sizes), 2),
        (min(sample_sizes), maxdegree),
        (max(sample_sizes), 2),
        (max(sample_sizes), maxdegree),
    ]

    start = time.time()
    bootstrap_results = bootstrap_bias_variance_analysis(
        sample_sizes=sample_sizes,
        maxdegree=maxdegree,
        n_bootstraps=n_bootstraps,
        noise_std=noise_std,
        test_size=test_size,
        random_state=random_state,
        parallel=parallel,
        max_workers=max_workers,
        save_diagonal_fits=diagonal_points,
        n_bootstrap_samples_to_plot=n_bootstrap_samples_to_plot,
    )
    bootstrap_time = time.time() - start
    print(f"Bootstrap computation time: {bootstrap_time:.2f} seconds\n")

    # Generate bootstrap plots
    plot_bias_variance_decomposition(bootstrap_results, sample_size_idx=0)
    plot_heatmaps(bootstrap_results)
    plot_diagonal_comparison(
        bootstrap_results, diagonal_points, n_bootstraps=n_bootstraps
    )
    

    # ===================================
    # PART H: CROSS-VALIDATION ANALYSIS
    # ===================================
    print("\n" + "=" * 70)
    print("PART H: CROSS-VALIDATION ANALYSIS")
    print("=" * 70)

    k_folds = 5
    lambda_values = np.logspace(-5, 2, 20)  # lambda grid for Ridge and Lasso

    start = time.time()
    cv_results = cross_validation_analysis(
        sample_sizes=sample_sizes,
        maxdegree=maxdegree,
        k_folds=k_folds,
        lambda_values=lambda_values,
        noise_std=noise_std,
        random_state=random_state,
        parallel=parallel,
        max_workers=max_workers,
    )
    cv_time = time.time() - start
    print(f"Cross-validation computation time: {cv_time:.2f} seconds\n")

    # Generate CV comparison plot
    plot_cv_vs_bootstrap_comparison(cv_results, bootstrap_results, sample_size_idx=0)

    # ===================================
    # MULTI-NOISE ANALYSIS
    # ===================================
    print("\n" + "=" * 70)
    print("MULTI-NOISE CROSS-VALIDATION ANALYSIS")
    print("=" * 70)

    # Create results directory
    os.makedirs("../cv_results", exist_ok=True)

    k_folds = 5
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    noise_configs = {
        "low": {"noise_std": 0.1, "description": "Low noise (σ=0.1)"},
        "medium": {"noise_std": 0.2, "description": "Medium noise (σ=0.2)"},
        "high": {"noise_std": 0.3, "description": "High noise (σ=0.3)"},
    }

    print(
        f"Configuration: {len(sample_sizes)} sample sizes, degree {maxdegree}, {k_folds}-fold CV"
    )
    print(f"Seeds: {seeds}, Lambda grid: {len(lambda_values)} values\n")

    total_start = time.time()

    # ################# Big computation block commented out to avoid long runtimes ##################
    # # # Run CV for all noise levels and seeds

    # for noise_level, config in noise_configs.items():
    #     print(f"{noise_level.upper()}: {config['description']}")

    #     for seed in seeds:
    #         start = time.time()
    #         cv_results = cross_validation_analysis(
    #             sample_sizes=sample_sizes,
    #             maxdegree=maxdegree,
    #             k_folds=k_folds,
    #             lambda_values=lambda_values,
    #             noise_std=config["noise_std"],
    #             random_state=seed,
    #             parallel=parallel,
    #             max_workers=max_workers,
    #         )

    #         # Save results
    #         seed_dir = f"../cv_results/noise_{noise_level}/seed_{seed}"
    #         os.makedirs(seed_dir, exist_ok=True)
    #         filename = f"{seed_dir}/cv_{noise_level}_seed{seed}_noise_std{config['noise_std']:.1f}.pkl"

    #         with open(filename, "wb") as f:
    #             pickle.dump(cv_results, f)

    #         print(f"  Seed {seed}: {time.time() - start:.1f}s")
    #     print()

    # ===================================
    # AGGREGATE AND ANALYZE RESULTS
    # ===================================
    total_elapsed = time.time() - total_start
    print(f"Total computation: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n")

    print("=" * 70)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("=" * 70 + "\n")

    def load_cv_results_for_noise(noise_level, seeds):
        """Load per-seed CV results for a given noise level."""
        noise_std_map = {"low": 0.1, "medium": 0.2, "high": 0.3}
        std = noise_std_map[noise_level]
        results_list = []
        for seed in seeds:
            filename = f"../cv_results/noise_{noise_level}/seed_{seed}/cv_{noise_level}_seed{seed}_noise_std{std:.1f}.pkl"
            with open(filename, "rb") as f:
                results_list.append(pickle.load(f))
        return results_list

    def aggregate_cv_results_across_seeds(cv_list):
        """Average MSE matrices across seeds."""
        base = cv_list[0]
        lambdas = list(base["lambda_values"])

        # Aggregate OLS
        ols_mse_mean = np.stack(
            [cv["ols"]["mse_matrix"] for cv in cv_list], axis=0
        ).mean(axis=0)
        ols_std_mean = np.stack([cv["ols"]["mse_std"] for cv in cv_list], axis=0).mean(
            axis=0
        )

        # Aggregate Ridge & Lasso per λ
        ridge_agg = {
            lam: {
                "mse_matrix": np.stack(
                    [cv["ridge"][lam]["mse_matrix"] for cv in cv_list], axis=0
                ).mean(axis=0),
                "mse_std": np.stack(
                    [cv["ridge"][lam]["mse_std"] for cv in cv_list], axis=0
                ).mean(axis=0),
            }
            for lam in lambdas
        }
        lasso_agg = {
            lam: {
                "mse_matrix": np.stack(
                    [cv["lasso"][lam]["mse_matrix"] for cv in cv_list], axis=0
                ).mean(axis=0),
                "mse_std": np.stack(
                    [cv["lasso"][lam]["mse_std"] for cv in cv_list], axis=0
                ).mean(axis=0),
            }
            for lam in lambdas
        }

        return {
            "ols": {"mse_matrix": ols_mse_mean, "mse_std": ols_std_mean},
            "ridge": ridge_agg,
            "lasso": lasso_agg,
            "sample_sizes": base["sample_sizes"],
            "degrees": base["degrees"],
            "lambda_values": base["lambda_values"],
            "k_folds": base["k_folds"],
        }

    def mean_relative_delta(cv_list, model):
        """Compute mean relative improvement: (OLS - Model_best) / OLS across seeds."""
        lambdas = np.asarray(cv_list[0]["lambda_values"])
        rel_deltas = []

        for cv in cv_list:
            ols_mse = cv["ols"]["mse_matrix"]
            model_cube = np.stack(
                [cv[model][lam]["mse_matrix"] for lam in lambdas], axis=2
            )
            best_model_mse = model_cube.min(axis=2)

            delta_rel = (ols_mse - best_model_mse) / np.maximum(ols_mse, 1e-12)
            rel_deltas.append(delta_rel)

        return np.stack(rel_deltas, axis=0).mean(axis=0)

    # Load and aggregate results
    cv_low_list = load_cv_results_for_noise("low", seeds)
    cv_med_list = load_cv_results_for_noise("medium", seeds)
    cv_high_list = load_cv_results_for_noise("high", seeds)

    cv_low = aggregate_cv_results_across_seeds(cv_low_list)
    cv_med = aggregate_cv_results_across_seeds(cv_med_list)
    cv_high = aggregate_cv_results_across_seeds(cv_high_list)

    # Compute relative improvements
    noise_results_for_plot = {}
    summary_stats = []

    for noise_level, cv_list, cv_agg, sigma in [
        ("low", cv_low_list, cv_low, 0.1),
        ("medium", cv_med_list, cv_med, 0.2),
        ("high", cv_high_list, cv_high, 0.3),
    ]:
        ridge_delta = mean_relative_delta(cv_list, "ridge")
        lasso_delta = mean_relative_delta(cv_list, "lasso")

        noise_results_for_plot[noise_level] = {
            "delta_ridge": ridge_delta,
            "delta_lasso": lasso_delta,
            "cv_results": cv_agg,
            "σ": sigma,
        }

        summary_stats.append(
            {
                "noise_level": noise_level,
                "sigma": sigma,
                "ridge_improvements": int((ridge_delta > 0.01).sum()),
                "ridge_max": ridge_delta.max(),
                "ridge_mean": ridge_delta.mean(),
                "lasso_improvements": int((lasso_delta > 0.01).sum()),
                "lasso_max": lasso_delta.max(),
                "lasso_mean": lasso_delta.mean(),
            }
        )

    # Generate plots
    print("Generating visualizations...\n")
    sample_sizes_arr = np.asarray(cv_low["sample_sizes"])
    degrees_arr = np.asarray(cv_low["degrees"])

    plot_delta_heatmap_multi_noise(
        noise_results_for_plot,
        sample_sizes_arr,
        degrees_arr,
        filename="figs/delta_heatmap_multi_noise_combined.pdf",
    )

    cv_dict_for_plot = {
        "low": {"cv_results": cv_low},
        "medium": {"cv_results": cv_med},
        "high": {"cv_results": cv_high},
    }

    plot_cv_heatmaps_multi_noise(
        cv_dict_for_plot,
        sample_sizes_arr,
        degrees_arr,
        k_folds=cv_low["k_folds"],
        filename="figs/cv_mse_multi_noise_combined.pdf",
    )

    # ===================================
    # FINAL SUMMARY
    # ===================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    # Bootstrap results
    print("\n" + "-" * 70)
    print("BOOTSTRAP ANALYSIS (OLS only)")
    print("-" * 70)
    best_idx = np.unravel_index(
        np.argmin(bootstrap_results["error_matrix"]),
        bootstrap_results["error_matrix"].shape,
    )
    best_n = bootstrap_results["sample_sizes"][best_idx[0]]
    best_degree = best_idx[1] + 1
    best_mse = bootstrap_results["error_matrix"][best_idx]
    best_bias2 = bootstrap_results["bias_squared_matrix"][best_idx]
    best_var = bootstrap_results["variance_matrix"][best_idx]

    print(f"Optimal model: n={best_n}, degree={best_degree}")
    print(f"  Test MSE:      {best_mse:.6f}")
    print(f"  Bias²:         {best_bias2:.6f}")
    print(f"  Variance:      {best_var:.6f}")
    print(f"  Bias² + Var:   {best_bias2 + best_var:.6f}")
    print(f"  Noise (σ²):    {bootstrap_results['noise_var']:.6f}")

    # Multi-noise CV results
    print("\n" + "-" * 70)
    print("MULTI-NOISE CROSS-VALIDATION ANALYSIS")
    print("-" * 70)
    print(
        f"{'Noise':>8} | {'σ':>5} | {'Ridge >1%':>10} | {'Max Δ%':>8} | {'Lasso >1%':>10} | {'Max Δ%':>8}"
    )
    print("-" * 70)

    total_cells = summary_stats[0]["ridge_improvements"] + (
        360 - summary_stats[0]["ridge_improvements"]
    )
    for stat in summary_stats:
        print(
            f"{stat['noise_level']:>8} | {stat['sigma']:>5.1f} | "
            f"{stat['ridge_improvements']:>4}/{total_cells:>3} | {100*stat['ridge_max']:>7.1f}% | "
            f"{stat['lasso_improvements']:>4}/{total_cells:>3} | {100*stat['lasso_max']:>7.1f}%"
        )

    print(f"\nComputation times:")
    print(
        f"  Bootstrap: {bootstrap_time:.1f}s, CV: {cv_time:.1f}s, Multi-noise: {total_elapsed:.1f}s"
    )
    print(f"\nPlots saved to figs/ directory")
    print("=" * 70)

    
    
    
    # Example zoom plot: Low noise, OLS vs LASSO
    # print("Low noise, N = all, degree = [7, 8, 9, 10, 11, 12, 13]")
    # plot_delta_heatmap_zoom(
    #     noise_results_for_plot,
    #     sample_sizes_arr,
    #     degrees_arr,
    #     noise_level="low",
    #     comparison="lasso",
    #     zoom_sample_sizes=None, 
    #     zoom_degrees=[7, 8, 9, 10, 11, 12, 13], 
    #     filename="figs/zoom_low_lasso.pdf"
    # )