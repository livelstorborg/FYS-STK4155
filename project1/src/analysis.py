from sklearn.model_selection import train_test_split
import pandas as pd

from .utils import polynomial_features, scale_data
from .regression import RegressionAnalysis


def analyze_mse_vs_degree(x, y, degrees, test_size=0.2, random_state=42, **kwargs):
    """
    Analyze MSE vs polynomial degree for both training and test sets.

    Parameters
    ----------
    x : np.ndarray
        Input x values
    y : np.ndarray
        True y values (Runge function)
    degrees : list or range
        Polynomial degrees to analyze
    test_size : float, optional
        Fraction of data to use for testing (default: 0.2)
    random_state : int, optional
        Random state for train/test split (default: 42)
    **kwargs : dict
        Additional parameters for RegressionAnalysis

    Returns
    -------
    results : dict
        Dictionary containing MSE results and analysis instances
    """

    # Storage for results
    train_mse_list = []
    test_mse_list = []
    analysis_instances = []

    print("Analyzing MSE vs Polynomial Degree...")
    print("-" * 50)

    for degree in degrees:
        print(f"Processing degree {degree}...")

        # Create data splits for this degree
        X = polynomial_features(x, degree)
        X_norm, y_centered, y_mean = scale_data(X, y)
        X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(
            X_norm, y_centered, x, test_size=test_size, random_state=random_state
        )

        data = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]

        # Create analysis instance for this degree
        analysis = RegressionAnalysis(data, degree=degree, **kwargs)

        # Fit analytical OLS solution
        analysis.fit_analytical()
        analysis.predict()
        analysis.calculate_metrics()

        # Get training MSE
        train_mse = analysis.get_train_mse(method="ols_analytical")

        # Store results
        train_mse_list.append(train_mse)
        test_mse_list.append(analysis.mse_ols_analytical)
        analysis_instances.append(analysis)

        print(
            f"  Train MSE: {train_mse:.6f}, Test MSE: {analysis.mse_ols_analytical:.6f}"
        )

    return {
        "degrees": list(degrees),
        "train_mse": train_mse_list,
        "test_mse": test_mse_list,
        "instances": analysis_instances,
    }


def compare_methods(analysis_instance, methods=["ols_analytical", "ridge_analytical"]):
    """
    Compare different regression methods using a single analysis instance.

    Parameters
    ----------
    analysis_instance : RegressionAnalysis
        Analysis instance with fitted models
    methods : list
        List of methods to compare

    Returns
    -------
    dict
        Comparison results
    """

    results = {}

    for method in methods:
        train_mse = analysis_instance.get_train_mse(method)

        # Get test MSE
        if method == "ols_analytical":
            test_mse = analysis_instance.mse_ols_analytical
            r2 = analysis_instance.r2_ols_analytical
        elif method == "ridge_analytical":
            test_mse = analysis_instance.mse_ridge_analytical
            r2 = analysis_instance.r2_ridge_analytical
        elif method == "ols_gd":
            test_mse = analysis_instance.mse_ols_gd
            r2 = analysis_instance.r2_ols_gd
        elif method == "ridge_gd":
            test_mse = analysis_instance.mse_ridge_gd
            r2 = analysis_instance.r2_ridge_gd

        results[method] = {"train_mse": train_mse, "test_mse": test_mse, "r2": r2}

    return results


def analyze_dependence_datapoints():
    pass


def analyze_dependence_lambda_datapoints(
    all_results, lambda_values, sample_sizes, degree
):
    """
    Create a DataFrame showing MSE for Ridge regression across different lambda values and sample sizes.

    Parameters
    ----------
    all_results : dict
        Dictionary with structure: all_results[sample_size][lambda_value]['train_mse'] or ['test_mse']
    lambda_values : array
        Array of lambda values used
    sample_sizes : list
        List of sample sizes used
    degree : int
        Polynomial degree to analyze (1-based indexing, so degree=1 means first degree)

    Returns
    -------
    pd.DataFrame
        DataFrame with lambda values as rows and sample sizes as columns
    """

    # Convert degree to 0-based index for accessing results
    degree_idx = degree - 1

    # Create empty DataFrame
    mse_data = {}

    for N in sample_sizes:
        mse_values = []
        for lam in lambda_values:
            # Get MSE for this lambda and sample size at the specified degree
            mse = all_results[N][lam]["test_mse"][degree_idx]  # Use test_mse
            mse_values.append(mse)

        mse_data[f"N={N}"] = mse_values

    df = pd.DataFrame(mse_data, index=[f"{lam:.1e}" for lam in lambda_values])
    df.index.name = "Lambda"

    print(f"\nPolynomial Degree {degree}")
    print("-" * 60)
    print(df.to_string(float_format="%.6f"))

    print(f"\nMinimum MSE for degree {degree}:")
    print("-" * 40)
    for col in df.columns:
        min_idx = df[col].idxmin()
        min_val = df[col].min()
        print(f"{col}: λ={min_idx}, MSE={min_val:.6f}")

    return df
