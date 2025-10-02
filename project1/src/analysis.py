from sklearn.model_selection import train_test_split
import pandas as pd

from .utils import polynomial_features, scale_data
from .regression import RegressionAnalysis











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
