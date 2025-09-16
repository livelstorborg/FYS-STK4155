import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

from .utils import polynomial_features, scale_data
from .regression import RegressionAnalysis



def analyze_mse_vs_degree(x, y, degrees, **kwargs):
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
    **kwargs : dict
        Additional parameters passed to create_analysis_instance()
    
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
        
        # Create analysis instance for this degree
        analysis = create_analysis_instance(x, y, degree, **kwargs)
        
        # Fit analytical OLS solution
        analysis.fit_analytical()
        analysis.predict()
        analysis.calculate_metrics()
        
        # Get training MSE
        train_mse = get_train_mse(analysis, method='ols_analytical')
        
        # Store results
        train_mse_list.append(train_mse)
        test_mse_list.append(analysis.mse_ols_analytical)
        analysis_instances.append(analysis)
        
        print(f"  Train MSE: {train_mse:.6f}, Test MSE: {analysis.mse_ols_analytical:.6f}")
    
    return {
        'degrees': list(degrees),
        'train_mse': train_mse_list,
        'test_mse': test_mse_list,
        'instances': analysis_instances
    }

def get_train_mse(analysis_instance, method='ols_analytical'):
    """
    Calculate training MSE for a specific method from an analysis instance.
    
    Parameters
    ----------
    analysis_instance : RegressionAnalysis
        Fitted analysis instance
    method : str
        Method to calculate MSE for ('ols_analytical', 'ridge_analytical', 
        'ols_gd', 'ridge_gd')
    
    Returns
    -------
    float
        Training MSE
    """
    
    if method == 'ols_analytical':
        if analysis_instance.theta_ols_analytical is None:
            raise ValueError("OLS analytical solution not fitted")
        y_train_pred = analysis_instance.X_train @ analysis_instance.theta_ols_analytical + analysis_instance.y_mean
        
    elif method == 'ridge_analytical':
        if analysis_instance.theta_ridge_analytical is None:
            raise ValueError("Ridge analytical solution not fitted")
        y_train_pred = analysis_instance.X_train @ analysis_instance.theta_ridge_analytical + analysis_instance.y_mean
        
    elif method == 'ols_gd':
        if analysis_instance.theta_ols_gd is None:
            raise ValueError("OLS gradient descent solution not fitted")
        y_train_pred = analysis_instance.X_train @ analysis_instance.theta_ols_gd + analysis_instance.y_mean
        
    elif method == 'ridge_gd':
        if analysis_instance.theta_ridge_gd is None:
            raise ValueError("Ridge gradient descent solution not fitted")
        y_train_pred = analysis_instance.X_train @ analysis_instance.theta_ridge_gd + analysis_instance.y_mean
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # True training values (unscaled)
    y_train_true = analysis_instance.y_train + analysis_instance.y_mean
    
    return mean_squared_error(y_train_true, y_train_pred)

def compare_methods(analysis_instance, methods=['ols_analytical', 'ridge_analytical']):
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
        train_mse = get_train_mse(analysis_instance, method)
        
        # Get test MSE
        if method == 'ols_analytical':
            test_mse = analysis_instance.mse_ols_analytical
            r2 = analysis_instance.r2_ols_analytical
        elif method == 'ridge_analytical':
            test_mse = analysis_instance.mse_ridge_analytical  
            r2 = analysis_instance.r2_ridge_analytical
        elif method == 'ols_gd':
            test_mse = analysis_instance.mse_ols_gd
            r2 = analysis_instance.r2_ols_gd
        elif method == 'ridge_gd':
            test_mse = analysis_instance.mse_ridge_gd
            r2 = analysis_instance.r2_ridge_gd
        
        results[method] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'r2': r2
        }
    
    return results

def analyze_dependence_datapoints():
    pass

def analyze_dependence_lambda_datapoints(all_results, lambda_values, sample_sizes, degree):
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
            mse = all_results[N][lam]['test_mse'][degree_idx]  # Use test_mse
            mse_values.append(mse)
        
        mse_data[f'N={N}'] = mse_values
    

    df = pd.DataFrame(mse_data, index=[f'{lam:.1e}' for lam in lambda_values])
    df.index.name = 'Lambda'
    
    print(f"\nPolynomial Degree {degree}")
    print("-" * 60)
    print(df.to_string(float_format='%.6f'))
    

    print(f"\nMinimum MSE for degree {degree}:")
    print("-" * 40)
    for col in df.columns:
        min_idx = df[col].idxmin()
        min_val = df[col].min()
        print(f"{col}: λ={min_idx}, MSE={min_val:.6f}")
    
    return df