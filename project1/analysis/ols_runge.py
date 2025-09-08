import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from project1.regression import OLS_parameters, polynomial_features, runge
from project1.utils.error_analysis import mse, r_squared

x = np.linspace(-1, 1, 1000)
y = runge(x) + np.random.normal(0, 0.1, size=x.shape)

results = []


def one_iter(x, i):
    X = polynomial_features(x, i)

    # scaling data
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean(axis=0)
    y_centered = y - y_mean

    # Split data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_centered, test_size=0.2
    )

    # Perform OLS regression
    theta_ols = OLS_parameters(X_train, y_train)
    y_pred_train = X_train @ theta_ols
    y_pred_test = X_test @ theta_ols

    # Calculate MSE
    mse_ols_train = mse(X_train, y_train, theta_ols)
    mse_ols_test = mse(X_test, y_test, theta_ols)

    # Calculate R^2
    r2_ols_train = r_squared(y_train, y_pred_train)
    r2_ols_test = r_squared(y_test, y_pred_test)

    return {
        "Degree": i,
        "MSE train": mse_ols_train,
        "MSE test": mse_ols_test,
        "R2 train": r2_ols_train,
        "R2 test": r2_ols_test,
    }


try:
    n_poly = 15
    for i in tqdm(range(1, n_poly + 1)):
        result = one_iter(x, i)
        results.append(result)
except KeyboardInterrupt:
    pass


df = pd.DataFrame(results)

print("\n=== Error metrics by polynomial degree ===")
print(df.round(4).to_string(index=False))
