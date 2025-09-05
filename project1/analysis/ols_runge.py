import numpy as np
from sklearn.model_selection import train_test_split

from project1.regression import OLS_parameters, polynomial_features, runge
from project1.utils.error_analysis import mse, r_squared

x = np.linspace(-1, 1, 1000)

for i in range(1, 15 + 1):
    X = polynomial_features(x, i)
    y = runge(x)

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
    mse_ols_train = mse(X_train, y_pred_train, theta_ols)
    mse_ols_test = mse(X_test, y_pred_test, theta_ols)

    # Calculate R^2
    r2_ols_train = r_squared(y_train, y_pred_train)
    r2_ols_test = r_squared(y_test, y_pred_test)

    # plot
    print(mse_ols_train, mse_ols_train, r2_ols_train, r2_ols_test)
