import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from project1.regression import OLS_parameters, polynomial_features, runge
from project1.utils.error_analysis import mse, r_squared
from project1.utils.plot import plot

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
        "MSE_train": mse_ols_train,
        "MSE_test": mse_ols_test,
        "R2_train": r2_ols_train,
        "R2_test": r2_ols_test,
        "theta": theta_ols,
    }


try:
    n_poly = 20
    for i in tqdm(range(1, n_poly + 1)):
        result = one_iter(x, i)
        results.append(result)
except KeyboardInterrupt:
    pass


df = pd.DataFrame(results)

# Use DataFrame columns for plotting and provide labels
# One figure for MSE
plot(
    df["Degree"],
    {"MSE (train)": df["MSE_train"], "MSE (test)": df["MSE_test"]},
    ylabel="MSE",
    figsize=(8, 6),
    filename="figs/deg_vs_MSE_OLS.pdf",
)
plt.show()

# One figure for R^2
plot(
    df["Degree"],
    {r"$R^2$ (train)": df["R2_train"], r"$R^2$ (test)": df["R2_test"]},
    ylabel=r"$R^2$",
    figsize=(8, 6),
    filename="figs/deg_vs_R2_OLS.pdf",
)
plt.show()
