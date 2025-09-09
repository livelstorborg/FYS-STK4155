import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from project1.regression import OLS, polynomial_features, runge
from project1.utils.error_analysis import mse, r_squared
from project1.utils.plot import plot



def one_iter(x, i):
    
    y = runge(x) + np.random.normal(0, 0.1, size=x.shape)
    
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
    model = OLS()
    model.fit(X_train, y_train)
    theta_ols = model.theta_
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

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
        "Theta": theta_ols,
    }



x = np.linspace(-1, 1, 1000)
results = []

try:
    n_poly = 15
    for i in tqdm(range(1, n_poly + 1)):
        result = one_iter(x, i)
        results.append(result)
except KeyboardInterrupt:
    pass

df = pd.DataFrame(results)

# Plot for MSE
plot(
    df["Degree"],
    {"MSE (train)": df["MSE_train"], "MSE (test)": df["MSE_test"]},
    ylabel="MSE",
    figsize=(8, 6),
    filename="../figs/deg_vs_MSE_OLS.pdf",
)
plt.show()

# Plot for R^2
plot(
    df["Degree"],
    {r"$R^2$ (train)": df["R2_train"], r"$R^2$ (test)": df["R2_test"]},
    ylabel=r"$R^2$",
    figsize=(8, 6),
    filename="../figs/deg_vs_R2_OLS.pdf",
)
plt.show()


df_subset = df[df["Degree"] <= 15]

# Convert the Series of arrays into a list of arrays
theta_list = df_subset["Theta"].to_list()

# Find the maximum degree to determine how many theta components we have
max_degree = max(len(t) for t in theta_list)

# Create dictionary to store all theta components 
theta_dict = {}
for i in range(max_degree):
    theta_dict[f"theta_{i + 1}"] = [t[i] if i < len(t) else np.nan for t in theta_list]

# Plot for evolution of theta_1
plot(
    df_subset["Degree"],
    {r"$\theta_1$": theta_dict["theta_1"]},  # Use LaTeX formatting for legend
    ylabel=r"Evolution of $\theta_1$", 
    figsize=(8, 6),
    filename="../figs/deg_vs_theta1.pdf",
)
plt.show()

# Analysis: Dependence on the number of data points and the polynomial degree
n_points_list = [50, 500, 1000, 10000, 1000000]
dependence_results = []

for n_points in tqdm(n_points_list, desc="Data points"):
    
    x_dep = np.linspace(-1, 1, n_points)
    for degree in range(1, 16):
        result = one_iter(x_dep, degree)
        result["N_points"] = n_points  # Add the data points info
        dependence_results.append(result)

df_dependence = pd.DataFrame(dependence_results)

# Plot: MSE vs polynomial degree for different numbers of data points
mse_vs_degree = {}
for n_points in n_points_list:
    subset = df_dependence[df_dependence["N_points"] == n_points]
    mse_vs_degree[f"N = {n_points}"] = subset["MSE_test"].values

plot(
    list(range(1, 16)),
    mse_vs_degree,
    ylabel="MSE",
    figsize=(8, 6),
    filename="../figs/mse_vs_degree_datapoints.pdf",
)
# plt.yscale('log')
plt.show()