import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from sklearn.model_selection import train_test_split

from src.plotting import *
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


np.random.seed(42)


N = 500
degree = 5
lam = 1e-2
num_iters = 1500


x = np.linspace(-1, 1, N)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise                 
X = polynomial_features(x, degree)


X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.25, random_state=42)
x_train = X_train[:, 0] 
x_test = X_test[:, 0] 
X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)
data = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]


# eta_list_gd = [0.37, 0.365, 0.2, 0.1]
# eta_list_momentum = [0.4, 0.3, 0.1, 0.05]
# eta_list_adagrad = [0.3, 0.25, 0.15, 0.1]
# eta_list_rmsprop = [0.02, 0.01, 0.005, 0.001]
# eta_list_adam = [0.2, 0.1, 0.05, 0.02]


eta_list_gd = [0.5, 0.37, 0.365, 0.2, 0.1]
eta_list_momentum = [0.2, 0.1, 0.05, 0.04]
eta_list_adagrad = [0.3, 0.2, 0.15, 0.1]
eta_list_rmsprop = [0.02, 0.01, 0.005, 0.001]
eta_list_adam = [0.15, 0.1, 0.05, 0.02]






# ============================================================================
#                       Setup for all optimizers
# ============================================================================


optimizers = {
    'gd': eta_list_gd,
    'momentum': eta_list_momentum,
    'adagrad': eta_list_adagrad,
    'rmsprop': eta_list_rmsprop,
    'adam': eta_list_adam
}


mse_history_ols_gd, mse_history_ols_momentum, mse_history_ols_adagrad, mse_history_ols_rmsprop, mse_history_ols_adam = [], [], [], [], []
mse_history_ridge_gd, mse_history_ridge_momentum, mse_history_ridge_adagrad, mse_history_ridge_rmsprop, mse_history_ridge_adam = [], [], [], [], []
mse_history_lasso_gd, mse_history_lasso_momentum, mse_history_lasso_adagrad, mse_history_lasso_rmsprop, mse_history_lasso_adam = [], [], [], [], []


for opt_name, eta_list in optimizers.items():
    for eta in eta_list:
        analysis = RegressionAnalysis(
            data,
            degree=degree,
            lam=lam,
            eta=eta,
            num_iters=num_iters,
            full_dataset=False,
            tol_relative=1e-5
        )
        analysis.fit(models=('ols', 'ridge', 'lasso'), opts=opt_name)

        # Store MSE histories for each model and optimizer
        if opt_name == 'gd':
            mse_history_ols_gd.append(analysis.runs[('ols', opt_name)]['history'])
            mse_history_ridge_gd.append(analysis.runs[('ridge', opt_name)]['history'])
            mse_history_lasso_gd.append(analysis.runs[('lasso', opt_name)]['history'])
        elif opt_name == 'momentum':
            mse_history_ols_momentum.append(analysis.runs[('ols', opt_name)]['history'])
            mse_history_ridge_momentum.append(analysis.runs[('ridge', opt_name)]['history'])
            mse_history_lasso_momentum.append(analysis.runs[('lasso', opt_name)]['history'])
        elif opt_name == 'adagrad':
            mse_history_ols_adagrad.append(analysis.runs[('ols', opt_name)]['history'])
            mse_history_ridge_adagrad.append(analysis.runs[('ridge', opt_name)]['history'])
            mse_history_lasso_adagrad.append(analysis.runs[('lasso', opt_name)]['history'])
        elif opt_name == 'rmsprop':
            mse_history_ols_rmsprop.append(analysis.runs[('ols', opt_name)]['history'])
            mse_history_ridge_rmsprop.append(analysis.runs[('ridge', opt_name)]['history'])
            mse_history_lasso_rmsprop.append(analysis.runs[('lasso', opt_name)]['history'])
        elif opt_name == 'adam':
            mse_history_ols_adam.append(analysis.runs[('ols', opt_name)]['history'])
            mse_history_ridge_adam.append(analysis.runs[('ridge', opt_name)]['history'])
            mse_history_lasso_adam.append(analysis.runs[('lasso', opt_name)]['history'])












# ============================================================================
#                 Comparing all gd methods: OLS and RIDGE
# ============================================================================

methods_data = [
    ('GD', eta_list_gd, mse_history_ols_gd, mse_history_ridge_gd, mse_history_lasso_gd),
    ('Momentum', eta_list_momentum, mse_history_ols_momentum, mse_history_ridge_momentum, mse_history_lasso_momentum),
    ('AdaGrad', eta_list_adagrad, mse_history_ols_adagrad, mse_history_ridge_adagrad, mse_history_lasso_adagrad),
    ('RMSprop', eta_list_rmsprop, mse_history_ols_rmsprop, mse_history_ridge_rmsprop, mse_history_lasso_rmsprop),
    ('Adam', eta_list_adam, mse_history_ols_adam, mse_history_ridge_adam, mse_history_lasso_adam)
]


# ----- OLS -----
ols_data = []
for method_name, etas, hist_ols, _, _ in methods_data:  # Added extra _ for lasso
    for i, eta in enumerate(etas):
        iterations = len(hist_ols[i])
        final_mse = hist_ols[i][-1]
        initial_mse = hist_ols[i][0]
        

        if final_mse > initial_mse:
            converged = 'Diverged'
            final_mse_display = '-'
        elif len(hist_ols[i]) == num_iters:
            converged = 'No'
            final_mse_display = f'{final_mse:.6f}'
        else:
            converged = 'Yes'
            final_mse_display = f'{final_mse:.6f}'
        
        ols_data.append({
            'Method': method_name,
            'Learning Rate': eta,
            'Iterations': iterations,
            'Final MSE': final_mse_display,
            'Converged': converged
        })

df_ols = pd.DataFrame(ols_data)


# ----- RIDGE -----
ridge_data = []
for method_name, etas, _, hist_ridge, _ in methods_data:  # Added extra _ for lasso
    for i, eta in enumerate(etas):
        iterations = len(hist_ridge[i])
        final_mse = hist_ridge[i][-1]
        initial_mse = hist_ridge[i][0]


        if final_mse > initial_mse:
            converged = 'Diverged'
            final_mse_display = '-'
        elif len(hist_ridge[i]) == num_iters:
            converged = 'No'
            final_mse_display = f'{final_mse:.6f}'
        else:
            converged = 'Yes'
            final_mse_display = f'{final_mse:.6f}'

        ridge_data.append({
            'Method': method_name,
            'Learning Rate': eta,
            'Iterations': iterations,
            'Final MSE': final_mse_display,
            'Converged': converged
        })

df_ridge = pd.DataFrame(ridge_data)






# ----- LASSO -----
lasso_data = []
for method_name, etas, _, _, hist_lasso in methods_data:  # Extract the 5th element (lasso histories)
    for i, eta in enumerate(etas):
        iterations = len(hist_lasso[i])
        final_mse = hist_lasso[i][-1]
        initial_mse = hist_lasso[i][0]


        if final_mse > initial_mse:
            converged = 'Diverged'
            final_mse_display = '-'
        elif len(hist_lasso[i]) == num_iters:  # Fixed: use hist_lasso instead of hist_ridge
            converged = 'No'
            final_mse_display = f'{final_mse:.6f}'
        else:
            converged = 'Yes'
            final_mse_display = f'{final_mse:.6f}'

        lasso_data.append({  # Fixed: append to lasso_data instead of ridge_data
            'Method': method_name,
            'Learning Rate': eta,
            'Iterations': iterations,
            'Final MSE': final_mse_display,
            'Converged': converged
        })

df_lasso = pd.DataFrame(lasso_data)

        

print("\n" + "="*80)
print("OLS RESULTS")
print("="*80)
print(df_ols.to_string(index=False))

print("\n\n" + "="*80)
print("RIDGE RESULTS (λ=0.01)")
print("="*80)
print(df_ridge.to_string(index=False))
print("="*80)

print("\n\n" + "="*80)
print("LASSO RESULTS (λ=0.01)")
print("="*80)
print(df_lasso.to_string(index=False))
print("="*80)