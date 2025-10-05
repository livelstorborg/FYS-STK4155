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


eta_list_gd_ols = [0.37, 0.365, 0.2, 0.1]
eta_list_momentum_ols = [0.4, 0.3, 0.1, 0.05]
eta_list_adagrad_ols = [0.3, 0.25, 0.15, 0.1]
eta_list_rmsprop_ols = [0.02, 0.01, 0.005, 0.001]
eta_list_adam_ols = [0.2, 0.1, 0.05, 0.02]

eta_list_gd_ridge = [0.37, 0.365, 0.2, 0.1]
eta_list_momentum_ridge = [0.22, 0.2, 0.15, 0.05]
eta_list_adagrad_ridge = [0.225, 0.22, 0.215, 0.15]
eta_list_rmsprop_ridge = [0.01, 0.005, 0.002, 0.001]
eta_list_adam_ridge = [0.25, 0.2, 0.1, 0.05, 0.02]






# ============================================================================
#                       Setup for all optimizers
# ============================================================================


optimizers_ols = {
    'gd': eta_list_gd_ols,
    'momentum': eta_list_momentum_ols,
    'adagrad': eta_list_adagrad_ols,
    'rmsprop': eta_list_rmsprop_ols,
    'adam': eta_list_adam_ols
}

optimizers_ridge = {
    'gd': eta_list_gd_ridge,
    'momentum': eta_list_momentum_ridge,
    'adagrad': eta_list_adagrad_ridge,
    'rmsprop': eta_list_rmsprop_ridge,
    'adam': eta_list_adam_ridge
}


mse_history_ols_gd, mse_history_ols_momentum, mse_history_ols_adagrad, mse_history_ols_rmsprop, mse_history_ols_adam = [], [], [], [], []
mse_history_ridge_gd, mse_history_ridge_momentum, mse_history_ridge_adagrad, mse_history_ridge_rmsprop, mse_history_ridge_adam = [], [], [], [], []


# Run OLS with OLS-specific eta values
for opt_name, eta_list in optimizers_ols.items():
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
        analysis.fit(models=('ols',), opts=opt_name)

        # Store MSE histories for OLS
        if opt_name == 'gd':
            mse_history_ols_gd.append(analysis.runs[('ols', opt_name)]['history'])
        elif opt_name == 'momentum':
            mse_history_ols_momentum.append(analysis.runs[('ols', opt_name)]['history'])
        elif opt_name == 'adagrad':
            mse_history_ols_adagrad.append(analysis.runs[('ols', opt_name)]['history'])
        elif opt_name == 'rmsprop':
            mse_history_ols_rmsprop.append(analysis.runs[('ols', opt_name)]['history'])
        elif opt_name == 'adam':
            mse_history_ols_adam.append(analysis.runs[('ols', opt_name)]['history'])

# Run Ridge with Ridge-specific eta values
for opt_name, eta_list in optimizers_ridge.items():
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
        analysis.fit(models=('ridge',), opts=opt_name)

        # Store MSE histories for Ridge
        if opt_name == 'gd':
            mse_history_ridge_gd.append(analysis.runs[('ridge', opt_name)]['history'])
        elif opt_name == 'momentum':
            mse_history_ridge_momentum.append(analysis.runs[('ridge', opt_name)]['history'])
        elif opt_name == 'adagrad':
            mse_history_ridge_adagrad.append(analysis.runs[('ridge', opt_name)]['history'])
        elif opt_name == 'rmsprop':
            mse_history_ridge_rmsprop.append(analysis.runs[('ridge', opt_name)]['history'])
        elif opt_name == 'adam':
            mse_history_ridge_adam.append(analysis.runs[('ridge', opt_name)]['history'])









# ============================================================================
#                  Gradient descent: OLS vs RIDGE
# ============================================================================



fig, ax = plt.subplots(figsize=(10, 6))

n_etas = len(eta_list_gd_ols)
colors = cm.plasma(np.linspace(0.15, 0.95, n_etas))

for i, eta in enumerate(eta_list_gd_ols):
    ax.plot(range(1, len(mse_history_ols_gd[i]) + 1), mse_history_ols_gd[i], '-', 
            color=colors[i], linewidth=2.5, label=f'OLS (η={eta})')
    

    iterations_ols = len(mse_history_ols_gd[i])
    final_mse_ols = mse_history_ols_gd[i][-1]
    initial_mse_ols = mse_history_ols_gd[i][0]
    if final_mse_ols < initial_mse_ols: 
        ax.axvline(iterations_ols, color=colors[i], linewidth=1.5, alpha=0.7)
    

    ax.plot(range(1, len(mse_history_ridge_gd[i]) + 1), mse_history_ridge_gd[i], '--', 
            color=colors[i], linewidth=2.5, label=f'Ridge (η={eta})', alpha=0.8)
    

    iterations_ridge = len(mse_history_ridge_gd[i])
    final_mse_ridge = mse_history_ridge_gd[i][-1]
    initial_mse_ridge = mse_history_ridge_gd[i][0]
    if final_mse_ridge < initial_mse_ridge:
        ax.axvline(iterations_ridge, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Iterations', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=16)
ax.legend(fontsize=16, loc='best', ncol=2)

plt.tight_layout()
plt.savefig('figs/gd_ols_vs_ridge.pdf', dpi=300, bbox_inches='tight')
# plt.show()







# ============================================================================
#                 Comparing all gd methods: OLS and RIDGE
# ============================================================================

methods_data = [
    ('GD', eta_list_gd_ols, eta_list_gd_ridge, mse_history_ols_gd, mse_history_ridge_gd),
    ('Momentum', eta_list_momentum_ols, eta_list_momentum_ridge, mse_history_ols_momentum, mse_history_ridge_momentum),
    ('AdaGrad', eta_list_adagrad_ols, eta_list_adagrad_ridge, mse_history_ols_adagrad, mse_history_ridge_adagrad),
    ('RMSprop', eta_list_rmsprop_ols, eta_list_rmsprop_ridge, mse_history_ols_rmsprop, mse_history_ridge_rmsprop),
    ('Adam', eta_list_adam_ols, eta_list_adam_ridge, mse_history_ols_adam, mse_history_ridge_adam)
]


# ----- OLS -----
ols_data = []
for method_name, etas_ols, etas_ridge, hist_ols, hist_ridge in methods_data:
    for i, eta in enumerate(etas_ols):
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
for method_name, etas_ols, etas_ridge, hist_ols, hist_ridge in methods_data:
    for i, eta in enumerate(etas_ridge):
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

        

print("\n" + "="*80)
print("OLS RESULTS")
print("="*80)
print(df_ols.to_string(index=False))

print("\n\n" + "="*80)
print("RIDGE RESULTS (λ=0.01)")
print("="*80)
print(df_ridge.to_string(index=False))
print("="*80)


