import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import solution_comparison
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


N = 300
degree = 15
lam = 1e-2  
eta = 1e-1
num_iters = 100000


x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise                 
X = polynomial_features(x, degree)           
 


# ==================================================================================================
#                          FULL DATASET ANALYSIS - OLS & RIDGE (analytical vs gd)
# ==================================================================================================
# X_norm, y_centered, _, _, y_mean = scale_data(X, y_noise) 
# data = [X_norm, y_centered, y_mean]
# analysis = RegressionAnalysis(
#     data, 
#     degree=degree, 
#     lam=lam, eta=eta, 
#     num_iters=num_iters,
#     full_dataset=True  
# )

# analysis.fit_many(models=('ols', 'ridge'), 
#                   opts=('analytical', 'gd'))

# solutions = [
#     analysis.runs[('ols', 'analytical')]['y_pred_test'],
#     analysis.runs[('ols', 'gd')]['y_pred_test'], 
#     analysis.runs[('ridge', 'analytical')]['y_pred_test'],
#     analysis.runs[('ridge', 'gd')]['y_pred_test'],
#     x        
# ]

# solution_comparison(x, y_noise, y_true, solutions=solutions, sample_size=N, degree=degree, lam=lam)






# ==================================================================================================
#                        TEST SPLIT ANALYSIS  - (analytical vs gd)
# ==================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
x_train = X_train[:, 0] 
x_test = X_test[:, 0] 

# Scaling the training data and using the same parameters to scale the test data (to avoid data leakage)
X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)



data_test = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]
analysis_test = RegressionAnalysis(
    data_test, 
    degree=degree, 
    lam=lam, eta=eta, 
    num_iters=num_iters,
    full_dataset=False  
)

analysis_test.fit(models=('ols', 'ridge'), 
                   opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam'))



sort_idx = np.argsort(x_test)
solutions_test = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'][sort_idx], 
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'][sort_idx],
    x_test[sort_idx]  # Also sort x_test to match
]



solution_comparison(x, y_noise, y_true, solutions=solutions_test, sample_size=N, degree=degree, lam=lam)






# ------------------------------- DIFFERENT GD METHODS ---------------------------------



# =============================================================================
#                          FULL DATASET ANALYSIS
# =============================================================================
X_norm, y_centered, X_mean, X_std, y_mean = scale_data(X, y_true)
data = [X_norm, y_centered, y_mean]
analysis = RegressionAnalysis(
    data, 
    degree=degree, 
    lam=lam, 
    eta=eta, 
    num_iters=num_iters,
    full_dataset=True
)

analysis.fit(models=('ols', 'ridge'), 
             opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam'))


solutions_ols = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'gd')]['y_pred_test'], 
    analysis.runs[('ols', 'momentum')]['y_pred_test'],
    analysis.runs[('ols', 'adagrad')]['y_pred_test'],
    analysis.runs[('ols', 'rmsprop')]['y_pred_test'],
    analysis.runs[('ols', 'adam')]['y_pred_test'],
    x,
]

solutions_ridge = [
    analysis.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis.runs[('ridge', 'gd')]['y_pred_test'],
    analysis.runs[('ridge', 'momentum')]['y_pred_test'], 
    analysis.runs[('ridge', 'adagrad')]['y_pred_test'],
    analysis.runs[('ridge', 'rmsprop')]['y_pred_test'],
    analysis.runs[('ridge', 'adam')]['y_pred_test'],
    x,
]

solution_comparison_gd(x, y_noise, y_true, solutions=solutions_ols, sample_size=N, degree=degree, lam=lam, test=False)
solution_comparison_gd(x, y_noise, y_true, solutions=solutions_ridge, sample_size=N, degree=degree, lam=lam, test=False)





# =============================================================================
#                        TEST SPLIT ANALYSIS  
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
x_train = X_train[:, 0] 
x_test = X_test[:, 0] 

# Scaling the training data and using the same parameters to scale the test data (to avoid data leakage)
X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)


data_test = [X_train_s, X_test_s, y_train_s, y_test_s, x_train, x_test, y_mean]
analysis_test = RegressionAnalysis(
    data_test, 
    degree=degree, 
    lam=lam, eta=eta, 
    num_iters=num_iters,
    full_dataset=False  
)

analysis_test.fit(models=('ols', 'ridge'), 
                   opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam'))

solutions_test_ols = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'], 
    analysis_test.runs[('ols', 'momentum')]['y_pred_test'],
    analysis_test.runs[('ols', 'adagrad')]['y_pred_test'],
    analysis_test.runs[('ols', 'rmsprop')]['y_pred_test'],
    analysis_test.runs[('ols', 'adam')]['y_pred_test'],
    x_test,       
]

solutions_test_ridge = [
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'],
    analysis_test.runs[('ridge', 'momentum')]['y_pred_test'], 
    analysis_test.runs[('ridge', 'adagrad')]['y_pred_test'],
    analysis_test.runs[('ridge', 'rmsprop')]['y_pred_test'],
    analysis_test.runs[('ridge', 'adam')]['y_pred_test'],
    x_test,       
]

solution_comparison_gd(x, y_noise, y_true, solutions=solutions_test_ols, sample_size=N, degree=degree, lam=lam, test=True)
solution_comparison_gd(x, y_noise, y_true, solutions=solutions_test_ridge, sample_size=N, degree=degree, lam=lam, test=True)






# ------------------------------- STOCHASTIC --------------------------------

# Parameters
N = 300
degree = 8
lam = 1e-2
eta = 5e-1  
num_iters = 1000

n_epochs = 200    
batch_size = 50


x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)
y_noise = y_true + random_noise
X = polynomial_features(x, degree)


X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
x_train = X_train[:, 0]  
x_test = X_test[:, 0]

X_train_norm, y_train_centered, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_norm, y_test_centered, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)

data = [X_train_norm, X_test_norm, y_train_centered, y_test_centered, x_train, x_test, y_mean]



analysis = RegressionAnalysis(
    data,
    degree=degree,
    lam=lam,
    eta=eta,
    num_iters=num_iters,
    batch_size=batch_size,
    n_epochs=n_epochs
)


analysis.fit_many(
    models=('ols',),
    opts=('gd', 'momentum', 'adagrad', 'rmsprop', 'adam', 
          'sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam'),
    batch_size=batch_size 
)


# Compare all optimizers for OLS
batch_optimizers = ['gd', 'momentum', 'adagrad', 'rmsprop', 'adam']
stoch_optimizers = ['sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam']

for opt in batch_optimizers + stoch_optimizers:
    train_mse = analysis.get_metric('ols', opt, 'train_mse')
    test_mse = analysis.get_metric('ols', opt, 'test_mse')
    test_r2 = analysis.get_metric('ols', opt, 'test_r2')
    history = analysis.runs[('ols', opt)]['history']
    final_loss = history[-1] if history else 0
    opt_type = 'Stochastic' if opt.startswith('sgd') else 'Batch'
    


sort_idx = np.argsort(x_test)
solutions_test = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'][sort_idx], 
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'][sort_idx],
    x_test[sort_idx]  # Also sort x_test to match
]



solution_comparison(x, y_noise, y_true, solutions=solutions_test, sample_size=N, degree=degree, lam=lam)


compare_gd()