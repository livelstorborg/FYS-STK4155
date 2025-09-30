import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import solution_comparison, solution_comparison_gd, compare_sgd
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


N = 300
degree = 15
lam = 1e-5
eta = 1e-1
num_iters = 10000

n_epochs = 200    
batch_size = 50


x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise                 
X = polynomial_features(x, degree)           
 


# ==================================================================================================
#                                    FULL DATASET ANALYSES
# ==================================================================================================

X_norm, y_centered, _, _, y_mean = scale_data(X, y_noise) 
data = [X_norm, y_centered, y_mean]
analysis = RegressionAnalysis(
    data, 
    degree=degree, 
    lam=lam, eta=eta, 
    num_iters=num_iters,
    full_dataset=True, 
    batch_size=batch_size,
    n_epochs=n_epochs 
)

analysis.fit(models=('ols', 'ridge'), 
            opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam',
                   'sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam'),
            batch_size=batch_size
)






# ------------------------------- OLS & RIDGE (analytical vs gd) ---------------------------------
# Do some analysis for finding the best lambda for ridge regression


solutions_ols = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'gd')]['y_pred_test'],
    x        
]

solutions_ridge = [
    analysis.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis.runs[('ridge', 'gd')]['y_pred_test'],
    x        
]

solution_comparison(x, y_noise, y_true, solutions=solutions_ols, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS')
solution_comparison(x, y_noise, y_true, solutions=solutions_ridge, sample_size=N, degree=degree, lam=lam, title='Full dataset - Ridge')




# ------------------------------- DIFFERENT GD METHODS ---------------------------------

solutions_ols_gd = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'gd')]['y_pred_test'], 
    analysis.runs[('ols', 'momentum')]['y_pred_test'],
    analysis.runs[('ols', 'adagrad')]['y_pred_test'],
    analysis.runs[('ols', 'rmsprop')]['y_pred_test'],
    analysis.runs[('ols', 'adam')]['y_pred_test'],
    x,
]

solutions_ridge_gd = [
    analysis.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis.runs[('ridge', 'gd')]['y_pred_test'],
    analysis.runs[('ridge', 'momentum')]['y_pred_test'], 
    analysis.runs[('ridge', 'adagrad')]['y_pred_test'],
    analysis.runs[('ridge', 'rmsprop')]['y_pred_test'],
    analysis.runs[('ridge', 'adam')]['y_pred_test'],
    x,
]


solution_comparison_gd(x, y_noise, y_true, solutions=solutions_ols_gd, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS GD Methods', test=False)
solution_comparison_gd(x, y_noise, y_true, solutions=solutions_ridge_gd, sample_size=N, degree=degree, lam=lam, title='Full dataset - Ridge GD Methods', test=False)



# ----------------------------- Stochastic --------------------------------

# OLS
sol_ols_sgd = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'gd')]['y_pred_test'],
    analysis.runs[('ols', 'sgd')]['y_pred_test'], 
    x 
]

sol_ols_sgd_momentum = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'momentum')]['y_pred_test'],
    analysis.runs[('ols', 'sgd_momentum')]['y_pred_test'], 
    x 
]

sol_ols_sgd_adagrad = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'adagrad')]['y_pred_test'],
    analysis.runs[('ols', 'sgd_adagrad')]['y_pred_test'], 
    x 
]

sol_ols_sgd_rmsprop = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'rmsprop')]['y_pred_test'],
    analysis.runs[('ols', 'sgd_rmsprop')]['y_pred_test'], 
    x 
]
    
sol_ols_sgd_adam = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'adam')]['y_pred_test'],
    analysis.runs[('ols', 'sgd_adam')]['y_pred_test'], 
    x 
]

compare_sgd(x, y_noise, y_true, solutions=sol_ols_sgd, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS SGD', type=None)
compare_sgd(x, y_noise, y_true, solutions=sol_ols_sgd_momentum, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS SGD Momentum', type='Momentum')
compare_sgd(x, y_noise, y_true, solutions=sol_ols_sgd_adagrad, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS SGD Adagrad', type='Adagrad')
compare_sgd(x, y_noise, y_true, solutions=sol_ols_sgd_rmsprop, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS SGD RMSprop', type='RMSprop')
compare_sgd(x, y_noise, y_true, solutions=sol_ols_sgd_adam, sample_size=N, degree=degree, lam=lam, title='Full dataset - OLS SGD Adam', type='Adam')







# ==================================================================================================
#                          TEST SPLIT ANALYSES
# ==================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
x_train = X_train[:, 0] 
x_test = X_test[:, 0] 

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
            opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam',
                   'sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam'),
            batch_size=batch_size
)

sort_idx = np.argsort(x_test)






# ------------------------------- (analytical vs gd) ---------------------------------

solutions_test_ols = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]  
]

solutions_test_ridge = [
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'][sort_idx],
    x_test[sort_idx]  
]

solution_comparison(x, y_noise, y_true, solutions=solutions_test_ols, sample_size=N, degree=degree, lam=lam, title='Test split - OLS')
solution_comparison(x, y_noise, y_true, solutions=solutions_test_ridge, sample_size=N, degree=degree, lam=lam, title='Test split - Ridge')






# ------------------------------- DIFFERENT GD METHODS ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
x_train = X_train[:, 0] 
x_test = X_test[:, 0] 

X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)



solutions_test_ols = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'][sort_idx], 
    analysis_test.runs[('ols', 'momentum')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'adagrad')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'rmsprop')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'adam')]['y_pred_test'][sort_idx],
    x_test[sort_idx],
]

solutions_test_ridge = [
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'momentum')]['y_pred_test'][sort_idx], 
    analysis_test.runs[('ridge', 'adagrad')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'rmsprop')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ridge', 'adam')]['y_pred_test'][sort_idx],
    x_test[sort_idx],
]

solution_comparison_gd(x, y_noise, y_true, solutions=solutions_test_ols, sample_size=N, degree=degree, lam=lam, title='Test split - OLS GD Methods', test=True)
solution_comparison_gd(x, y_noise, y_true, solutions=solutions_test_ridge, sample_size=N, degree=degree, lam=lam, title='Test split - Ridge GD Methods', test=True)






# ----------------------------- Stochastic --------------------------------

# ----------------------------- Stochastic --------------------------------

# OLS
sol_test_ols_sgd = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'sgd')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]
]

sol_test_ols_sgd_momentum = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'momentum')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'sgd_momentum')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]
]

sol_test_ols_sgd_adagrad = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'adagrad')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'sgd_adagrad')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]
]

sol_test_ols_sgd_rmsprop = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'rmsprop')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'sgd_rmsprop')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]
]
    
sol_test_ols_sgd_adam = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'adam')]['y_pred_test'][sort_idx],
    analysis_test.runs[('ols', 'sgd_adam')]['y_pred_test'][sort_idx], 
    x_test[sort_idx]
]

compare_sgd(x, y_noise, y_true, solutions=sol_test_ols_sgd, sample_size=N, degree=degree, lam=lam, title='Test split - OLS SGD', type=None)
compare_sgd(x, y_noise, y_true, solutions=sol_test_ols_sgd_momentum, sample_size=N, degree=degree, lam=lam, title='Test split - OLS SGD Momentum', type='Momentum')
compare_sgd(x, y_noise, y_true, solutions=sol_test_ols_sgd_adagrad, sample_size=N, degree=degree, lam=lam, title='Test split - OLS SGD Adagrad', type='Adagrad')
compare_sgd(x, y_noise, y_true, solutions=sol_test_ols_sgd_rmsprop, sample_size=N, degree=degree, lam=lam, title='Test split - OLS SGD RMSprop', type='RMSprop')
compare_sgd(x, y_noise, y_true, solutions=sol_test_ols_sgd_adam, sample_size=N, degree=degree, lam=lam, title='Test split - OLS SGD Adam', type='Adam')



