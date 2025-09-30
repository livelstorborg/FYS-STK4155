import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso  # Add this import

from src.plotting import *
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge



N = 300
degree = 15
lam = 1e-5
eta = 1e-1
num_iters = 10000


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
    full_dataset=True
)

analysis.fit(models=('lasso',), 
            opts=('gd', 'momentum', 'adagrad', 'rmsprop', 'adam')
)


sklearn_lasso = Lasso(alpha=lam, max_iter=num_iters, tol=1e-8)
sklearn_lasso.fit(X_norm, y_centered)
y_pred_sklearn = sklearn_lasso.predict(X_norm) + y_mean

solutions_lasso = [
    y_pred_sklearn,  # Add sklearn as baseline
    analysis.runs[('lasso', 'gd')]['y_pred_test'],
    analysis.runs[('lasso', 'momentum')]['y_pred_test'],
    analysis.runs[('lasso', 'adagrad')]['y_pred_test'],
    analysis.runs[('lasso', 'rmsprop')]['y_pred_test'],
    analysis.runs[('lasso', 'adam')]['y_pred_test'],
    x,
]

compare_gd(x, y_noise, y_true, solutions=solutions_lasso, sample_size=N, degree=degree, lam=lam, type='lasso', test=False)


# ==================================================================================================
#                                    TEST SPLIT ANALYSES
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

analysis_test.fit(models=('lasso'), 
            opts=('analytical', 'gd', 'momentum', 'adagrad', 'rmsprop', 'adam')
)


sklearn_lasso_test = Lasso(alpha=lam, max_iter=num_iters, tol=1e-8)
sklearn_lasso_test.fit(X_train_s, y_train_s)
y_pred_sklearn_test = sklearn_lasso_test.predict(X_test_s) + y_mean

sort_idx = np.argsort(x_test)

solutions_test_lasso = [
    y_pred_sklearn_test[sort_idx],  # Add sklearn as baseline
    analysis_test.runs[('lasso', 'gd')]['y_pred_test'][sort_idx],
    analysis_test.runs[('lasso', 'momentum')]['y_pred_test'][sort_idx],
    analysis_test.runs[('lasso', 'adagrad')]['y_pred_test'][sort_idx],
    analysis_test.runs[('lasso', 'rmsprop')]['y_pred_test'][sort_idx],
    analysis_test.runs[('lasso', 'adam')]['y_pred_test'][sort_idx],
    x_test[sort_idx],
]

compare_gd(x, y_noise, y_true, solutions=solutions_test_lasso, sample_size=N, degree=degree, lam=lam, type='lasso', test=True)

