import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import optimizer_comparison, method_comparison 
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge

N = 1000
degree = 8
lam = 1e-2
eta = 1e-2
num_iters = 1000
batch_size = 50
n_epochs = 100

x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)
y_noise = y_true + random_noise
X = polynomial_features(x, degree)

X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
X_train_norm, y_train_centered, X_mean, X_std, y_mean = scale_data(X_train, y_train)
X_test_norm, y_test_centered, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)

data = [X_train_norm, X_test_norm, y_train_centered, y_test_centered, None, None, y_mean]



# =============================================================================
#                          ANALYSIS 1: OLS ALL OPTIMIZERS
# =============================================================================
analysis1 = RegressionAnalysis(
    data,
    degree=degree,
    lam=lam,
    eta=eta,
    num_iters=num_iters,
    batch_size=batch_size,
    n_epochs=n_epochs,
    random_state=42
)

analysis1.fit_many(
    models=('ols',),
    opts=('gd', 'momentum', 'adagrad', 'rmsprop', 'adam', 
          'sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam')
)

ols_results = {}
for opt in ['gd', 'momentum', 'adagrad', 'rmsprop', 'adam', 
            'sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam']:
    ols_results[opt] = {
        'test_mse': analysis1.get_metric('ols', opt, 'test_mse'),
        'y_pred': analysis1.runs[('ols', opt)]['y_pred_test'],
        'history': analysis1.runs[('ols', opt)]['history']
    }

best_full_batch = min(['gd', 'momentum', 'adagrad', 'rmsprop', 'adam'], 
                     key=lambda opt: ols_results[opt]['test_mse'])
best_stochastic = min(['sgd', 'sgd_momentum', 'sgd_adagrad', 'sgd_rmsprop', 'sgd_adam'],
                     key=lambda opt: ols_results[opt]['test_mse'])


optimizer_comparison()

# =============================================================================
#                    ANALYSIS 2: BEST METHODS FOR ALL MODELS
# =============================================================================
analysis2 = RegressionAnalysis(
    data,
    degree=degree,
    lam=lam,
    eta=eta,
    num_iters=num_iters,
    batch_size=batch_size,
    n_epochs=n_epochs,
    random_state=42
)

analysis2.fit_many(
    models=('ols', 'ridge', 'lasso'),
    opts=(best_full_batch, best_stochastic)
)

method_results = {}
for model in ['ols', 'ridge', 'lasso']:
    for opt in [best_full_batch, best_stochastic]:
        method_results[f'{model}_{opt}'] = {
            'test_mse': analysis2.get_metric(model, opt, 'test_mse'),
            'y_pred': analysis2.runs[(model, opt)]['y_pred_test'],
            'history': analysis2.runs[(model, opt)]['history']
        }

method_comparison()

