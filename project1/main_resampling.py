import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from src.utils import runge



# def plot_error_vs_degree(degrees, errors, biases=None, variances=None, method_name='', logscale=True):
#     """
#     Plot error vs polynomial degree.
    
#     Parameters:
#     -----------
#     degrees : array-like
#         Polynomial degrees
#     errors : array-like
#         Error/MSE values
#     biases : array-like, optional
#         Bias² values (only for bootstrap)
#     variances : array-like, optional
#         Variance values (only for bootstrap)
#     method_name : str
#         Method name for title (e.g., 'OLS', 'Ridge')
#     logscale : bool
#         Use log scale for y-axis
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(degrees, errors, label='Error/MSE', linewidth=2)
    
#     if biases is not None:
#         plt.plot(degrees, biases, label='Bias²', linewidth=2)
#     if variances is not None:
#         plt.plot(degrees, variances, label='Variance', linewidth=2)
    
#     if logscale:
#         plt.yscale('log')
    
#     plt.xlabel('Polynomial Degree', fontsize=12)
#     plt.ylabel('Error', fontsize=12)
#     plt.title(f'Error vs Polynomial Degree ({method_name})', fontsize=14)
#     plt.legend(fontsize=11)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_error_vs_datasize(N_list, errors, biases=None, variances=None, degree=None, method_name='', logscale=False):
#     """
#     Plot error vs dataset size.
    
#     Parameters:
#     -----------
#     N_list : array-like
#         Dataset sizes
#     errors : array-like
#         Error/MSE values
#     biases : array-like, optional
#         Bias² values (only for bootstrap)
#     variances : array-like, optional
#         Variance values (only for bootstrap)
#     degree : int, optional
#         Polynomial degree used
#     method_name : str
#         Method name for title (e.g., 'OLS', 'Ridge')
#     logscale : bool
#         Use log scale for y-axis
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(N_list, errors, marker='o', label='Error/MSE', linewidth=2)
    
#     if biases is not None:
#         plt.plot(N_list, biases, marker='s', label='Bias²', linewidth=2)
#     if variances is not None:
#         plt.plot(N_list, variances, marker='^', label='Variance', linewidth=2)
    
#     if logscale:
#         plt.yscale('log')
    
#     plt.xlabel('Dataset Size (N)', fontsize=12)
#     plt.ylabel('Error', fontsize=12)
    
#     title = f'Error vs Dataset Size ({method_name})'
#     if degree is not None:
#         title += f' (Degree={degree})'
#     plt.title(title, fontsize=14)
    
#     plt.legend(fontsize=11)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()





# def bootstrap_analysis(x_train, y_train, x_test, y_test, degree, method='ols', lam=0.0, n_bootstraps=100):
#     """
#     Perform bootstrap analysis for bias-variance trade-off.
    
#     Parameters:
#     -----------
#     x_train, y_train : Training data
#     x_test, y_test : Test data
#     degree : Polynomial degree
#     method : 'ols', 'ridge', or 'lasso'
#     lam : Regularization parameter (for ridge/lasso)
#     n_bootstraps : Number of bootstrap samples
    
#     Returns:
#     --------
#     error, bias, variance : float
#     """
#     # Select model
#     if method.lower() == 'ols':
#         model = make_pipeline(PolynomialFeatures(degree=degree), 
#                             LinearRegression(fit_intercept=False))
#     elif method.lower() == 'ridge':
#         model = make_pipeline(PolynomialFeatures(degree=degree), 
#                             Ridge(alpha=lam, fit_intercept=False))
#     elif method.lower() == 'lasso':
#         model = make_pipeline(PolynomialFeatures(degree=degree), 
#                             Lasso(alpha=lam, fit_intercept=False, max_iter=10000))
#     else:
#         raise ValueError(f"Unknown method: {method}. Use 'ols', 'ridge', or 'lasso'")
    
#     # Bootstrap predictions
#     y_pred = np.empty((y_test.shape[0], n_bootstraps))
#     for i in range(n_bootstraps):
#         x_, y_ = resample(x_train, y_train)
#         y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
    
#     # Calculate metrics
#     error = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
#     bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
#     variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

#     return error, bias, variance


# def cv_sklearn(x, y, degree, k_folds=5, method='ols', lam=0.01):
#     """
#     Perform k-fold cross-validation using sklearn's cross_val_score.
    
#     Parameters:
#     -----------
#     x, y : Full dataset
#     degree : Polynomial degree
#     k_folds : Number of folds
#     method : 'ols', 'ridge', or 'lasso'
#     lam : Regularization parameter (for ridge/lasso)
    
#     Returns:
#     --------
#     mse_mean : Mean MSE across folds
#     mse_std : Standard deviation of MSE across folds
#     mse_folds : Array of MSE for each fold
#     """
#     # Select model
#     if method.lower() == 'ols':
#         model = LinearRegression(fit_intercept=False)
#     elif method.lower() == 'ridge':
#         model = Ridge(alpha=lam, fit_intercept=False)
#     elif method.lower() == 'lasso':
#         model = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
#     else:
#         raise ValueError(f"Unknown method: {method}. Use 'ols', 'ridge', or 'lasso'")
    
#     # Create polynomial features
#     poly = PolynomialFeatures(degree=degree)
#     X = poly.fit_transform(x.reshape(-1, 1))
    
#     # Initialize KFold
#     kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
#     # Perform cross-validation
#     scores = cross_val_score(model, X, y, 
#                             scoring='neg_mean_squared_error', 
#                             cv=kfold)
    
#     # Convert negative MSE to positive
#     mse_folds = -scores
    
#     return np.mean(mse_folds), np.std(mse_folds), mse_folds
















# np.random.seed(42)






# # =================================================================================
# #                    Bias-Variance Trade-off: Polynomial Degrees
# # =================================================================================

# # Setup
# N = 300
# degrees = range(51)
# n_bootstraps = 100
# k_folds = 5

# x = np.linspace(-1, 1, N).reshape(-1, 1)
# y = runge(x) + np.random.normal(0, 0.1, x.shape)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# boot_mse_degree = []
# boot_bias_degree = []
# boot_var_degree = []
# cv_mse_degree = []


# for deg in degrees:
#     # Bootstrap
#     error, bias, variance = bootstrap_analysis(x_train, y_train, x_test, y_test, deg, method='ols', n_bootstraps=n_bootstraps)
#     boot_mse_degree.append(error)
#     boot_bias_degree.append(bias)
#     boot_var_degree.append(variance)
    
#     # Cross-validation
#     mse_mean, _, _ = cv_sklearn(x, y, deg, k_folds=k_folds, method='ols')
#     cv_mse_degree.append(mse_mean)

# plot_error_vs_degree(degrees, boot_mse_degree, boot_bias_degree, boot_var_degree, method_name='Bootstrap OLS', logscale=True)

# plot_error_vs_degree(degrees, cv_mse_degree, method_name='CV OLS', logscale=True)


# # =================================================================================
# #                    Bias-Variance Trade-off: Dataset Sizes
# # =================================================================================

# N_list = np.arange(50, 501, 50)
# degree = 10
# n_bootstraps = 100
# k_folds = 5

# boot_errors_n = []
# boot_biases_n = []
# boot_variances_n = []
# cv_mses_n = []

# for N in N_list:
#     # Generate data
#     x = np.linspace(-1, 1, N).reshape(-1, 1)
#     y = runge(x) + np.random.normal(0, 0.1, x.shape)

#     # Bootstrap
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     error, bias, variance = bootstrap_analysis(x_train, y_train, x_test, y_test, degree, method='ols', n_bootstraps=n_bootstraps)
#     boot_errors_n.append(error)
#     boot_biases_n.append(bias)
#     boot_variances_n.append(variance)
    
#     # Cross-validation
#     mse_mean, _, _ = cv_sklearn(x, y, degree, k_folds=k_folds, method='ols')
#     cv_mses_n.append(mse_mean)


# plot_error_vs_datasize(N_list, boot_errors_n, _, _, degree=degree, method_name='Bootstrap OLS')
# plot_error_vs_datasize(N_list, cv_mses_n, degree=degree, method_name='CV OLS')









n = 500
n_boostraps = 100
degree = 15  # A quite high value, just to show.
noise = 0.1

# Make data set.
x = np.linspace(-1, 1, n).reshape(-1, 1)
y = runge(x) + np.random.normal(0, 0.1, x.shape)

# Hold out some test data that is never used in training.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Combine x transformation and model into one operation.
# Not neccesary, but convenient.
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

# The following (m x n_bootstraps) matrix holds the column vectors y_pred
# for each bootstrap iteration.
y_pred = np.empty((y_test.shape[0], n_boostraps))
for i in range(n_boostraps):
    x_, y_ = resample(x_train, y_train)

    # Evaluate the new model on the same test data each time.
    y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this 
# maintains the column vector form. Dropping this yields very unexpected results.
error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
print('Error:', error)
print('Bias^2:', bias)
print('Var:', variance)
print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

plt.plot(x[::5, :], y[::5, :], label='f(x)')
plt.scatter(x_test, y_test, label='Data points')
plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')
plt.legend()
plt.show()





import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 300
n_boostraps = 2000
maxdegree = 20

# Make data set
x = np.linspace(-1, 1, n).reshape(-1, 1)
noise_std = 0.1
noise_var = noise_std**2  # True theoretical variance = 0.01
epsilon = np.random.normal(0, noise_std, x.shape)
y = runge(x) + epsilon

error = np.zeros(maxdegree)
bias_squared = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
expected_error = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

    polydegree[degree] = degree
    error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias_squared[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
    variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))
    expected_error[degree] = variance[degree] + bias_squared[degree]   # Use true variance

plt.hlines(noise_var, 0, maxdegree, colors='k', linestyles='dashed', label='Noise level')
plt.plot(polydegree, expected_error, label='Expected error (with noise)')
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias_squared, label='Bias^2')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()