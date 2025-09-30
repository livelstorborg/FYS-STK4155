import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.utils import polynomial_features, scale_data, runge, OLS_parameters
from src.regression import RegressionAnalysis


def bootstrap_bias_variance(X_train, X_test, y_train, y_test, n_bootstraps):
    """Perform bootstrap resampling to calculate bias-variance decomposition."""

    n_test = len(X_test)
    y_pred_bootstrap = np.zeros((n_test, n_bootstraps))
    
    for i in range(n_bootstraps):
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        theta_boot = OLS_parameters(X_boot, y_boot)
        y_pred_bootstrap[:, i] = X_test @ theta_boot
    

    y_pred_mean = np.mean(y_pred_bootstrap, axis=1)
    bias_squared = np.mean((y_test - y_pred_mean) ** 2)
    variance = np.mean(np.var(y_pred_bootstrap, axis=1))
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'y_pred_mean': y_pred_mean,
        'y_pred_all': y_pred_bootstrap
    }


# ==================================================================================================
#                          MAIN ANALYSIS
# ==================================================================================================

np.random.seed(42)

N = 100
x = np.linspace(-1, 1, N)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise
degrees = range(1, 21)

train_mse = np.zeros(len(degrees))
test_mse = np.zeros(len(degrees))
bias_squared = np.zeros(len(degrees))
variance = np.zeros(len(degrees))

n_bootstraps = 100

for degree in degrees:
    
    X = polynomial_features(x, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noise, test_size=0.2, random_state=42)
    
    X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train, y_train)
    X_test_s, y_test_s, _, _, _ = scale_data(X_test, y_test, X_mean, X_std, y_mean)
    
    # Fit OLS for train/test MSE
    data = [X_train_s, X_test_s, y_train_s, y_test_s, None, None, y_mean]
    analysis = RegressionAnalysis(data, degree=degree, full_dataset=False)
    analysis.fit(models='ols', opts='analytical')
    
    train_mse[degree-1] = analysis.get_metric('ols', 'analytical', 'train_mse')
    test_mse[degree-1] = analysis.get_metric('ols', 'analytical', 'test_mse')
    y_test_unscaled = y_test_s + y_mean
    
    results = bootstrap_bias_variance(X_train_s, X_test_s, y_train_s, y_test_unscaled, n_bootstraps=n_bootstraps)
    
    bias_squared[degree-1] = results['bias_squared']
    variance[degree-1] = results['variance']
    



# ==================================================================================================
#                          PLOTS
# ==================================================================================================


plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mse, label='Training Sample')
plt.plot(degrees, test_mse, label='Test Sample')
plt.xlabel('Model Complexity', fontsize=14)
plt.ylabel('Prediction Error', fontsize=14)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(degrees, bias_squared, label='Bias²')
plt.plot(degrees, variance, label='Variance')
plt.plot(degrees, test_mse, label='Test MSE')
plt.xlabel('Model Complexity', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Bias-Variance Trade-off', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

