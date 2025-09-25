import numpy as np
from sklearn.model_selection import train_test_split

from src.plotting import solution_comparison
from src.regression import RegressionAnalysis
from src.utils import polynomial_features, scale_data, runge


N = 500
degree = 8  
lam = 1e-2  
eta = 1e-2  
num_iters = 10000


x = np.linspace(-1, 1, N)
np.random.seed(42)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise                 
X = polynomial_features(x, degree)           
 


# =============================================================================
#                          FULL DATASET ANALYSIS
# =============================================================================
X_norm, y_centered, _, _, y_mean = scale_data(X, y_noise) 
data = [X_norm, y_centered, y_mean]
analysis = RegressionAnalysis(
    data, 
    degree=degree, 
    lam=lam, eta=eta, 
    num_iters=num_iters,
    full_dataset=True  
)

analysis.fit_many(models=('ols', 'ridge'), 
                  opts=('analytical', 'gd'))

solutions = [
    analysis.runs[('ols', 'analytical')]['y_pred_test'],
    analysis.runs[('ols', 'gd')]['y_pred_test'], 
    analysis.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis.runs[('ridge', 'gd')]['y_pred_test'],
    x        
]

solution_comparison(x, y_noise, y_true, solutions=solutions, sample_size=N, degree=degree, lam=lam, test=False)






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

analysis_test.fit_many(models=('ols', 'ridge'), 
                      opts=('analytical', 'gd'))


solutions_test = [
    analysis_test.runs[('ols', 'analytical')]['y_pred_test'],
    analysis_test.runs[('ols', 'gd')]['y_pred_test'], 
    analysis_test.runs[('ridge', 'analytical')]['y_pred_test'],
    analysis_test.runs[('ridge', 'gd')]['y_pred_test'],
    x_test       
]

solution_comparison(x, y_noise, y_true, solutions=solutions_test, sample_size=N, degree=degree, lam=lam, test=True)
