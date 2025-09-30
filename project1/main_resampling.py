import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.linear_model import Lasso
from src.utils import polynomial_features, scale_data, runge, OLS_parameters, Ridge_parameters


np.random.seed(42)

N = 300
lam = 1e-5
eta = 1e-1
num_iters = 10000

x = np.linspace(-1, 1, N)
random_noise = np.random.normal(0, 0.1, N)
y_true = runge(x)   
y_noise = y_true + random_noise

degrees = range(1, 16)

# For plotting mse_train and mse_test vs degree
mse_train = np.zeros(len(degrees))
mse_test = np.zeros(len(degrees))

# For plotting bias-variance decomposition vs degree



# For plotting the cv 



for degree in degrees:

    X = polynomial_features(x, degree)
    

   