import sys
sys.path.insert(0, '/Users/livestorborg/Desktop/FYS-STK4155/project2/code')

import numpy as np
from sklearn.model_selection import train_test_split
from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, Linear
from src.losses import MSE
from src.optimizers import GD, RMSprop, Adam
from src.training import train
from src.metrics import mse
from src.utils import runge, polynomial_features, scale_data, OLS_parameters, inverse_scale_y

SEED = 42
np.random.seed(SEED)

# ========================================
#                 OLS
# ========================================
N = 300
x = np.linspace(-1, 1, N)
y_true = runge(x)
y_noise = y_true + np.random.normal(0, 0.01, N)

X_poly = polynomial_features(x, p=14, intercept=False)

# Split
X_train_poly, X_test_poly, y_train, y_test = train_test_split(
    X_poly, y_noise, test_size=0.2, random_state=SEED
)

# OLS
theta_ols = OLS_parameters(X_train_poly, y_train)
y_pred_ols = X_test_poly @ theta_ols
ols_mse = mse(y_test.reshape(-1, 1), y_pred_ols.reshape(-1, 1))


# ========================================
#         Neural Network
# ========================================
# Split raw x for NN
X_train_raw, X_test_raw, y_train_nn, y_test_nn = train_test_split(
    x.reshape(-1, 1), y_noise.reshape(-1, 1), 
    test_size=0.2, random_state=SEED
)

# Scale for NN
X_train_s, y_train_s, X_mean, X_std, y_mean = scale_data(X_train_raw, y_train_nn)
X_test_s, y_test_s, _, _, _ = scale_data(X_test_raw, y_test_nn, X_mean, X_std, y_mean)

# Compute y_test_real once (used in all loops)
y_test_real = inverse_scale_y(y_test_s, y_mean)




eta_gd = np.logspace(-3, 1, 20)     
eta_rms = np.logspace(-3, -1, 20)    
eta_adam = np.logspace(-3, -1, 20)




# ============ Plain GD (full batch)=============
print("\nTesting GD with different learning rates...")
best_gd_eta = None
best_gd_mse = float('inf')

for eta in eta_gd:
    nn_gd = NeuralNetwork(1, [50, 1], [Sigmoid(), Linear()], MSE(), seed=SEED)
    train(nn_gd, X_train_s, y_train_s, X_test_s, y_test_s, GD(eta), 
          epochs=500, batch_size=len(X_train_s), verbose=False, seed=SEED)  # ← full batch
    y_pred_gd_s = nn_gd.predict(X_test_s)
    y_pred_gd = inverse_scale_y(y_pred_gd_s, y_mean)
    gd_mse = mse(y_test_real, y_pred_gd)

    print(f"  eta={eta:.4f}  MSE={gd_mse:.6f}")
    
    if gd_mse < best_gd_mse:  # ← Fixed variable names
        best_gd_mse = gd_mse
        best_gd_eta = eta

# ============ RMSprop (test different etas) =============
print("\nTesting RMSprop with different learning rates...")
best_rms_eta = None
best_rms_mse = float('inf')

for eta in eta_rms:
    nn_rms = NeuralNetwork(1, [50, 1], [Sigmoid(), Linear()], MSE(), seed=SEED)
    train(nn_rms, X_train_s, y_train_s, X_test_s, y_test_s, RMSprop(eta), 
          epochs=500, batch_size=32, verbose=False, seed=SEED)
    y_pred_rms_s = nn_rms.predict(X_test_s)
    y_pred_rms = inverse_scale_y(y_pred_rms_s, y_mean)
    rms_mse = mse(y_test_real, y_pred_rms)
    
    print(f"  eta={eta:.4f}  MSE={rms_mse:.6f}")
    
    if rms_mse < best_rms_mse:
        best_rms_mse = rms_mse
        best_rms_eta = eta

# ============ Adam (test different etas) =============
print("\nTesting Adam with different learning rates...")
best_adam_eta = None
best_adam_mse = float('inf')

for eta in eta_adam:
    nn_adam = NeuralNetwork(1, [50, 1], [Sigmoid(), Linear()], MSE(), seed=SEED)
    train(nn_adam, X_train_s, y_train_s, X_test_s, y_test_s, Adam(eta), 
          epochs=500, batch_size=32, verbose=False, seed=SEED)
    y_pred_adam_s = nn_adam.predict(X_test_s)
    y_pred_adam = inverse_scale_y(y_pred_adam_s, y_mean)
    adam_mse = mse(y_test_real, y_pred_adam)
    
    print(f"  eta={eta:.4f}  MSE={adam_mse:.6f}")
    
    if adam_mse < best_adam_mse:
        best_adam_mse = adam_mse
        best_adam_eta = eta

# ========================================
# Results
# ========================================
print("\n" + "="*50)
print("PART B) RESULTS")
print("="*50)
print(f"OLS (deg 14):      {ols_mse:.6f}")
print(f"NN + GD:           {best_gd_mse:.6f}  (eta={best_gd_eta:.4f})")
print(f"NN + RMSprop:      {best_rms_mse:.6f}  (eta={best_rms_eta:.4f})")
print(f"NN + Adam:         {best_adam_mse:.6f}  (eta={best_adam_eta:.4f})")
print("="*50)