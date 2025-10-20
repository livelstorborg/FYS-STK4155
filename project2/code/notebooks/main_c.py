import sys
sys.path.insert(0, '/Users/livestorborg/Desktop/FYS-STK4155/project2/code')

import numpy as np
from autograd import grad
import autograd.numpy as anp
from src.neural_network import NeuralNetwork
from src.activations import Sigmoid, Linear
from src.losses import MSE

from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import torch

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)



# =========================================================
#          Verify MSE gradient with autograd
# =========================================================
y_true = np.array([[1.0], [2.0], [3.0]])
y_pred = np.array([[1.1], [1.9], [3.2]])

loss_fn = MSE()
our_loss = loss_fn.forward(y_true, y_pred)
our_grad = loss_fn.backward(y_true, y_pred)

def mse_auto(y_pred, y_true): return anp.mean((y_true - y_pred)**2)
auto_grad_fn = grad(mse_auto, 0)
auto_grad = auto_grad_fn(y_pred, y_true)

print(f'Our gradient: {our_grad}')
print(f'Autograd gradient: {auto_grad}')
diff = np.max(np.abs(our_grad - auto_grad))
print(f'Gradient computation: {diff < 1e-6}')





# =========================================================
#     Verify NN backprop gradients with autograd
# =========================================================
nn = NeuralNetwork(2, [3, 1], [Sigmoid(), Linear()], MSE(), seed=SEED)
X = np.array([[0.5, 0.3], [0.2, 0.8]])
y = np.array([[1.0], [0.5]])
our_grads = nn.compute_gradient(X, y)

def nn_loss_auto(layers, X, y):
    a = X
    for (W, b), act in zip(layers, nn.activations):
        z = anp.dot(a, W.T) + b
        a = 1 / (1 + anp.exp(-z)) if isinstance(act, Sigmoid) else z
    return anp.mean((y - a)**2)

auto_grad_fn = grad(nn_loss_auto, 0)
auto_grads = auto_grad_fn(nn.layers, X, y)
wdiff = np.max(np.abs(our_grads[0][0] - auto_grads[0][0]))
bdiff = np.max(np.abs(our_grads[0][1] - auto_grads[0][1]))
ok = (wdiff < 1e-6 and bdiff < 1e-6)
print(f"Backprop check: {wdiff < 1e-6 and bdiff < 1e-6}")




# =========================================================
#   Compare results vs Scikit-Learn, TensorFlow, PyTorch
# =========================================================
x = np.linspace(-1, 1, 300).reshape(-1, 1)
y = 1 / (1 + 25 * x**2)



# --- Scikit-Learn ---
sk_model = MLPRegressor(hidden_layer_sizes=(5,), activation='logistic',
                        solver='lbfgs', max_iter=500, random_state=SEED)
sk_model.fit(x, y.ravel())
mse_sk = np.mean((sk_model.predict(x).reshape(-1, 1) - y)**2)



# --- TensorFlow ---
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer='adam', loss='mse')
tf_model.fit(x, y, epochs=100, verbose=0)
mse_tf = tf_model.evaluate(x, y, verbose=0)



# --- PyTorch ---
X_t, Y_t = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
model_t = torch.nn.Sequential(torch.nn.Linear(1, 5), torch.nn.Sigmoid(), torch.nn.Linear(5, 1))
opt = torch.optim.Adam(model_t.parameters(), lr=0.01)
loss_fn_t = torch.nn.MSELoss()
for _ in range(200):
    opt.zero_grad()
    loss = loss_fn_t(model_t(X_t), Y_t)
    loss.backward()
    opt.step()
mse_torch = float(loss_fn_t(model_t(X_t), Y_t))



print(f"Scikit-Learn MSE: {mse_sk:.4e}")
print(f"TensorFlow  MSE: {mse_tf:.4e}")
print(f"PyTorch     MSE: {mse_torch:.4e}")
print("=" * 60)