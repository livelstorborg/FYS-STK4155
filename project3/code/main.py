import jax.nn as jnn
from src.fd_scheme import fd_solve
from src.analytical import u_exact
from src.plotting import plot_solution, plot_training_loss, plot_all_heatmaps
from src.nn.model import MLP
from src.nn.train import train_pinn
from src.nn.evaluation import compare_nn_and_exact, test_explicit_scheme
from src.experiment import run_architecture_sweep


if __name__ == "__main__":
    # ---------- Part b (test scheme with t1 and t2) ----------
    # test_explicit_scheme(Nx=10, T=0.5, alpha=0.4, t1=0.07, t2=0.3)
    # test_explicit_scheme(Nx=100, T=0.5, alpha=0.4, t1=0.07, t2=0.3)

    # ---------- Part c ----------
    Nx = 100
    T = 0.5
    u_num, x, t = fd_solve(Nx=Nx, T=T, alpha=0.4)
    u_true = u_exact(x, t)
    plot_solution(x, u_num[-1], u_true[-1], title="t=0.5")

    model, losses = train_pinn(layers=[2, 32, 32, 1], steps=5000)
    compare_nn_and_exact(model, Nx=Nx, T=T)

    model, losses = train_pinn(layers=[2, 32, 32, 1], steps=5000)
    plot_training_loss(losses)
    compare_nn_and_exact(model, Nx=Nx, T=T)

# ---------- Experiment model architecutre ----------

# results = run_architecture_sweep(
#     hidden_widths=[16, 32, 64],
#     num_hidden_layers=[1, 2, 3],
#     activation_fns={
#         "tanh": jnn.tanh,
#         "sigmoid": jnn.sigmoid,
#         "relu": jnn.relu,
#         "leaky_relu": jnn.leaky_relu,
#     },
#     steps=3000,
#     seeds=(0,),
#     T=0.5,
#     Nx_eval=100,
#     Nt_eval=100,
# )
# plot_all_heatmaps(results)
