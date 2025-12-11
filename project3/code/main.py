import jax.numpy as jnp
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
    """
    Nx = 100
    T = 0.5

    _, x, t = fd_solve(Nx=Nx, T=T, alpha=0.4)
    u_true = u_exact(x, t)

    t1, t2 = 0.07, 0.30
    i1 = jnp.argmin(jnp.abs(t - t1))
    i2 = jnp.argmin(jnp.abs(t - t2))

    model, losses = train_pinn(
        layers=[2, 128, 128, 128, 128, 1],
        steps=1000,
        N_int=1000,
        N_ic=200,
        lambda_ic=1.0,
        lambda_bc=1.0,
        lr=1e-3,
        nu=1.0,
    )

    # Ensure the PINN evaluation uses the same temporal grid as the FD solver
    # so that time indices (i1, i2) align between `u_true` and `u_pinn`.
    u_pinn, _, _, _ = compare_nn_and_exact(
        model, Nx=Nx, Nt=len(t) - 1, T=T, return_only=True
    )

    plot_solution(
        x,
        u_pinn[i1],
        u_true[i1],
        title=rf"PINN vs exact at t_1 \approx {float(t[i1]):.3f}",
    )

    plot_solution(
        x,
        u_pinn[i2],
        u_true[i2],
        title=rf"PINN vs exact at t_2 \approx {float(t[i2]):.3f}",
    )

    compare_nn_and_exact(model, Nx=Nx, T=T)

    plot_training_loss(losses)
    """


# ---------- Experiment model architecutre (Part d)----------

# results = run_architecture_sweep(
#     hidden_widths=[32, 64, 128],
#     num_hidden_layers=[2, 3, 4],
#     activation_fns={
#         'tanh': jnn.tanh,
#         'sin': jnp.sin,
#         'GeLU': jnn.gelu,
#         'swish': jnn.swish,
#         'ReLU': jnn.relu,
#     },
#     T=0.5,
#     steps=10000,
#     N_int=1000,
#     N_bc=500,
#     N_ic=200,
#     lr=5e-4,
#     seeds=(4, 2, 16, 8, 29, 3, 21, 9, 0, 42),
#     save_to_csv=False,
#     use_pre_computed=True,
#     data_dir="data",
# )

# print(results)

# plot_all_heatmaps(results)

import sys
import os

sys.path.append(os.path.abspath("../"))

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
from src.fd_scheme import fd_solve
from src.analytical import u_exact
from src.plotting import (
    plot_solution,
    plot_training_loss,
    plot_all_heatmaps,
    plot_scheme_errors_t1,
    plot_scheme_errors_t2,
    plot_3d_surface,
    subplot_3d_surface,
)
from src.nn.model import MLP
from src.nn.train import train_pinn
from src.nn.evaluation import compare_nn_and_exact, test_explicit_scheme, evaluate_pinn
from src.experiment import run_architecture_sweep

u_fd, x, t = fd_solve(Nx=100, T=0.5, alpha=0.4)
model, losses = train_pinn(
    steps=5000, layers=[2, 32, 32, 32, 1], activations=[jnn.silu, jnn.silu, jnn.silu]
)
u_nn = evaluate_pinn(model, x, t)
u_true = u_exact(x, t)


error_nn = np.abs(u_nn - u_true)
error_fd = np.abs(u_fd - u_true)

rotations = [45, 135, 225, 315]

pinn_surfaces = [error_nn for _ in rotations]
fd_surfaces = [error_fd for _ in rotations]

subplot_3d_surface(
    x,
    t,
    pinn_surfaces,
    elev=20,
    azims=rotations,
    save_path="code/figs/error_surface_pinn.pdf",
)

subplot_3d_surface(
    x,
    t,
    fd_surfaces,
    elev=20,
    azims=rotations,
    save_path="code/figs/error_surface_fd.pdf",
)
