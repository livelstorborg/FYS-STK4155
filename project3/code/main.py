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

# Finite Difference scheme
u_fd, x, t = fd_solve(Nx=100, T=0.5, alpha=0.4)

u_true = u_exact(x, t)

# SiLU Activation
model_silu, losses_silu = train_pinn(
    steps=10000,
    layers=[2, 32, 32, 32, 1],
    activations=[jnn.silu, jnn.silu, jnn.silu],
    N_int=10000,
    N_ic=1000,
    N_bc=1000,
)
# Use coarser temporal grid for PINN evaluation (no stability constraint)
# t_pinn = jnp.linspace(0, 0.5, 101)
u_nn_silu = evaluate_pinn(model_silu, x, t)

# Sinusoidal Activation
model_sin, losses_sin = train_pinn(
    steps=10000,
    layers=[2, 32, 32, 32, 1],
    activations=[jnp.sin, jnp.sin, jnp.sin],
    N_int=10000,
    N_ic=1000,
    N_bc=1000,
)
u_nn_sin = evaluate_pinn(model_sin, x, t)


# Error surfaces
error_fd = np.abs(u_fd - u_true)
u_true_pinn = u_exact(x, t)
error_nn_silu = np.abs(u_nn_silu - u_true_pinn)
error_nn_sin = np.abs(u_nn_sin - u_true_pinn)


# Plotting
rotations = [45, 135, 225, 315]

fd_surfaces = [error_fd for _ in rotations]
pinn_surfaces_silu = [error_nn_silu for _ in rotations]


# Surface plots of solutions
plot_3d_surface(
    x,
    t,
    u_fd,
    elev=20,
    azim=45,
    save_path="figs/fd_solution.pdf",
    title="Finite Difference Solution",
)

plot_3d_surface(
    x,
    t,
    u_nn_silu,
    elev=20,
    azim=45,
    save_path="figs/pinn_solution_silu.pdf",
    title="PINN Solution (SiLU Activation)",
)

plot_3d_surface(
    x,
    t,
    u_nn_sin,
    elev=20,
    azim=45,
    save_path="figs/pinn_solution_sin.pdf",
    title="PINN Solution (Sinusoidal Activation)",
)

subplot_3d_surface(
    x,
    t,
    fd_surfaces,
    elev=20,
    azims=rotations,
    save_path="figs/fd_error_surfaces.pdf",
    title="Absolute Error of Finite Difference Scheme",
)

subplot_3d_surface(
    x,
    t,
    pinn_surfaces_silu,
    elev=20,
    azims=rotations,
    save_path="figs/pinn_error_surfaces_silu.pdf",
    title="Absolute Error of PINN Solution (SiLU)",
)

subplot_3d_surface(
    x,
    t,
    [error_nn_sin for _ in rotations],
    elev=20,
    azims=rotations,
    save_path="figs/pinn_error_surfaces_sin.pdf",
    title="Absolute Error of PINN Solution (Sinusoidal)",
)
