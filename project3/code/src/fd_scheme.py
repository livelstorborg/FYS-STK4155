import jax.numpy as jnp
from src.grids import create_grid


def fd_solve(Nx, T, alpha):
    x, t, dx, dt = create_grid(Nx, T, alpha)
    Nt = len(t)

    u = jnp.zeros((Nt, Nx + 1))
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))

    for n in range(Nt - 1):
        u = u.at[n + 1, 1:Nx].set(
            u[n, 1:Nx] + alpha * (u[n, 2 : Nx + 1] - 2 * u[n, 1:Nx] + u[n, 0 : Nx - 1])
        )
        u = u.at[n + 1, 0].set(0.0)
        u = u.at[n + 1, Nx].set(0.0)
    return u, x, t
