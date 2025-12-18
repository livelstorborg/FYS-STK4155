import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
def create_grid(Nx, T, alpha=0.4):
    dx = 1.0 / Nx
    dt = alpha * dx**2
    Nt = int(T / dt)

    x = jnp.linspace(0, 1, Nx + 1)
    t = jnp.linspace(0, T, Nt + 1)
    return x, t, dx, dt


# -----------------------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------------------
def u_exact(x, t):
    return jnp.sin(jnp.pi * x) * jnp.exp(-(jnp.pi**2) * t[:, None])


# -----------------------------------------------------------------------------
# Finite difference solver
# -----------------------------------------------------------------------------
def fd_solve(Nx, T, alpha):
    x, t, dx, dt = create_grid(Nx, T, alpha)

    # create_grid returns t with length Nt + 1 (including t=0).
    # The number of timesteps is therefore len(t) - 1. Use that
    # for the time-stepping loop and allocate u with shape
    # (Nt+1, Nx+1) so that u[n] corresponds to time index n.
    Nt = len(t) - 1

    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))

    for n in range(Nt):
        u = u.at[n + 1, 1:Nx].set(
            u[n, 1:Nx]
            + alpha * (u[n, 2 : Nx + 1] - 2 * u[n, 1:Nx] + u[n, 0 : Nx - 1])
        )
        u = u.at[n + 1, 0].set(0.0)
        u = u.at[n + 1, Nx].set(0.0)

    return u, x, t
