import jax.numpy as jnp


def create_grid(Nx, T, alpha=0.4):
    dx = 1.0 / Nx
    dt = alpha * dx**2
    Nt = int(T / dt)

    x = jnp.linspace(0, 1, Nx + 1)
    t = jnp.linspace(0, T, Nt + 1)
    return x, t, dx, dt
