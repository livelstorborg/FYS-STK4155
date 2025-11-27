import jax.numpy as jnp


def u_exact(x, t):
    return jnp.sin(jnp.pi * x) * jnp.exp(-(jnp.pi**2) * t[:, None])
