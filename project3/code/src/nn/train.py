import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from .losses import loss_fn
from .model import MLP


def sample_points(N_int=512, N_bc=64, N_ic=64, T=0.5):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=1.0)
    t_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=T)

    x_bc = jnp.vstack([jnp.zeros((N_bc, 1)), jnp.ones((N_bc, 1))])
    t_bc = jax.random.uniform(k2, (2 * N_bc, 1), minval=0.0, maxval=T)

    x_ic = jax.random.uniform(k3, (N_ic, 1), minval=0.0, maxval=1.0)
    t_ic = jnp.zeros((N_ic, 1))
    u_ic = jnp.sin(jnp.pi * x_ic)

    return x_int, t_int, x_bc, t_bc, x_ic, t_ic, u_ic


def train_pinn(
    layers=[2, 64, 64, 1],
    steps=5000,
    T=0.5,
    lambda_ic=10.0,
    lambda_bc=10.0,
    lr=1e-3,
):
    model = MLP(layers, key=jax.random.PRNGKey(0))
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    x_int, t_int, x_bc, t_bc, x_ic, t_ic, u_ic = sample_points(T=T)
    losses = []

    def L(m):
        return loss_fn(
            m,
            x_int,
            t_int,
            x_bc,
            t_bc,
            x_ic,
            t_ic,
            u_ic,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
        )

    for _ in range(steps):
        loss, grads = nnx.value_and_grad(L)(model)
        opt.update(model, grads)
        losses.append(loss)

    return model, jnp.array(losses)
