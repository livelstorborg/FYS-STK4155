import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from .losses import loss_fn
from .model import MLP


def sample_points(
    key,
    N_int: int = 512,
    N_bc: int = 64,
    N_ic: int = 64,
    T: float = 0.5,
):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=1.0)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    x_bc = jnp.vstack(
        [
            jnp.zeros((N_bc, 1)),
            jnp.ones((N_bc, 1)),
        ]
    )
    t_bc = jax.random.uniform(k3, (2 * N_bc, 1), minval=0.0, maxval=T)
    x_ic = jax.random.uniform(k4, (N_ic, 1), minval=0.0, maxval=1.0)
    t_ic = jnp.zeros((N_ic, 1))
    u_ic = jnp.sin(jnp.pi * x_ic)

    return x_int, t_int, x_bc, t_bc, x_ic, t_ic, u_ic


def train_pinn(
    layers=[2, 64, 64, 1],
    activations=None,
    steps=5000,
    T=0.5,
    lambda_ic=10.0,
    lambda_bc=10.0,
    lr=1e-3,
    seed=0,
):
    if activations is None:
        activations = [jax.nn.tanh] * (len(layers) - 2)

    main_key = jax.random.PRNGKey(seed)
    key_model, key_data = jax.random.split(main_key)

    model = MLP(layers, activations, key=key_model)

    model = MLP(layers=layers, activations=activations, key=jax.random.PRNGKey(0))
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    x_int, t_int, x_bc, t_bc, x_ic, t_ic, u_ic = sample_points(key_data, T=T)
    losses = []

    L = nnx.jit(
        lambda m: loss_fn(
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
    )

    for _ in range(steps):
        loss, grads = nnx.value_and_grad(L)(model)
        opt.update(model, grads)
        losses.append(loss)

    return model, jnp.array(losses)
