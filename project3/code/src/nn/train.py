import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from .losses import loss_fn
from .model import MLP


def sample_points(
    key,
    N_int=512,
    N_bc=64,
    N_ic=256,
    T=0.5,
    L=1.0,
):
    k1, k2, k3 = jax.random.split(key, 3)

    # Interior collocation points
    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    # Boundary conditions at x = 0 and x = L
    t_bc = jax.random.uniform(k3, (N_bc, 1), minval=0.0, maxval=T)
    x_bc = jnp.vstack(
        [
            jnp.zeros((N_bc, 1)),  # x = 0
            L * jnp.ones((N_bc, 1)),  # x = L
        ]
    )
    t_bc = jnp.vstack([t_bc, t_bc])

    # Initial condition at t = 0
    x_ic = jnp.linspace(0.0, L, N_ic).reshape(-1, 1)
    t_ic = jnp.zeros_like(x_ic)
    y_ic = jnp.sin(jnp.pi * x_ic / L)

    return x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic


def train_pinn(
    layers=[2, 64, 64, 1],
    activations=None,
    steps=5000,
    N_int=512,
    N_bc=64,
    N_ic=256,
    T=0.5,
    L=1.0,
    lambda_ic=200.0,
    lambda_bc=10.0,
    nu=1.0,
    c=0.0,
    lr=1e-3,
    seed=0,
):
    if activations is None:
        activations = [jax.nn.tanh] * (len(layers) - 2)

    main_key = jax.random.PRNGKey(seed)
    key_model, key_loop = jax.random.split(main_key)

    model = MLP(layers, activations, key=key_model)
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    losses = []
    key = key_loop

    for step in range(steps):
        key, subkey = jax.random.split(key)
        x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic = sample_points(
            subkey,
            N_int=N_int,
            N_bc=N_bc,
            N_ic=N_ic,
            T=T,
            L=L,
        )

        def loss_func(m):
            return loss_fn(
                m,
                x_int,
                t_int,
                x_bc,
                t_bc,
                x_ic,
                t_ic,
                y_ic,
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc,
                nu=nu,
            )

        loss, grads = nnx.value_and_grad(loss_func)(model)
        opt.update(model, grads)

        losses.append(loss)

        if step % 500 == 0 or step == steps - 1:
            print(f"[step {step:4d}] loss = {float(loss):.3e}")

    return model, jnp.array(losses)
