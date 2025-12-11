import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from .losses import loss_fn, loss_fn_hard
from .model import MLP, MLP_HardBC


def sample_points(
    key,
    N_int=512,
    N_bc=64,
    N_ic=256,
    T=0.5,
    L=1.0,
):
    k1, k2, k3 = jax.random.split(key, 3)

    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    t_bc = jax.random.uniform(k3, (N_bc, 1), minval=0.0, maxval=T)
    x_bc = jnp.vstack([jnp.zeros((N_bc, 1)), L * jnp.ones((N_bc, 1))])
    t_bc = jnp.vstack([t_bc, t_bc])

    x_ic = jnp.linspace(0.0, L, N_ic).reshape(-1, 1)
    t_ic = jnp.zeros_like(x_ic)
    y_ic = jnp.sin(jnp.pi * x_ic / L)

    return x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic


def train_pinn(
    layers=[2, 64, 64, 1],
    activations=None,
    steps=5000,
    N_int=1000,
    N_bc=500,
    N_ic=200,
    T=0.5,
    L=1.0,
    lambda_ic=1.0,
    lambda_bc=1.0,
    nu=1.0,
    c=0.0,
    lr=5e-4,
    seed=0,
    use_hard_bc=True,
):
    if activations is None:
        activations = [jax.nn.tanh] * (len(layers) - 2)

    main_key = jax.random.PRNGKey(seed)
    key_model, key_loop = jax.random.split(main_key)

    # Create model based on BC type
    if use_hard_bc:
        model = MLP_HardBC(layers, activations, key=key_model)
    else:
        model = MLP(layers, activations, key=key_model)
    
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=1000,
        decay_rate=0.95,
    )
    opt = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    losses = []
    key = key_loop

    # ============ JIT-COMPILED TRAINING STEP ============
    @jax.jit
    def train_step_soft(model, opt, key):
        """JIT-compiled step for soft BC"""
        key, subkey = jax.random.split(key)
        x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic = sample_points(
            subkey, N_int, N_bc, N_ic, T, L
        )

        def loss_func(m):
            return loss_fn(
                m, x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic,
                lambda_ic=lambda_ic, lambda_bc=lambda_bc, nu=nu
            )

        loss, grads = nnx.value_and_grad(loss_func)(model)
        opt.update(model, grads)
        return model, opt, key, loss

    @jax.jit
    def train_step_hard(model, opt, key):
        """JIT-compiled step for hard BC"""
        key, subkey = jax.random.split(key)
        x_int, t_int, _, _, _, _, _ = sample_points(
            subkey, N_int, N_bc, N_ic, T, L
        )

        def loss_func(m):
            return loss_fn_hard(m, x_int, t_int, nu=nu)

        loss, grads = nnx.value_and_grad(loss_func)(model)
        opt.update(model, grads)
        return model, opt, key, loss

    # Choose which training step to use
    train_step = train_step_hard if use_hard_bc else train_step_soft

    # ============ TRAINING LOOP ============
    for step in range(steps):
        model, opt, key, loss = train_step(model, opt, key)
        losses.append(loss)

        if step % 500 == 0 or step == steps - 1:
            print(f"[step {step:4d}] loss = {float(loss):.3e}")

    return model, jnp.array(losses)