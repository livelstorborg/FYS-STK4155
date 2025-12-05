import jax
import jax.numpy as jnp
import flax.nnx as nnx


def pde_residual(model, xt, nu):
    def u_single(z):
        return model(z[None, :])[0, 0]

    jac = jax.vmap(jax.grad(u_single))(xt)
    u_x = jac[:, 0]
    u_t = jac[:, 1]

    def u_x_single(z):
        return jax.grad(u_single)(z)[0]

    u_xx = jax.vmap(jax.grad(u_x_single))(xt)[:, 0]

    return (u_t - nu * u_xx) ** 2


def loss_fn(
    model,
    x_int,
    t_int,
    x_bc,
    t_bc,
    x_ic,
    t_ic,
    y_ic,
    lambda_ic: float = 500.0,
    lambda_bc: float = 200.0,
    nu: float = 1.0,
):
    xt_int = jnp.concatenate([x_int, t_int], axis=1)
    loss_pde = pde_residual(model, xt_int, nu).mean()

    xt_bc = jnp.concatenate([x_bc, t_bc], axis=1)
    u_bc = model(xt_bc)
    loss_bc = jnp.mean(u_bc**2)

    xt_ic = jnp.concatenate([x_ic, t_ic], axis=1)
    u_ic_pred = model(xt_ic)
    loss_ic = jnp.mean((u_ic_pred - y_ic) ** 2)

    return loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
