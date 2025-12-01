import jax
import jax.numpy as jnp
from flax import nnx


@nnx.jit
def pde_residual(model, x, t):
    xt = jnp.hstack([x, t])

    def u_scalar(z):
        return model(z[None, :])[0, 0]

    def dudt(z):
        return jax.grad(u_scalar, argnums=0)(z)[1]

    def d2udx2(z):
        def dudx(zz):
            return jax.grad(u_scalar, argnums=0)(zz)[0]

        return jax.grad(dudx)(z)[0]

    u_t = jax.vmap(dudt)(xt)
    u_xx = jax.vmap(d2udx2)(xt)

    return (u_t - u_xx) ** 2


def loss_fn(
    model,
    x_int,
    t_int,
    x_bc,
    t_bc,
    x_ic,
    t_ic,
    y_ic,
    lambda_ic: float = 1.0,
    lambda_bc: float = 1.0,
):
    loss_pde = pde_residual(model, x_int, t_int).mean()

    u_bc = model(jnp.hstack([x_bc, t_bc]))
    loss_bc = (u_bc**2).mean()

    u_ic_pred = model(jnp.hstack([x_ic, t_ic]))
    loss_ic = ((u_ic_pred - y_ic) ** 2).mean()

    return loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
