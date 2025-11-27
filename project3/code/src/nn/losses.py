import jax
import jax.numpy as jnp


def pde_residual(model, x, t):
    xt = jnp.hstack([x, t])

    def u_scalar(z):
        return model(z)[0]

    grad_u = jax.vmap(jax.grad(u_scalar))(xt)
    hess_u = jax.vmap(jax.hessian(u_scalar))(xt)

    u_t = grad_u[:, 1]
    u_xx = hess_u[:, 0, 0]

    return (u_t - u_xx) ** 2


def loss_fn(
    model, x_int, t_int, x_bc, t_bc, x_ic, t_ic, y_ic, lambda_ic=1.0, lambda_bc=1.0
):
    loss_pde = pde_residual(model, x_int, t_int).mean()
    loss_bc = model(jnp.hstack([x_bc, t_bc])) ** 2
    loss_ic = (model(jnp.hstack([x_ic, t_ic])) - y_ic) ** 2
    return loss_pde + lambda_ic * loss_ic.mean() + lambda_bc * loss_bc.mean()
