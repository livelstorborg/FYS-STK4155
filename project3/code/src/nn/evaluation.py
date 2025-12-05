import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.analytical import u_exact
from src.fd_scheme import fd_solve
from src.plotting import plot_solution


# ---------- Part b test ----------
def test_explicit_scheme(Nx=100, T=0.5, alpha=0.4, t1=0.07, t2=0.30):
    u_num, x, t = fd_solve(Nx=Nx, T=T, alpha=alpha)

    i1 = jnp.argmin(jnp.abs(t - t1))
    i2 = jnp.argmin(jnp.abs(t - t2))

    u_true = u_exact(x, t)

    plot_solution(
        x,
        u_num[i1],
        u_true[i1],
        title=rf"$t_1$ $\approx$ {float(t[i1]):.3f}, $\Delta$x = {1 / Nx:.2f}",
    )
    plot_solution(
        x,
        u_num[i2],
        u_true[i2],
        title=rf"$t_2$ $\approx$ {float(t[i2]):.3f}, $\Delta$x = {1 / Nx:.2f}",
    )


# ---------- Part c ----------
def compare_nn_and_exact(model, Nx=200, Nt=100, T=1.0, return_only=False):
    x = jnp.linspace(0, 1, Nx + 1)
    t = jnp.linspace(0, T, Nt + 1)

    X, Tt = jnp.meshgrid(x, t)

    xt = jnp.stack([X.ravel(), Tt.ravel()], axis=1)
    u_pred_flat = model(xt).reshape(Nt + 1, Nx + 1)
    u_true = u_exact(x, t)
    error = jnp.abs(u_pred_flat - u_true)

    if return_only:
        return u_pred_flat, u_true, x, t

    plt.figure(figsize=(7, 5))
    plt.imshow(u_pred_flat, extent=[0, 1, 0, T], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("PINN prediction $u_\\theta(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(u_true, extent=[0, 1, 0, T], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("Analytical solution $u(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(error, extent=[0, 1, 0, T], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("Absolute error $|u_\\theta - u|$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(x, u_pred_flat[-1], label="PINN")
    plt.plot(x, u_true[-1], "--", label="Exact")
    plt.title(f"Solution at t = {T}")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return X, Tt, u_pred_flat, u_true, error


def compute_error_metrics(model, Nx=100, Nt=100, T=0.5):
    x = jnp.linspace(0.0, 1.0, Nx + 1)
    t = jnp.linspace(0.0, T, Nt + 1)
    X, Tt = jnp.meshgrid(x, t)

    xt = jnp.stack([X.ravel(), Tt.ravel()], axis=1)

    u_pred = model(xt).reshape(Nt + 1, Nx + 1)
    u_true = u_exact(x, t)

    error = u_pred - u_true
    abs_error = jnp.abs(error)

    num = jnp.sqrt(jnp.mean(error**2))
    den = jnp.sqrt(jnp.mean(u_true**2))
    L2_rel = num / den

    Linf = jnp.max(abs_error)

    return float(L2_rel), float(Linf)
