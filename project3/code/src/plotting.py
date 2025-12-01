import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_solution(x, u_num, u_true, title=""):
    plt.plot(x, u_num, label="Numerical")
    plt.plot(x, u_true, "--", label="Analytical")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_training_loss(losses):
    steps = jnp.arange(len(losses))
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps, losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN training loss")
    plt.grid(alpha=0.3)
    plt.show()
