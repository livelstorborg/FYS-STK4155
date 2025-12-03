import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import numpy as np


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


# ============================================================
#  HEATMAP: L2 error as a function of width × depth
# ============================================================
def plot_heatmap_width_depth(df, activation):
    """
    Plot a heatmap for a specific activation function showing the
    relative L2 error for each combination of (width, depth).

    df : pandas DataFrame from run_architecture_sweep
    activation : string key for activation function (e.g. "tanh")
    """

    data = df[df["activation"] == activation]

    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    # Matrix for heatmap: rows=depth, cols=width
    M = np.zeros((len(depths), len(widths)))

    for _, row in data.iterrows():
        i = depths.index(row["hidden_layers"])
        j = widths.index(row["width"])
        M[i, j] = row["L2_rel_mean"]

    plt.figure(figsize=(7, 5))
    plt.imshow(M, cmap="viridis", origin="lower")
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.xlabel("Width (nodes per hidden layer)")
    plt.ylabel("Depth (number of hidden layers)")
    plt.title(f"Relative $L^2$ Error Heatmap (activation = {activation})")
    plt.colorbar(label=r"$L^2$ error")
    plt.tight_layout()
    plt.show()


# ============================================================
#  LINE PLOT: Error vs width (for a chosen activation)
# ============================================================
def plot_error_vs_width(df, activation):
    """
    Line plots of L2 error vs width for each depth, for a chosen activation.
    """

    data = df[df["activation"] == activation]

    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    plt.figure(figsize=(7, 5))

    for L in depths:
        subset = data[data["hidden_layers"] == L].sort_values("width")
        plt.plot(
            subset["width"], subset["L2_rel_mean"], marker="o", label=f"{L} layer(s)"
        )

    plt.xlabel("Width (nodes per layer)")
    plt.ylabel("Relative $L^2$ error")
    plt.title(f"L2 Error vs Width (activation = {activation})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  LINE PLOT: Error vs depth (for a chosen activation)
# ============================================================
def plot_error_vs_depth(df, activation):
    """
    L2 error vs depth for each width, for a chosen activation.
    """

    data = df[df["activation"] == activation]

    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    plt.figure(figsize=(7, 5))

    for W in widths:
        subset = data[data["width"] == W].sort_values("hidden_layers")
        plt.plot(
            subset["hidden_layers"], subset["L2_rel_mean"], marker="o", label=f"W={W}"
        )

    plt.xlabel("Depth (hidden layers)")
    plt.ylabel("Relative $L^2$ error")
    plt.title(f"L2 Error vs Depth (activation = {activation})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  BAR CHART: Activation function comparison
# ============================================================
def plot_error_vs_activation(df, width, depth):
    """
    Compare activation functions for a fixed architecture (width, depth).
    """

    subset = df[(df["width"] == width) & (df["hidden_layers"] == depth)]

    if subset.empty:
        raise ValueError(f"No results found for width={width}, depth={depth}.")

    plt.figure(figsize=(6, 4))
    plt.bar(subset["activation"], subset["L2_rel_mean"])
    plt.xlabel("Activation function")
    plt.ylabel("Relative $L^2$ error")
    plt.title(f"Activation Comparison (width={width}, depth={depth})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
#  AUTOMATICALLY PLOT ALL HEATMAPS
# ============================================================
def plot_all_heatmaps(df, save_dir="figs"):
    """
    Automatically generate and save heatmaps for all activation functions
    present in the architecture sweep results.
    """
    os.makedirs(save_dir, exist_ok=True)

    activations = df["activation"].unique()

    for act in activations:
        print(f"Creating heatmap for activation: {act}")
        fig = plot_heatmap_width_depth(df, activation=act, show=False)

        filename = f"heatmap_activation_{act}.pdf"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {filepath}")
