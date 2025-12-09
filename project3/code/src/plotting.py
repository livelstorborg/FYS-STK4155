import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import numpy as np


def plot_solution(x, u_num, u_true, title="", filepath="figs/plot.pdf"):
    os.makedirs("figs", exist_ok=True)
    plt.plot(x, u_true, label="Analytical", color="red")
    plt.plot(x, u_num, "--", label="Numerical", color="blue")
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("u(x, t)", fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.savefig(filepath)
    plt.show()


def plot_training_loss(losses):
    losses_np = jnp.asarray(losses)

    steps = np.arange(len(losses_np))
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps, losses_np)
    plt.xlabel("Training step")
    plt.ylabel("Loss (log scale)")
    plt.title(f"PINN training loss — {len(losses_np)} steps")
    plt.grid(alpha=0.3)
    plt.show()
    return losses_np


# ============================================================
#  HEATMAP: L2 error as a function of width × depth
# ============================================================
def plot_heatmap_width_depth(df, activation, show=True):
    data = df[df["activation"] == activation]

    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    M = np.zeros((len(depths), len(widths)))

    for _, row in data.iterrows():
        i = depths.index(row["hidden_layers"])
        j = widths.index(row["width"])
        M[i, j] = row["L2_rel_mean"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap="viridis", origin="lower")

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)

    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)

    ax.set_xlabel("Width (nodes per hidden layer)", fontsize=16)
    ax.set_ylabel("Depth (number of hidden layers)", fontsize=16)
    ax.set_title(f"Relative $L^2$ Error ({activation})", fontsize=18)

    # ----------------------------
    # Add numbers inside each cell
    # ----------------------------
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j, i, f"{M[i, j]:.3f}",
                ha="center", va="center",
                color="white", fontsize=10, alpha=0.8
            )

    # -----------------------------------
    # Draw red rectangle around min value
    # -----------------------------------
    min_row, min_col = np.unravel_index(np.argmin(M), M.shape)
    rect = plt.Rectangle(
        (min_col - 0.5, min_row - 0.5), 1, 1,
        facecolor="none", edgecolor="red", linewidth=2
    )
    ax.add_patch(rect)

    # --------------------
    # Make grid lines faint
    # --------------------
    ax.set_xticks(np.arange(-0.5, len(widths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.3), linewidth=0.7)
    ax.grid(which="major", alpha=0)  # hide old major grid

    cbar = fig.colorbar(im, ax=ax, label=r"$L^2$ error")
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r"$L^2$ error", fontsize=18)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================
#  LINE PLOT: Error vs width (for a chosen activation)
# ============================================================
def plot_error_vs_width(df, activation, show=True):
    data = df[df["activation"] == activation]
    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for L in depths:
        subset = data[data["hidden_layers"] == L].sort_values("width")
        ax.plot(
            subset["width"], subset["L2_rel_mean"], marker="o", label=f"{L} layer(s)"
        )

    ax.set_xlabel("Width (nodes per layer)")
    ax.set_ylabel("Relative $L^2$ error")
    ax.set_title(f"L2 Error vs Width (activation = {activation})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================
#  LINE PLOT: Error vs depth (for a chosen activation)
# ============================================================
def plot_error_vs_depth(df, activation, show=True):
    data = df[df["activation"] == activation]
    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for W in widths:
        subset = data[data["width"] == W].sort_values("hidden_layers")
        ax.plot(
            subset["hidden_layers"], subset["L2_rel_mean"], marker="o", label=f"W={W}"
        )

    ax.set_xlabel("Depth (hidden layers)")
    ax.set_ylabel("Relative $L^2$ error")
    ax.set_title(f"L2 Error vs Depth (activation = {activation})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


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
