import matplotlib.pyplot as plt
import numpy as np


def plot_error(
    polynomial_deg,
    error_dict,
    *,
    ylabel: str = "Error",
    figsize: tuple[int, int] = (8, 5),
    filename: str | None = None,
):
    """Plot error vs polynomial degree for multiple series on one figure.

    Parameters
    ----------
    polynomial_deg : array-like
        Sequence of polynomial degrees (x-axis).
    error_dict : dict[str, array-like]
        Dictionary of label -> error values (each will be plotted).
    ylabel : str, optional
        Label for y-axis, defaults to "Error".
    figsize : tuple[int, int], optional
        Figure size in inches, defaults to (8, 5).
    filename : str | None, optional
        If provided, saves the figure to this path.
    """
    x = np.asarray(polynomial_deg)

    fig, ax = plt.subplots(figsize=figsize)

    for label, error in error_dict.items():
        y = np.asarray(error)
        ax.plot(x, y, linestyle="-", marker="o", label=label)

    ax.grid(True)
    ax.set_xlabel("Polynomial Degree", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig
