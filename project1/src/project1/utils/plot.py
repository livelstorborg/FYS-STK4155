import matplotlib.pyplot as plt
import numpy as np


def plot(
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


def plot_error(
    x,
    series: dict,
    *,
    xlabel: str = "Polynomial Degree",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = (6, 4),
    filename: str | None = None,
):
    """Plot multiple series on one figure.

    Parameters
    ----------
    x : array-like
        Shared x-values.
    series : dict[label, y-array]
        Mapping from legend label to y-values to plot.
    xlabel, ylabel : str, optional
        Axis labels.
    figsize : tuple, optional
        Figure size.
    filename : str | None, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x_arr = np.asarray(x)
    for label, y in series.items():
        ax.plot(x_arr, np.asarray(y), marker="o", label=label)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend()
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return ax
