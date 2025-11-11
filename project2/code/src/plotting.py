import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from sklearn.metrics import confusion_matrix


def smooth_graph(data, window_size=20):
    """Smooths a 1D NumPy array using a simple moving average."""
    if window_size <= 1:
        return data
    if len(data) < window_size:
        return data
    
    data = np.asarray(data)
    
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, window, mode='valid')
    
    assert len(smoothed) == len(data), f"Length mismatch: smoothed={len(smoothed)}, data={len(data)}"
    
    return smoothed



# ===================================================================
#                       Part b)
# ===================================================================


def plot_learning_curves_with_std_on_ax(ax, train_mean, train_std, val_mean, val_std, ols_baseline_mse, 
                                        avg_es_epoch, avg_es_val_mse, 
                                        pt_train_history=None, pt_val_history=None, title='', ylabel_enabled=True,
                                        current_N=None, current_SIGMA=None, show_es_line=True):
    """
    Plot learning curves on a given axis, showing mean and smoothed mean for the 
    custom implementation, and optionally full learning curves for the PyTorch baseline.
    The OLS baseline is only plotted if N=300 and sigma=0.1.
    """
    epochs = range(len(train_mean))
    
    # Plot mean curves (Custom Implementation)
    ax.plot(epochs, train_mean, color='C0', linewidth=2, alpha=0.3, zorder=1)
    ax.plot(epochs, val_mean, color='C1', linewidth=2, alpha=0.3, zorder=1)

    train_mean_smooth = smooth_graph(train_mean)
    val_mean_smooth = smooth_graph(val_mean)

    ax.plot(epochs, train_mean_smooth, color='C0', label='Train MSE (Mean)', linewidth=2, zorder=2)
    ax.plot(epochs, val_mean_smooth, color='C1', label='Validation MSE (Mean)', linewidth=2, zorder=2)
    
    # Plot Early Stopping point (Vertical Line)
    if show_es_line and avg_es_epoch > 0:
        ax.axvline(
            x=avg_es_epoch, 
            color='black', linewidth=2.5, alpha=0.7,
            zorder=5, 
            label=f'Early Stopping \n(Epoch$\\approx${round(avg_es_epoch)}, MSE$\\approx${avg_es_val_mse:.4f})'
        )

    # PyTorch Baselines (FULL CURVES)
    if pt_train_history is not None and pt_val_history is not None:
        pt_epochs = range(len(pt_train_history))
        pt_train_smooth = smooth_graph(pt_train_history, window_size=40)
        pt_val_smooth = smooth_graph(pt_val_history, window_size=40)

        ax.plot(pt_epochs, pt_train_smooth, label=f'Train MSE (PyTorch)', 
                linewidth=2, color='darkorchid', linestyle='--', zorder=3)
        ax.plot(pt_epochs, pt_val_smooth, label=f'Validation MSE (PyTorch)', 
                linewidth=2, color='deeppink', linestyle='--', zorder=3)

    # OLS baseline (CONDITIONAL PLOTTING)
    if current_N == 300 and current_SIGMA == 0.1:
        ax.axhline(y=ols_baseline_mse, color='r', linestyle='--', 
                   label=f'Validation MSE OLS ({ols_baseline_mse:.4f})', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=18)
    if ylabel_enabled:
        ax.set_ylabel('Loss', fontsize=18, labelpad=-3)

    ax.set_yscale('log')
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=18)








# ===================================================================
#                       Part d)
# ===================================================================















# ===================================================================
#                       Part e)
# ===================================================================


def plot_combined_learning_curves(
    results_l1,
    results_l2,
    ml_baseline_l1,
    ml_baseline_l2,
    activation_name,
    architecture_title,
    config,
    plot_early_stopping=True,
    include_final_mse=False,
):
    """
    Plots L1 (Lasso) and L2 (Ridge) learning curves in separate subplots.
    Includes vertical lines for average early stopping epochs.
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- LEFT SUBPLOT: L1 (Lasso) ---
    ax_l1 = axes[0]
    h1 = results_l1["history"]
    train_loss_l1 = h1["train_loss"]
    val_loss_l1 = h1["val_loss"]

    # Get final MSE values
    final_train_mse_l1 = train_loss_l1[-1]
    final_val_mse_l1 = val_loss_l1[-1]

    # Create legend labels with optional final MSE
    if include_final_mse:
        train_label_l1 = f"Train MSE (Mean) - {final_train_mse_l1:.4f}"
        val_label_l1 = f"Validation MSE (Mean) - {final_val_mse_l1:.4f}"
    else:
        train_label_l1 = "Train MSE (Mean)"
        val_label_l1 = "Validation MSE (Mean)"

    ax_l1.plot(train_loss_l1, color="C0", alpha=0.15, linewidth=1)
    ax_l1.plot(val_loss_l1, color="C1", alpha=0.15, linewidth=1)
    ax_l1.plot(
        smooth_graph(train_loss_l1), color="C0", label=train_label_l1, linewidth=2
    )
    ax_l1.plot(
        smooth_graph(val_loss_l1), color="C1", label=val_label_l1, linewidth=2
    )

    # L1 Benchmark Baseline (Lasso) - ALWAYS PLOT
    ax_l1.axhline(
        y=ml_baseline_l1,
        color="red",
        linestyle="--",
        label=f"Validation MSE Lasso ({ml_baseline_l1:.4f})",
        linewidth=2,
    )
    
    # L1 Early Stopping Line (Conditional)
    if plot_early_stopping and results_l1["avg_es_epoch"] > 0:
        ax_l1.axvline(
            x=results_l1["avg_es_epoch"],
            color="black",
            linewidth=2.5,
            alpha=0.7,
            zorder=5,
            label=f'Early Stopping (Epoch $\\approx$ {round(results_l1["avg_es_epoch"])}, MSE $\\approx$ {results_l1["avg_es_val_mse"]:.4f})'
        )

    # L1 subplot formatting
    ax_l1.set_xlabel("Epoch", fontsize=16)
    ax_l1.set_ylabel("Loss", fontsize=16, labelpad=5)
    ax_l1.set_yscale("log")
    ax_l1.set_title(
        f"L1: $\\mathbf{{\eta}}$ = {results_l1['best_eta']:.2e}, $\\mathbf{{\lambda}}$ = {results_l1['best_lam']:.2e}",
        fontsize=16,
        fontweight="bold",
        pad=10
    )
    ax_l1.legend(fontsize=14, loc="upper right")
    ax_l1.grid(True, alpha=0.3, which="both")
    ax_l1.tick_params(axis="both", which="major", labelsize=16)

    # --- RIGHT SUBPLOT: L2 (Ridge) ---
    ax_l2 = axes[1]
    h2 = results_l2["history"]
    train_loss_l2 = h2["train_loss"]
    val_loss_l2 = h2["val_loss"]

    # Get final MSE values
    final_train_mse_l2 = train_loss_l2[-1]
    final_val_mse_l2 = val_loss_l2[-1]

    # Create legend labels with optional final MSE
    if include_final_mse:
        train_label_l2 = f"Train MSE (Mean) - {final_train_mse_l2:.4f}"
        val_label_l2 = f"Validation MSE (Mean) - {final_val_mse_l2:.4f}"
    else:
        train_label_l2 = "Train MSE (Mean)"
        val_label_l2 = "Validation MSE (Mean)"

    ax_l2.plot(train_loss_l2, color="darkorchid", alpha=0.15, linewidth=1)
    ax_l2.plot(val_loss_l2, color="deeppink", alpha=0.15, linewidth=1)
    ax_l2.plot(
        smooth_graph(train_loss_l2),
        color="darkorchid",
        label=train_label_l2,
        linewidth=2,
    )
    ax_l2.plot(
        smooth_graph(val_loss_l2),
        color="deeppink",
        label=val_label_l2,
        linewidth=2,
    )

    # L2 Benchmark Baseline (Ridge) - ALWAYS PLOT
    ax_l2.axhline(
        y=ml_baseline_l2,
        color="red",
        linestyle="--",
        label=f"Validation MSE Ridge ({ml_baseline_l2:.4f})",
        linewidth=2,
    )
    
    # L2 Early Stopping Line (Conditional)
    if plot_early_stopping and results_l2["avg_es_epoch"] > 0:
        ax_l2.axvline(
            x=results_l2["avg_es_epoch"],
            color="black",
            linewidth=2.5,
            alpha=0.7,
            zorder=5,
            label=f'Early Stopping (Epoch $\\approx$ {round(results_l2["avg_es_epoch"])}, MSE $\\approx$ {results_l2["avg_es_val_mse"]:.4f})'
        )

    # L2 subplot formatting
    ax_l2.set_xlabel("Epoch", fontsize=16)
    ax_l2.set_yscale("log")
    ax_l2.set_title(
        f"L2: $\\mathbf{{\eta}}$ = {results_l2['best_eta']:.2e}, $\\mathbf{{\lambda}}$ = {results_l2['best_lam']:.2e}",
        fontsize=16,
        fontweight="bold",
        pad=10
    )
    ax_l2.legend(fontsize=14, loc="upper right")
    ax_l2.grid(True, alpha=0.3, which="both")
    ax_l2.tick_params(axis="both", which="major", labelsize=16)

    # --- Overall figure title ---
    fig.suptitle(
        f'Learning Curves with {config["optimizer_class"].__name__} and {activation_name} ($\\mathbf{{N}}$={config["N_SELECTED"]}, $\\mathbf{{\\sigma}}$={config["SIGMA_SELECTED"]}) - Averaged over {config["LR_TRIALS"]} Runs\n'
        f'{architecture_title}',
        fontsize=20,
        fontweight="bold",
        y=0.97
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Mandatory File Saving ---
    Path("figs").mkdir(exist_ok=True)
    filename_lc = f'figs/lc_combined_N{config["N_SELECTED"]}_sigma{config["SIGMA_SELECTED"]}_{config["optimizer_class"].__name__}_{activation_name}_Exp{config["EXPERIMENT_SELECTED"]}.pdf'
    plt.savefig(filename_lc, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"\n✓ Saved Combined Learning Curve: {filename_lc}")


def lambda_eta_heatmap(
    metric_array,
    eta_vals,
    lambda_vals,
    metric_name="MSE",
    dataset="Validation",
    cmap="viridis",
    figsize=(10, 8),
    annot=True,
    maximize=False,
    optimizer_name="Optimizer",
    reg_type="Reg",
    N=0,
    sigma=0,
    exp_id=0,
    activation_name="Act",
    n_trials=1,
):
    """
    Plots the validation MSE grid as a heatmap with enhanced aesthetics
    and saves it with a descriptive filename.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = sns.heatmap(
        metric_array,
        annot=annot,
        fmt=".4f" if annot else None,
        annot_kws={"fontsize": 20},  # Cell annotation fontsize
        cmap=cmap,
        ax=ax,
        xticklabels=[f"{int(np.log10(e))}" for e in eta_vals],  # Eta is columns/x-axis
        yticklabels=[
            f"{int(np.log10(l))}" for l in lambda_vals
        ],  # Lambda is rows/y-axis
        cbar_kws={"label": metric_name},
    )

    # Set colorbar label font size and tick label size
    cbar = im.collections[0].colorbar
    cbar.set_label(metric_name, fontsize=16)
    cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick label size

    # Invert y-axis so smaller lambda values (smaller log) are at the bottom
    ax.invert_yaxis()

    # Find best value location
    if maximize:
        best_idx = np.unravel_index(np.argmax(metric_array), metric_array.shape)
    else:
        best_idx = np.unravel_index(np.argmin(metric_array), metric_array.shape)

    # i_best is the row (lambda index), j_best is the column (eta index)
    i_best, j_best = best_idx

    # Add red box around best cell
    rect = Rectangle(
        (j_best, i_best), 1, 1, linewidth=3, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # --- Title and labels with consistent sizing ---
    ax.set_title(
        f"{optimizer_name} with {activation_name} ({reg_type}, $\\mathbf{{N}}$={N}, $\\mathbf{{\\sigma}}$={sigma})\nAveraged over {n_trials} Runs",
        fontsize=22,
        fontweight="bold",
    )
    ax.set_xlabel(r"$\log_{10}(\eta)$", fontsize=20)
    ax.set_ylabel(r"$\log_{10}(\lambda)$", fontsize=20)

    # Set tick parameters for better visualization
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    plt.tight_layout()

    # --- Mandatory File Saving ---
    Path("figs").mkdir(exist_ok=True)
    filename = f"figs/heatmap_N{N}_sigma{sigma}_{optimizer_name}_{reg_type}_{activation_name}_Exp{exp_id}.pdf"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ Saved Heatmap: {filename}")
    # --- End Saving ---

    return fig, ax










# ===================================================================
#                       Part f)
# ===================================================================


def plot_lambda_eta_heatmaps(results, lambda_values, eta_values, activation, optimizer):
    """
    Plot lambda-eta heatmaps for each network architecture in a combined subplot figure.
    """
    n_architectures = len(results["n_layers"])

    # Determine grid layout for subplots
    if n_architectures <= 2:
        n_rows, n_cols = 1, n_architectures
        figsize = (7 * n_architectures, 6)
    elif n_architectures <= 4:
        n_rows, n_cols = 2, 2
        figsize = (14, 12)
    elif n_architectures <= 6:
        n_rows, n_cols = 2, 3
        figsize = (18, 12)
    else:
        n_rows, n_cols = 3, 3
        figsize = (18, 15)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_architectures == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_architectures):
        ax = axes[idx]

        n_layers = results["n_layers"][idx]
        n_nodes = results["n_nodes"][idx]
        grid_results = results["grid_results"][idx]

        # Convert grid results to 2D array
        lambda_vals = np.array(grid_results["lambda_values"])
        eta_vals = np.array(grid_results["eta_values"])
        accuracies = np.array(grid_results["val_accuracies"])

        # Create heatmap data
        heatmap_data = np.full((len(eta_values), len(lambda_values)), np.nan)

        # Map results back to the grid
        for lam, eta, acc in zip(lambda_vals, eta_vals, accuracies):
            lambda_idx = np.argmin(np.abs(lambda_values - lam))
            eta_idx = np.argmin(np.abs(eta_values - eta))
            heatmap_data[eta_idx, lambda_idx] = acc

        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            xticklabels=[f"{int(np.log10(lam))}" for lam in lambda_values],
            yticklabels=[f"{int(np.log10(eta))}" for eta in eta_values],
            cbar_kws={"label": "Accuracy"},
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar=True,
            annot_kws={"fontsize": 14},
        )
        
        # Increase colorbar label font size
        cbar = ax.collections[0].colorbar
        cbar.set_label("Accuracy", fontsize=14)

        # Mark best configuration with a red rectangle
        best_lambda = results["best_lambda"][idx]
        best_eta = results["best_eta"][idx]

        best_lambda_idx = np.argmin(np.abs(lambda_values - best_lambda))
        best_eta_idx = np.argmin(np.abs(eta_values - best_eta))

        rect = Rectangle(
            (best_lambda_idx, best_eta_idx),
            1,
            1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=16)
        ax.set_ylabel(r"$\log_{10}(\eta)$", fontsize=16)
        ax.set_title(
            f'$\\mathbf{{{n_layers}}}$ layer(s) ' + r'$\times$ ' + f'$\\mathbf{{{n_nodes}}}$ nodes',
            fontsize=18,
            fontweight="bold",
        )

    # Hide unused subplots
    for idx in range(n_architectures, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        r"Validation Accuracy: $\mathbf{\lambda}$-$\boldsymbol{\eta}$ Search",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save figure
    Path("figs").mkdir(exist_ok=True)
    save_path = f"figs/lambda_eta_heatmaps_all_{activation}_{optimizer}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n✓ Saved combined heatmaps: {save_path}")
    plt.close()


def plot_architecture_comparison_heatmaps(
    results, activation, optimizer, save_name=None
):
    """
    Plot side-by-side heatmaps of validation accuracy and training accuracy
    for different architectures, with red rectangle around the best val acc
    """
    if save_name is None:
        save_name = f'figs/architecture_comparison_heatmaps_{activation}_{optimizer}.pdf'
    
    n_layers = np.array(results["n_layers"])
    n_nodes = np.array(results["n_nodes"])
    val_accuracies = np.array(results["val_accuracy"])
    train_accuracies = np.array(results["train_accuracy"])

    # Create pivot tables
    unique_layers = sorted(set(n_layers))
    unique_nodes = sorted(set(n_nodes))

    val_heatmap_data = np.zeros((len(unique_nodes), len(unique_layers)))
    train_heatmap_data = np.zeros((len(unique_nodes), len(unique_layers)))

    for i, nodes in enumerate(unique_nodes):
        for j, layers in enumerate(unique_layers):
            mask = (n_layers == layers) & (n_nodes == nodes)
            if np.any(mask):
                val_heatmap_data[i, j] = val_accuracies[mask][0]
                train_heatmap_data[i, j] = train_accuracies[mask][0]

    # Find best value location in the validation data
    best_idx = np.unravel_index(np.argmax(val_heatmap_data), val_heatmap_data.shape)
    i_best, j_best = best_idx

    # Create subplot plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Validation Accuracy Heatmap ---
    sns.heatmap(
        val_heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        xticklabels=unique_layers,
        yticklabels=unique_nodes,
        cbar_kws={"label": "Accuracy"},
        vmin=0.0,
        vmax=1.0,
        ax=axes[0],
        annot_kws={"fontsize": 18},
        linewidths=2,
        linecolor='white',
    )
    
    # Increase colorbar label and tick font sizes
    cbar = axes[0].collections[0].colorbar
    cbar.set_label("Accuracy", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # Increase axis tick label sizes
    axes[0].tick_params(axis='both', labelsize=20)

    # Add red rectangle to the best validation accuracy
    rect = Rectangle(
        (j_best, i_best), 1, 1, linewidth=4, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)

    axes[0].set_xlabel("Number of Hidden Layers", fontsize=22)
    axes[0].set_ylabel("Nodes per Hidden Layer", fontsize=22)
    axes[0].set_title("Validation Accuracy", fontsize=22, fontweight="bold", pad=20)

    # --- Training Accuracy Heatmap ---
    sns.heatmap(
        train_heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        xticklabels=unique_layers,
        yticklabels=unique_nodes,
        cbar_kws={"label": "Accuracy"},
        vmin=0.0,
        vmax=1.0,
        ax=axes[1],
        annot_kws={"fontsize": 18},
        linewidths=2,
        linecolor='white',
    )
    
    # Increase colorbar label and tick font sizes
    cbar = axes[1].collections[0].colorbar
    cbar.set_label("Accuracy", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # Increase axis tick label sizes
    axes[1].tick_params(axis='both', labelsize=16)

    axes[1].set_xlabel("Number of Hidden Layers", fontsize=20)
    axes[1].set_ylabel("Nodes per Hidden Layer", fontsize=20)
    axes[1].set_title("Training Accuracy", fontsize=20, fontweight="bold", pad=20)

    plt.suptitle(
        r"Accuracy vs Network Architecture with optimal $\mathbf{\lambda}$ and $\boldsymbol{\eta}$",
        fontsize=24,
        fontweight="bold",
    )

    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(save_name, bbox_inches='tight')
    print(f"✓ Saved architecture comparison heatmaps: {save_name}")
    plt.close()





def plot_confusion_matrix(y_true, y_pred, activation, optimizer, n_layers=None, n_nodes=None, save_name=None):
    """Plot confusion matrix for classification results with viridis colormap"""
    if save_name is None:
        save_name = f'figs/confusion_matrix_{activation}_{optimizer}.pdf'
    
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=range(10),
        yticklabels=range(10),
        cbar_kws={"label": "Count"},
        ax=ax,
        annot_kws={"fontsize": 16},
    )
    
    # Increase colorbar label and tick font sizes
    cbar = ax.collections[0].colorbar
    cbar.set_label("Count", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Increase x and y tick label sizes
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

    ax.set_xlabel("Predicted Label", fontsize=22)
    ax.set_ylabel("True Label", fontsize=22)
    
    # Create title with architecture info if provided
    if n_layers is not None and n_nodes is not None:
        title = f"Confusion Matrix - MNIST Classification\n{n_layers} layer(s) × {n_nodes} nodes"
    else:
        title = "Confusion Matrix - MNIST Classification"
    
    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(save_name, bbox_inches="tight")
    print(f"✓ Saved confusion matrix: {save_name}")
    plt.close()

    # Print classification report
    print("\nPer-class accuracy:")
    print("-" * 40)
    for digit in range(10):
        digit_mask = y_true == digit
        if np.sum(digit_mask) > 0:
            digit_acc = np.sum(y_pred[digit_mask] == digit) / np.sum(digit_mask)
            print(f"  Digit {digit}: {digit_acc:.4f}")
        else:
            print(f"  Digit {digit}: No true samples.")

    print("-" * 40)
