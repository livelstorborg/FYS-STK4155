from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix


TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 14
ANNOT_FONTSIZE = 14

HIDDEN_LAYER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]
OUTPUT_LAYER_COLOR = "#000000"


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


def _set_tick_params(ax):
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


@dataclass(frozen=True)
class PlotSettings:
    activation_names: Sequence[str]
    noise_floor: float
    max_plot_mse: float
    mse_ylim: Tuple[float, float] = (0.02, 0.2)


def _maybe_eval_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            cleaned = value.replace("\n", " ").strip()
            cleaned = cleaned.strip("[]")
            if not cleaned:
                return []
            arr = np.fromstring(cleaned, sep=" ")
            if arr.size:
                return arr.tolist()
            return []
    if isinstance(value, (float, int)):
        if isinstance(value, float) and np.isnan(value):
            return []
    return value if isinstance(value, list) else []


def _parse_history_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            def _coerce(value):
                if isinstance(value, list):
                    return value
                if hasattr(value, "tolist"):
                    arr = value.tolist()
                    return arr if isinstance(arr, list) else _maybe_eval_list(arr)
                return _maybe_eval_list(value)
            df[col] = df[col].apply(_coerce)
    return df


def _add_irreducible_error_line(ax, settings: PlotSettings) -> None:
    label = "Irreducible Error"
    line_kwargs = dict(
        color="red",
        linestyle="--",
        alpha=0.85,
        linewidth=1.8,
        zorder=10,
        clip_on=False,
    )
    if label not in {line.get_label() for line in ax.get_lines()}:
        ax.axhline(settings.noise_floor, label=label, **line_kwargs)
    else:
        ax.axhline(settings.noise_floor, **line_kwargs)


def aggregate_results_for_plotting(
    df: pd.DataFrame, settings: PlotSettings
) -> pd.DataFrame:
    df = df[df["test_mse_mean"].notna()].copy()
    group_cols = ["activation", "n_layers", "n_nodes"]
    df["valid_mse"] = df["test_mse_mean"] <= settings.max_plot_mse

    counts = (
        df.groupby(group_cols)
        .agg(
            total_runs=("test_mse_mean", "count"),
            success_runs=("valid_mse", "sum"),
        )
        .reset_index()
    )

    all_stats = (
        df.groupby(group_cols)
        .agg(test_mse_all_mean=("test_mse_mean", "mean"))
        .reset_index()
    )
    df_valid = df[df["valid_mse"]].copy()
    if df_valid.empty:
        stats = counts[group_cols].copy()
        stats_columns = [
            "n_params",
            "train_mse_mean",
            "train_mse_std",
            "val_mse_mean",
            "val_mse_std",
            "test_mse_mean",
            "test_mse_std",
            "learning_rate",
            "n_trials",
            "grad_norm_final_mean",
            "grad_norm_final_std",
            "grad_norm_mean_mean",
            "grad_norm_mean_std",
            "batch_norm",
        ]
        for col in stats_columns:
            stats[col] = np.nan
    else:
        stats = (
            df_valid.groupby(group_cols)
            .agg(
                n_params=("n_params", "first"),
                train_mse_mean=("train_mse_mean", "mean"),
                train_mse_std=("train_mse_std", "mean"),
                val_mse_mean=("val_mse_mean", "mean"),
                val_mse_std=("val_mse_std", "mean"),
                test_mse_mean=("test_mse_mean", "mean"),
                test_mse_std=("test_mse_std", "mean"),
                learning_rate=("learning_rate", "first"),
                n_trials=("train_mse", "count"),
                grad_norm_final_mean=("grad_norm_final_mean", "mean"),
                grad_norm_final_std=("grad_norm_final_std", "mean"),
                grad_norm_mean_mean=("grad_norm_mean_mean", "mean"),
                grad_norm_mean_std=("grad_norm_mean_std", "mean"),
                batch_norm=("batch_norm", "first"),
            )
            .reset_index()
        )

    noise_value = settings.noise_floor
    if "noise_floor" in df.columns and df["noise_floor"].notna().any():
        noise_value = float(df["noise_floor"].dropna().iloc[0])

    df_agg = counts.merge(stats, how="left", on=group_cols)
    df_agg["noise_floor"] = noise_value
    df_agg = df_agg.merge(all_stats, how="left", on=group_cols)
    df_agg["n_trials"] = df_agg["success_runs"]
    df_agg["overfitting_gap"] = df_agg["val_mse_mean"] - df_agg["train_mse_mean"]
    df_agg["overfitting_ratio"] = df_agg["test_mse_mean"] / df_agg["train_mse_mean"]
    return df_agg



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

def _compute_layer_metrics_map(
    per_layer_records: List[dict], tail_fraction: float = 0.5
) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    ratios: Dict[Tuple[str, int, int], List[np.ndarray]] = {}
    for rec in per_layer_records:
        arr = np.array(rec.get("per_layer_grad_history"))
        if arr is None or arr.size == 0:
            continue
        n_epochs = arr.shape[0]
        start_idx = int(n_epochs * (1 - tail_fraction))
        layer_means = np.nanmean(arr[start_idx:], axis=0)
        if np.isnan(layer_means).all():
            continue
        key = (rec.get("activation"), int(rec.get("n_layers")), int(rec.get("n_nodes")))
        ratios.setdefault(key, []).append(layer_means)

    result: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for key, mats in ratios.items():
        avg_layers = np.nanmean(np.stack(mats, axis=0), axis=0)
        if np.isnan(avg_layers).all():
            result[key] = {"mean": np.nan, "final": np.nan, "ratio": np.nan}
            continue
        max_val = np.nanmax(avg_layers)
        min_val = np.nanmin(avg_layers)
        ratio = (
            np.nan if min_val <= 0 or not np.isfinite(max_val) else max_val / min_val
        )
        result[key] = {
            "mean": float(np.nanmean(avg_layers)),
            "final": float(avg_layers[-1]) if avg_layers.size else np.nan,
            "ratio": float(ratio),
        }
    return result


def plot_three_row_architecture_overview(
    df_results: pd.DataFrame,
    layer_metrics_map: Dict[Tuple[str, int, int], Dict[str, float]],
    fig_dir: Path,
    settings: PlotSettings,
    use_bn: bool,
    show_figures: bool = False,
) -> None:
    df_agg = df_results[
        (df_results["test_mse_mean"].notna())
        & (df_results["test_mse_mean"] < settings.max_plot_mse)
    ].copy()

    bn_label = "With BatchNorm" if use_bn else "Without BatchNorm"
    n_cols = len(settings.activation_names)
    fig, axes = plt.subplots(3, n_cols, figsize=(20, 13))

    for idx, activation_name in enumerate(settings.activation_names):
        df_act = df_agg[df_agg["activation"] == activation_name]
        if df_act.empty:
            for row in range(3):
                axes[row, idx].axis("off")
            continue

        layers = sorted(df_act["n_layers"].unique())
        nodes = sorted(df_act["n_nodes"].unique())

        heatmap_test = df_act.pivot(
            index="n_layers", columns="n_nodes", values="test_mse_mean"
        ).reindex(index=layers, columns=nodes)
        heatmap_test_all = df_act.pivot(
            index="n_layers", columns="n_nodes", values="test_mse_all_mean"
        ).reindex(index=layers, columns=nodes)
        success_map = df_act.pivot(
            index="n_layers", columns="n_nodes", values="success_runs"
        ).reindex(index=layers, columns=nodes)
        total_map = df_act.pivot(
            index="n_layers", columns="n_nodes", values="total_runs"
        ).reindex(index=layers, columns=nodes)

        vmin_test = np.nanmin(heatmap_test.values)
        vmax_test = np.nanmax(heatmap_test.values)

        annot_matrix = heatmap_test.copy().astype(object)
        for i, layer in enumerate(layers):
            for j, node in enumerate(nodes):
                val = heatmap_test.iloc[i, j]
                fallback = heatmap_test_all.iloc[i, j]
                succ = success_map.iloc[i, j] if success_map is not None else np.nan
                tot = total_map.iloc[i, j] if total_map is not None else np.nan
                succ_txt = ""
                if not np.isnan(tot):
                    tot_int = int(tot)
                    succ_int = int(succ) if not np.isnan(succ) else 0
                    if succ_int < tot_int:
                        succ_txt = f"({succ_int}/{tot_int})"
                if np.isnan(val):
                    if np.isnan(fallback):
                        annot_matrix.iloc[i, j] = succ_txt
                    else:
                        annot_matrix.iloc[i, j] = f"{fallback:.3f}\n{succ_txt}"
                else:
                    annot_matrix.iloc[i, j] = f"{val:.3f}\n{succ_txt}"

        heatmap0 = sns.heatmap(
            heatmap_test,
            annot=annot_matrix.values,
            fmt="",
            cmap="viridis",
            ax=axes[0, idx],
            cbar_kws={"label": "Test MSE"},
            vmin=vmin_test,
            vmax=vmax_test,
            annot_kws={"fontsize": ANNOT_FONTSIZE},
        )
        if heatmap0.collections:
            cbar0 = heatmap0.collections[0].colorbar
            cbar0.ax.tick_params(labelsize=TICK_FONTSIZE)
            if idx == n_cols - 1:
                cbar0.set_label("Test MSE", fontsize=LABEL_FONTSIZE)
            else:
                cbar0.set_label("")
        axes[0, idx].set_ylabel("")

        if df_act["test_mse_mean"].notna().any():
            best = df_act.loc[df_act["test_mse_mean"].idxmin()]
            axes[0, idx].add_patch(
                Rectangle(
                    (nodes.index(best["n_nodes"]), layers.index(best["n_layers"])),
                    1,
                    1,
                    linewidth=3,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            best_text = f"{best['test_mse_mean']:.4f}"
        else:
            best_text = "N/A"

        for i, layer in enumerate(layers):
            for j, node in enumerate(nodes):
                succ = success_map.iloc[i, j] if success_map is not None else np.nan
                tot = total_map.iloc[i, j] if total_map is not None else np.nan
                if not np.isnan(tot) and tot > 0 and succ == 0:
                    axes[0, idx].add_patch(
                        Rectangle(
                            (j, i),
                            1,
                            1,
                            facecolor="lightgrey",
                            edgecolor="grey",
                            alpha=0.6,
                            linewidth=0,
                        )
                    )

        axes[0, idx].set_title(f"Best: {best_text}", fontsize=TITLE_FONTSIZE - 2)
        axes[0, idx].set_xlabel("")
        axes[0, idx].invert_yaxis()
        _set_tick_params(axes[0, idx])
        axes[0, idx].tick_params(labelbottom=False)
        if idx != 0:
            axes[0, idx].tick_params(labelleft=False)

        heatmap_ratio = df_act.pivot(
            index="n_layers", columns="n_nodes", values="overfitting_ratio"
        ).reindex(index=layers, columns=nodes)
        heatmap1 = sns.heatmap(
            heatmap_ratio,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            ax=axes[1, idx],
            cbar_kws={"label": "Test/Train MSE"},
            center=1.0,
            annot_kws={"fontsize": ANNOT_FONTSIZE},
        )
        if heatmap1.collections:
            cbar1 = heatmap1.collections[0].colorbar
            cbar1.ax.tick_params(labelsize=TICK_FONTSIZE)
            if idx == n_cols - 1:
                cbar1.set_label("Test/Train MSE", fontsize=LABEL_FONTSIZE)
            else:
                cbar1.set_label("")
        axes[1, idx].set_ylabel("")
        axes[1, idx].set_title("")
        axes[1, idx].set_xlabel("")
        axes[1, idx].invert_yaxis()
        _set_tick_params(axes[1, idx])
        axes[1, idx].tick_params(labelbottom=False)
        if idx != 0:
            axes[1, idx].tick_params(labelleft=False)

        df_act_layer = df_act.copy()
        df_act_layer["layer_grad_max_min"] = df_act_layer.apply(
            lambda row: layer_metrics_map.get(
                (row["activation"], int(row["n_layers"]), int(row["n_nodes"])),
                {},
            ).get("ratio", np.nan),
            axis=1,
        )
        heatmap_layer_ratio = df_act_layer.pivot(
            index="n_layers", columns="n_nodes", values="layer_grad_max_min"
        ).reindex(index=layers, columns=nodes)
        positive_layer = heatmap_layer_ratio.values[heatmap_layer_ratio.values > 0]
        if positive_layer.size > 0:
            heatmap2 = sns.heatmap(
                heatmap_layer_ratio,
                annot=True,
                fmt=".1f",
                cmap="magma",
                ax=axes[2, idx],
                norm=plt.matplotlib.colors.LogNorm(
                    vmin=np.nanmin(positive_layer), vmax=np.nanmax(positive_layer)
                ),
                cbar_kws={"label": "log max/min gradient"},
                annot_kws={"fontsize": ANNOT_FONTSIZE},
            )
            if heatmap2.collections:
                cbar2 = heatmap2.collections[0].colorbar
                cbar2.ax.tick_params(labelsize=TICK_FONTSIZE)
                if idx == n_cols - 1:
                    cbar2.set_label("log max/min gradient", fontsize=LABEL_FONTSIZE)
                else:
                    cbar2.set_label("")
        else:
            axes[2, idx].axis("off")
        axes[2, idx].set_title("")
        axes[2, idx].set_xlabel("")
        axes[2, idx].set_ylabel("")
        axes[2, idx].invert_yaxis()
        _set_tick_params(axes[2, idx])
        if idx != 0:
            axes[2, idx].tick_params(labelleft=False)

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.88,
        wspace=0.05,
        hspace=0.2268,
    )
    fig.supxlabel("Nodes per Layer", fontsize=LABEL_FONTSIZE)
    fig.supylabel("Number of Layers", fontsize=LABEL_FONTSIZE)

    row_titles = [
        "Test MSE",
        "Generalization Ratio",
        "Layer Imbalance",
    ]
    fig.canvas.draw()
    for row_idx, row_title in enumerate(row_titles):
        row_axes = axes[row_idx, :]
        top = max(ax.get_position().y1 for ax in row_axes)
        offset = 0.019 if row_idx == 0 else 0.012
        fig.text(
            0.5,
            top + offset,
            row_title,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONTSIZE - 2,
        )
    for idx, activation_name in enumerate(settings.activation_names):
        col_axes = axes[:, idx]
        left = min(ax.get_position().x0 for ax in col_axes)
        right = max(ax.get_position().x1 for ax in col_axes)
        top_col = max(ax.get_position().y1 for ax in col_axes)
        fig.text(
            (left + right) / 2,
            top_col + 0.043,
            activation_name,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )

    fig.suptitle(
        f"Architecture Overview ({bn_label})",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        y=0.97,
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bn" if use_bn else ""
    out_path = fig_dir / f"architecture_overview_3row{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    if show_figures:
        plt.show()
    plt.close(fig)


def _collect_per_layer_stats(
    df_depth: pd.DataFrame, max_epochs: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_histories: List[np.ndarray] = []
    std_histories: List[np.ndarray] = []
    dead_histories: List[np.ndarray] = []

    for _, row in df_depth.iterrows():
        mean_data = row.get("per_layer_grad_history")
        if mean_data is None or len(mean_data) == 0:
            continue
        mean_arr = np.array(mean_data, dtype=float)
        mean_histories.append(mean_arr)

        std_data = row.get("per_layer_grad_std_history")
        if std_data is not None and len(std_data) > 0:
            std_arr = np.array(std_data, dtype=float)
        else:
            std_arr = np.zeros_like(mean_arr)
        std_histories.append(std_arr)

        dead_data = row.get("per_layer_dead_history")
        if isinstance(dead_data, float) and np.isnan(dead_data):
            dead_data = None
        if dead_data is not None:
            dead_list = _maybe_eval_list(dead_data)
        else:
            dead_list = []
        if dead_list:
            dead_arr = np.array(dead_list, dtype=float)
        else:
            dead_arr = np.zeros_like(mean_arr)
        dead_histories.append(dead_arr)

    if not mean_histories:
        return np.array([]), np.array([]), np.array([])

    max_epochs_all = max(arr.shape[0] for arr in mean_histories)
    target_epochs = (
        min(max_epochs_all, max_epochs) if max_epochs is not None else max_epochs_all
    )
    padded_means = []
    padded_stds = []
    padded_dead = []
    for mean_arr, std_arr, dead_arr in zip(
        mean_histories, std_histories, dead_histories
    ):
        if mean_arr.shape[0] < max_epochs_all:
            pad_len = max_epochs_all - mean_arr.shape[0]
            mean_pad = np.tile(mean_arr[-1:], (pad_len, 1))
            std_pad = np.tile(std_arr[-1:], (pad_len, 1))
            dead_pad = np.tile(dead_arr[-1:], (pad_len, 1))
            mean_arr = np.vstack([mean_arr, mean_pad])
            std_arr = np.vstack([std_arr, std_pad])
            dead_arr = np.vstack([dead_arr, dead_pad])
        mean_arr = mean_arr[:target_epochs]
        std_arr = std_arr[:target_epochs]
        dead_arr = dead_arr[:target_epochs]
        padded_means.append(mean_arr)
        padded_stds.append(std_arr)
        padded_dead.append(dead_arr)

    avg_means = np.mean(padded_means, axis=0)
    avg_stds = np.mean(padded_stds, axis=0)
    avg_dead = np.mean(padded_dead, axis=0)
    return avg_means, avg_stds, avg_dead


def _select_max_complexity_configs(
    df_raw: pd.DataFrame, max_mse: float
) -> Dict[str, Tuple[int, int]]:
    configs: Dict[str, Tuple[int, int]] = {}
    if "activation" not in df_raw.columns:
        return configs
    df_hist = df_raw
    if "train_history" in df_hist.columns:
        df_hist = df_hist[df_hist["train_history"].notna()]
        if df_hist.empty:
            df_hist = df_raw

    for activation_name in df_hist["activation"].dropna().unique():
        df_act = df_hist[df_hist["activation"] == activation_name]
        if df_act.empty:
            continue
        if "test_mse" in df_act.columns:
            df_success = df_act[df_act["test_mse"] <= max_mse]
        elif "test_mse_mean" in df_act.columns:
            df_success = df_act[df_act["test_mse_mean"] <= max_mse]
        else:
            df_success = df_act
        subset = df_success if not df_success.empty else df_act
        subset = subset.sort_values(["n_layers", "n_nodes"], ascending=[False, False])
        row = subset.iloc[0]
        configs[activation_name] = (int(row["n_layers"]), int(row["n_nodes"]))
    return configs


def _aggregate_history_series(
    series: pd.Series, max_len: int | None = None
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    arrays: List[np.ndarray] = []
    for hist in series:
        data = _maybe_eval_list(hist)
        if not data:
            continue
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return None, None
    max_len_hist = max(len(arr) for arr in arrays)
    target_len = min(max_len_hist, max_len) if max_len is not None else max_len_hist
    padded = []
    for arr in arrays:
        if len(arr) < target_len:
            pad_val = arr[-1]
            pad = np.full(target_len - len(arr), pad_val)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[:target_len]
        padded.append(arr)
    stacked = np.vstack(padded)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def _aggregate_dead_records(
    records: List[dict], target_len: int | None = None
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    arrays: List[np.ndarray] = []
    for rec in records:
        dead_hist = rec.get("per_layer_dead_history")
        if dead_hist is None or len(dead_hist) == 0:
            continue
        arr = np.array(dead_hist, dtype=float)
        if arr.ndim != 2:
            continue
        arrays.append(arr)
    if not arrays:
        return None, None
    max_rec_len = max(arr.shape[0] for arr in arrays)
    target = min(max_rec_len, target_len) if target_len is not None else max_rec_len
    padded = []
    for arr in arrays:
        if arr.shape[0] < max_rec_len:
            pad = np.tile(arr[-1:], (max_rec_len - arr.shape[0], 1))
            arr = np.vstack([arr, pad])
        arr = arr[:target]
        padded.append(arr)
    stacked = np.stack(padded, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def plot_learning_vs_death_curves(
    df_raw: pd.DataFrame,
    per_layer_records: List[dict],
    fig_dir: Path,
    settings: PlotSettings,
    use_bn: bool,
    per_layer_epochs: int | None,
    custom_configs: Dict[str, Tuple[int, int]] | None = None,
    show_figures: bool = False,
) -> None:
    if "train_history" not in df_raw.columns:
        return

    df_histories = df_raw.copy()
    df_histories = _parse_history_columns(
        df_histories, ["train_history", "val_history"]
    )

    configs = (
        custom_configs
        if custom_configs
        else _select_max_complexity_configs(df_histories, settings.max_plot_mse)
    )
    if not configs:
        return

    bn_label = "With BatchNorm" if use_bn else "Without BatchNorm"

    fig, axes = plt.subplots(
        2,
        len(settings.activation_names),
        figsize=(5 * len(settings.activation_names), 6),
        sharex="col",
        constrained_layout=True,
    )

    for col_idx, activation_name in enumerate(settings.activation_names):
        ax_mse = axes[0, col_idx]
        ax_dead = axes[1, col_idx]

        config = configs.get(activation_name)
        if not config:
            ax_mse.axis("off")
            ax_dead.axis("off")
            continue
        target_layers, target_nodes = config

        mask = (
            (df_histories["activation"] == activation_name)
            & (df_histories["n_layers"] == target_layers)
            & (df_histories["n_nodes"] == target_nodes)
        )
        df_act = df_histories[mask]
        if df_act.empty:
            ax_mse.axis("off")
            ax_dead.axis("off")
            continue

        df_act = df_act[df_act["train_history"].notna()]
        if df_act.empty:
            ax_mse.axis("off")
            ax_dead.axis("off")
            continue

        train_mean, train_std = _aggregate_history_series(
            df_act["train_history"], per_layer_epochs
        )
        val_mean, val_std = _aggregate_history_series(
            df_act["val_history"], per_layer_epochs
        )
        if train_mean is None or val_mean is None:
            ax_mse.axis("off")
            ax_dead.axis("off")
            continue

        target_len = len(train_mean)
        epochs = np.arange(1, target_len + 1)

        record_matches = [
            rec
            for rec in per_layer_records
            if rec.get("activation") == activation_name
            and int(rec.get("n_layers", -1)) == target_layers
            and int(rec.get("n_nodes", -1)) == target_nodes
        ]
        dead_mean, dead_std = _aggregate_dead_records(record_matches, target_len)

        ax_mse.plot(
            epochs, train_mean, label="Train MSE", color="steelblue", linewidth=2
        )
        if train_std is not None:
            ax_mse.fill_between(
                epochs,
                train_mean - train_std,
                train_mean + train_std,
                color="steelblue",
                alpha=0.2,
            )
        ax_mse.plot(epochs, val_mean, label="Validation MSE", color="tomato")
        if val_std is not None:
            ax_mse.fill_between(
                epochs,
                val_mean - val_std,
                val_mean + val_std,
                color="tomato",
                alpha=0.2,
            )
        curves = np.concatenate([train_mean, val_mean])
        positive_curves = curves[curves > 0]
        if positive_curves.size == 0:
            ax_mse.axis("off")
            ax_dead.axis("off")
            continue
        min_curve = np.min(positive_curves)
        max_curve = np.max(positive_curves)
        ymin = min(min_curve * 0.7, settings.noise_floor * 0.8)
        ymin = max(ymin, 1e-6)
        ymax = max(max_curve * 1.3, settings.noise_floor * 1.2)
        ax_mse.set_ylabel("MSE", fontsize=LABEL_FONTSIZE)
        title_text = f"{activation_name} | L={target_layers}, N={target_nodes}"
        ax_mse.set_title(title_text, fontsize=TITLE_FONTSIZE)
        ax_mse.set_yscale("log")
        ax_mse.set_ylim(ymin, ymax)
        _add_irreducible_error_line(ax_mse, settings)
        ax_mse.grid(True, alpha=0.3)
        _set_tick_params(ax_mse)
        ax_mse.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

        if dead_mean is not None:
            num_layers = dead_mean.shape[1]
            for layer_idx in range(num_layers):
                is_output = layer_idx == num_layers - 1
                color = (
                    OUTPUT_LAYER_COLOR
                    if is_output
                    else HIDDEN_LAYER_COLORS[layer_idx % len(HIDDEN_LAYER_COLORS)]
                )
                label = "Output" if is_output else f"Layer {layer_idx + 1}"
                alive_curve = np.clip(1.0 - dead_mean[:, layer_idx], 0.0, 1.0)
                ax_dead.plot(
                    epochs,
                    alive_curve,
                    label=label,
                    color=color,
                )
            ax_dead.set_ylabel("Fraction alive", fontsize=LABEL_FONTSIZE)
            ax_dead.legend(ncol=1, fontsize=LEGEND_FONTSIZE, loc="upper right")
        else:
            ax_dead.text(
                0.5,
                0.5,
                "No neuron-death data",
                ha="center",
                va="center",
                transform=ax_dead.transAxes,
            )
            ax_dead.set_ylabel("Fraction alive", fontsize=LABEL_FONTSIZE)
        _set_tick_params(ax_dead)
        ax_dead.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        ax_dead.grid(True, alpha=0.3)

    fig.suptitle(
        f"Learning vs Neuron Death ({bn_label})",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bn" if use_bn else ""
    out_path = fig_dir / f"learning_vs_death_overview{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    if show_figures:
        plt.show()
    plt.close(fig)


def plot_per_layer_gradient_evolution(
    per_layer_records: List[dict],
    fig_dir: Path,
    settings: PlotSettings,
    use_bn: bool,
    per_layer_width: int | None = None,
    per_layer_max_epochs: int | None = None,
    show_figures: bool = False,
) -> None:
    if not per_layer_records:
        print("Warning: No per-layer gradient data available for plotting")
        return

    df_per_layer = pd.DataFrame(per_layer_records)

    df_filtered = df_per_layer
    if per_layer_width is not None:
        df_filtered = df_filtered[df_filtered["n_nodes"] == per_layer_width]
        if df_filtered.empty:
            print(
                f"Warning: No per-layer data for width={per_layer_width}. Showing mixed widths."
            )
            df_filtered = df_per_layer
            per_layer_width = None

    depths_to_plot = sorted(df_filtered["n_layers"].unique())
    if not depths_to_plot:
        print("Warning: No depth data available for per-layer plotting")
        return

    fig, axes = plt.subplots(
        len(depths_to_plot),
        len(settings.activation_names),
        figsize=(18, 3 * len(depths_to_plot)),
        squeeze=False,
    )

    for row_idx, depth in enumerate(depths_to_plot):
        for col_idx, activation_name in enumerate(settings.activation_names):
            ax = axes[row_idx, col_idx]
            df_act = df_filtered[df_filtered["activation"] == activation_name]
            if per_layer_width is not None:
                df_act = df_act[df_act["n_nodes"] == per_layer_width]
            df_depth = df_act[df_act["n_layers"] == depth]

            if df_depth.empty:
                ax.axis("off")
                continue

            avg_per_layer, std_per_layer, dead_per_layer = _collect_per_layer_stats(
                df_depth, max_epochs=per_layer_max_epochs
            )
            if avg_per_layer.size == 0:
                ax.axis("off")
                continue

            n_layers = avg_per_layer.shape[1]
            epochs = np.arange(avg_per_layer.shape[0])
            for layer_idx in range(n_layers):
                is_output = layer_idx == n_layers - 1
                label = "Output" if is_output else f"Layer {layer_idx + 1}"
                color = (
                    OUTPUT_LAYER_COLOR
                    if is_output
                    else HIDDEN_LAYER_COLORS[layer_idx % len(HIDDEN_LAYER_COLORS)]
                )
                layer_grad = np.clip(avg_per_layer[:, layer_idx], 1e-12, None)
                ax.plot(
                    epochs,
                    layer_grad,
                    color=color,
                    linewidth=1.8 if is_output else 1.2,
                    label=label,
                )

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(f"Depth {depth}\nGradient Norm", fontsize=LABEL_FONTSIZE)
            if row_idx == len(depths_to_plot) - 1:
                ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
            if row_idx == 0:
                title = activation_name
                if per_layer_width is not None:
                    title += f" | N={per_layer_width}"
                ax.set_title(title, fontsize=TITLE_FONTSIZE)

            _set_tick_params(ax)
            ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, ncol=1)

    bn_label = "With BatchNorm" if use_bn else "Without BatchNorm"
    fig.suptitle(
        f"Per-Layer Gradient Evolution ({bn_label})",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bn" if use_bn else ""
    out_path = fig_dir / f"per_layer_gradient_evolution{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    if show_figures:
        plt.show()
    plt.close(fig)


def print_summary_table(
    df_results: pd.DataFrame,
    settings: PlotSettings,
    use_bn: bool,
    layer_metrics_map: Dict[Tuple[str, int, int], Dict[str, float]],
) -> None:
    df_agg = df_results[df_results["test_mse_mean"].notna()].copy()
    df_agg["grad_ratio_mean_final"] = df_agg["grad_norm_mean_mean"] / df_agg[
        "grad_norm_final_mean"
    ].clip(lower=1e-12)
    df_agg["grad_cv_trials"] = np.where(
        df_agg["grad_norm_mean_mean"] > 0,
        df_agg["grad_norm_mean_std"] / df_agg["grad_norm_mean_mean"],
        np.nan,
    )

    def get_layer_imbalance(row):
        key = (row["activation"], int(row["n_layers"]), int(row["n_nodes"]))
        return layer_metrics_map.get(key, {}).get("ratio", np.nan)

    df_agg["layer_imbalance"] = df_agg.apply(get_layer_imbalance, axis=1)

    df_sorted = df_agg.sort_values("test_mse_mean")

    bn_label = "With BatchNorm" if use_bn else "Without BatchNorm"
    print(f"\n{'='*150}")
    print(f"SUMMARY TABLE ({bn_label})")
    print(f"{'='*150}")

    def fmt_mean_std(mean, std):
        if np.isnan(mean) or np.isnan(std):
            return "      -        "
        return f"{mean:.4f}±{std:.4f}"

    def fmt_ratio(value):
        if value is None or not np.isfinite(value):
            return "   -   "
        return f"{value:.2f}"

    print(
        f"{'Activation':<12} {'Layers':<8} {'Nodes':<8} {'Params':<8} "
        f"{'Train MSE':<16} {'Val MSE':<16} {'Test MSE':<16} {'Test/Train':<10} "
        f"{'MeanGrad':<12} {'GradCV':<10} {'GradRatio':<10} {'LayerImb':<12} {'Succ':<8} {'LR':<10}"
    )
    print(f"{'-'*150}")
    for _, row in df_sorted.head(20).iterrows():
        succ_runs = int(row.get("success_runs", 0))
        total_runs = int(row.get("total_runs", 0))
        succ_field = f"{succ_runs}/{total_runs}"
        print(
            f"{row['activation']:<12}"
            f"{int(row['n_layers']):<8}"
            f"{int(row['n_nodes']):<8}"
            f"{int(row['n_params']):<8}"
            f"{fmt_mean_std(row['train_mse_mean'], row['train_mse_std'])}  "
            f"{fmt_mean_std(row['val_mse_mean'], row['val_mse_std'])}  "
            f"{fmt_mean_std(row['test_mse_mean'], row['test_mse_std'])}  "
            f"{fmt_ratio(row.get('overfitting_ratio', np.nan)):>10}"
            f"{row.get('grad_norm_mean_mean', np.nan):<12.2e}"
            f"{row.get('grad_cv_trials', np.nan):<10.2f}"
            f"{row.get('grad_ratio_mean_final', np.nan):<10.2f}"
            f"{row.get('layer_imbalance', np.nan):<12.2f}"
            f"{succ_field:<8}"
            f"{row.get('learning_rate', np.nan):<10.2e}"
        )
    print(f"{'='*150}\n")


def generate_part_d_plots(
    df_plot: pd.DataFrame,
    df_raw: pd.DataFrame,
    per_layer_records: List[dict],
    fig_dir: Path,
    settings: PlotSettings,
    use_bn: bool,
    per_layer_width: int | None,
    per_layer_epochs: int | None = None,
    learning_curve_configs: Dict[str, Tuple[int, int]] | None = None,
    show_figures: bool = False,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    layer_metrics_map = _compute_layer_metrics_map(per_layer_records)

    print(f"\nGenerating plots...")
    plot_three_row_architecture_overview(
        df_plot,
        layer_metrics_map,
        fig_dir,
        settings,
        use_bn,
        show_figures=show_figures,
    )
    print(f"  ✓ Three-row architecture overview")

    plot_learning_vs_death_curves(
        df_raw,
        per_layer_records,
        fig_dir,
        settings,
        use_bn,
        per_layer_epochs=per_layer_epochs,
        custom_configs=learning_curve_configs,
        show_figures=show_figures,
    )
    print(f"  ✓ Learning vs neuron-death curves")

    plot_per_layer_gradient_evolution(
        per_layer_records,
        fig_dir,
        settings,
        use_bn,
        per_layer_width=per_layer_width,
        per_layer_max_epochs=per_layer_epochs,
        show_figures=show_figures,
    )
    print(f"  ✓ Per-layer gradient evolution")

    print_summary_table(df_plot, settings, use_bn, layer_metrics_map)















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
    plt.show()
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
    plt.show()
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
    plt.show()
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
