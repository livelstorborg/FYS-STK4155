"""Part D orchestration utilities (architecture sweep with/without BatchNorm)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os

from .activations import LeakyReLU, Linear, ReLU, Sigmoid
from .losses import MSE
from .neural_network import NeuralNetwork
from .optimizers import Adam
from .training import train
from .utils import generate_data, split_scale_data
from .plotting import (
    PlotSettings,
    aggregate_results_for_plotting,
    generate_part_d_plots,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)
DEFAULT_SPLIT_SEEDS = [42]

USE_BATCH_NORM = True
TRACK_PER_LAYER = True

N_SAMPLES = 300
NOISE_STD = 0.1
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
NOISE_FLOOR = NOISE_STD**2
MAX_PLOT_MSE = 1.0

N_LAYERS_DEFAULT = [1, 2, 3, 4]
N_NODES_DEFAULT = [50, 100, 150, 200, 250]
ACTIVATIONS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "LeakyReLU": LeakyReLU}

EPOCHS = 1000
BATCH_SIZE = 50
DEFAULT_ETA = 0.01
N_TRIALS = 5

CALIBRATION_DEPTHS = [1, 2]
CALIBRATION_WIDTHS = [50, 100]

LR_SEARCH_CONFIG = {
    "SEED": SEED,
    "ETA_VALS": list(np.logspace(-5, 0, 10)),
    "LR_TRIALS": 5,
    "EPOCHS_LR_SEARCH": 500,
    "BATCH_SIZE": BATCH_SIZE,
}

CODE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = CODE_DIR / "notebooks"
RESULTS_DIR = NOTEBOOKS_DIR / "results" / "part_d"
FIGURES_DIR = NOTEBOOKS_DIR / "figs" / "part_d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PLOT_SETTINGS = PlotSettings(
    activation_names=list(ACTIVATIONS.keys()),
    noise_floor=NOISE_FLOOR,
    max_plot_mse=MAX_PLOT_MSE,
)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def _prepare_data(split_seed: int) -> Dict[str, np.ndarray]:
    X, y = generate_data(n_samples=N_SAMPLES, noise_level=NOISE_STD, seed=split_seed)
    splits = split_scale_data(
        X, y, train_ratio=TRAIN_SIZE, val_ratio=VAL_SIZE, seed=split_seed
    )
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        *_,
    ) = splits
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# -----------------------------------------------------------------------------
# Learning-rate calibration
# -----------------------------------------------------------------------------


def _calibrate_learning_rates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_batch_norm: bool,
) -> Dict[str, Dict[int, Dict[int, float]]]:
    bn_label = "WITH" if use_batch_norm else "WITHOUT"
    print(f"\nCalibrating learning rates ({bn_label} BatchNorm)...")
    lookup: Dict[str, Dict[int, Dict[int, float]]] = {}
    for activation_name, activation_class in ACTIVATIONS.items():
        lookup[activation_name] = {}
        for depth in CALIBRATION_DEPTHS:
            lookup[activation_name][depth] = {}
            for width in CALIBRATION_WIDTHS:
                layer_sizes = [width] * depth + [1]
                activations = [activation_class() for _ in range(depth)] + [Linear()]
                best_eta, _ = _find_best_eta(
                    layer_sizes=layer_sizes,
                    activations=activations,
                    optimizer_class=Adam,
                    X_train_scaled=X_train,
                    y_train_scaled=y_train,
                    X_val_scaled=X_val,
                    y_val_scaled=y_val,
                    config=LR_SEARCH_CONFIG,
                    use_batch_norm=use_batch_norm,
                )
                if not np.isfinite(best_eta):
                    best_eta = DEFAULT_ETA
                lookup[activation_name][depth][width] = float(best_eta)
                print(
                    f"  {activation_name:<10} | depth={depth:<2} | width={width:<3} | η = {best_eta:.2e}"
                )
    return lookup


def _find_best_eta(
    layer_sizes,
    activations,
    optimizer_class,
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
    config,
    use_batch_norm=False,
):
    """Wrapper around find_best_eta_d that adds BatchNorm support."""
    n_jobs = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
    input_size = X_train_scaled.shape[1]

    def train_single_eta_trial(trial, eta):
        """Train with one eta value and one trial"""
        trial_seed = config["SEED"] + trial
        model = NeuralNetwork(
            network_input_size=input_size,
            layer_output_sizes=layer_sizes,
            activations=activations,
            loss=MSE(),
            seed=trial_seed,
            lambda_reg=0.0,
            use_batch_norm=use_batch_norm,
        )

        train(
            nn=model,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
            optimizer=optimizer_class(eta=eta, update_batch_norm_params=use_batch_norm),
            epochs=config["EPOCHS_LR_SEARCH"],
            batch_size=config["BATCH_SIZE"],
            stochastic=True,
            task="regression",
            early_stopping=True,
            patience=150,
            verbose=False,
            seed=trial_seed,
        )

        y_val_pred = model.predict(X_val_scaled)
        val_mse = np.mean((y_val_scaled - y_val_pred) ** 2)

        return trial, eta, val_mse

    experiments = [
        (trial, eta)
        for trial in range(config["LR_TRIALS"])
        for eta in config["ETA_VALS"]
    ]

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(train_single_eta_trial)(trial, eta) for trial, eta in experiments
    )

    val_mse_all_trials = [
        [None for _ in config["ETA_VALS"]] for _ in range(config["LR_TRIALS"])
    ]
    for trial, eta, val_mse in results:
        eta_idx = config["ETA_VALS"].index(eta)
        val_mse_all_trials[trial][eta_idx] = val_mse

    avg_val_mse = np.mean(val_mse_all_trials, axis=0)
    best_idx = np.argmin(avg_val_mse)
    best_eta = config["ETA_VALS"][best_idx]

    return best_eta, {
        "eta_vals": config["ETA_VALS"],
        "avg_val_mse": avg_val_mse,
        "std_val_mse": np.std(val_mse_all_trials, axis=0),
        "best_eta": best_eta,
        "best_val_mse": avg_val_mse[best_idx],
        "all_trials": val_mse_all_trials,
    }


def _interp_log(points: List[int], values: Dict[int, float], target: int) -> float:
    if target <= points[0]:
        return values[points[0]]
    if target >= points[-1]:
        return values[points[-1]]
    for low, high in zip(points[:-1], points[1:]):
        if low <= target <= high:
            v_low = values[low]
            v_high = values[high]
            frac = (target - low) / (high - low)
            return float(
                np.exp(np.log(v_low) + frac * (np.log(v_high) - np.log(v_low)))
            )
    return values[points[-1]]


def _get_eta_for_config(
    lr_table: Dict[str, Dict[int, Dict[int, float]]],
    activation_name: str,
    depth: int,
    width: int,
) -> float:
    depth_dict = lr_table[activation_name]
    depth_keys = sorted(depth_dict.keys())

    def eta_at_depth(d_key: int) -> float:
        width_dict = depth_dict[d_key]
        width_keys = sorted(width_dict.keys())
        return _interp_log(width_keys, width_dict, width)

    if depth <= depth_keys[0]:
        return eta_at_depth(depth_keys[0])
    if depth >= depth_keys[-1]:
        return eta_at_depth(depth_keys[-1])
    for d_low, d_high in zip(depth_keys[:-1], depth_keys[1:]):
        if d_low <= depth <= d_high:
            eta_low = eta_at_depth(d_low)
            eta_high = eta_at_depth(d_high)
            frac = (depth - d_low) / (d_high - d_low)
            return float(
                np.exp(np.log(eta_low) + frac * (np.log(eta_high) - np.log(eta_low)))
            )
    return eta_at_depth(depth_keys[-1])


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def _train_model_instance(
    activation_class,
    n_layers: int,
    n_nodes: int,
    learning_rate: float,
    trial_seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    use_batch_norm: bool,
    track_per_layer: bool = False,
) -> Tuple[NeuralNetwork, Dict[str, np.ndarray]]:
    layer_sizes = [n_nodes] * n_layers + [1]
    activations = [activation_class() for _ in range(n_layers)] + [Linear()]
    model = NeuralNetwork(
        network_input_size=X_train.shape[1],
        layer_output_sizes=layer_sizes,
        activations=activations,
        loss=MSE(),
        seed=trial_seed,
        use_batch_norm=use_batch_norm,
    )
    history = train(
        nn=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer=Adam(eta=learning_rate, update_batch_norm_params=use_batch_norm),
        epochs=epochs,
        batch_size=batch_size,
        stochastic=True,
        task="regression",
        early_stopping=False,
        verbose=False,
        seed=trial_seed,
        track_per_layer_gradients=track_per_layer,
    )
    return model, history


def _train_single_model(
    activation_name: str,
    activation_class,
    n_layers: int,
    n_nodes: int,
    learning_rate: float,
    trial_seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_batch_norm: bool,
    epochs: int,
    batch_size: int,
    track_per_layer: bool = False,
) -> Dict[str, np.ndarray]:
    model, history = _train_model_instance(
        activation_class,
        n_layers,
        n_nodes,
        learning_rate,
        trial_seed,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_batch_norm=use_batch_norm,
        track_per_layer=track_per_layer,
    )
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    n_params = sum(W.size + b.size for W, b in model.layers)

    if use_batch_norm and model.batch_norm_layers:
        for bn in model.batch_norm_layers:
            n_params += bn.gamma.size + bn.beta.size

    grad_history = np.array(history.get("gradient_norms", []), dtype=float)
    valid_grad = grad_history[~np.isnan(grad_history)]
    grad_norm_final = float(valid_grad[-1]) if valid_grad.size else np.nan
    grad_norm_mean = float(np.nanmean(valid_grad)) if valid_grad.size else np.nan

    result = {
        "activation": activation_name,
        "n_layers": n_layers,
        "n_nodes": n_nodes,
        "n_params": n_params,
        "train_mse": float(np.mean((y_train - y_train_pred) ** 2)),
        "val_mse": float(np.mean((y_val - y_val_pred) ** 2)),
        "test_mse": float(np.mean((y_test - y_test_pred) ** 2)),
        "train_history": history["train_loss"],
        "val_history": history["val_loss"],
        "grad_history": grad_history,
        "final_epoch": history["final_epoch"],
        "learning_rate": learning_rate,
        "grad_norm_final": grad_norm_final,
        "grad_norm_mean": grad_norm_mean,
        "trial_seed": trial_seed,
        "batch_norm": use_batch_norm,
    }

    if track_per_layer and history.get("per_layer_gradient_norms") is not None:
        result["per_layer_grad_history"] = history["per_layer_gradient_norms"]
        if history.get("per_layer_gradient_stds") is not None:
            result["per_layer_grad_std_history"] = history["per_layer_gradient_stds"]
        if history.get("per_layer_dead_fraction") is not None:
            result["per_layer_dead_history"] = history["per_layer_dead_fraction"]

    return result


def _run_architecture_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depth_eta: Dict[str, Dict[int, Dict[int, float]]],
    activation_spaces: Dict[str, Dict[str, List[int]]],
    n_trials: int,
    n_jobs: int,
    use_batch_norm: bool,
    track_per_layer: bool = False,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    jobs = []
    for activation_name, activation_class in ACTIVATIONS.items():
        for n_layers in activation_spaces[activation_name]["layers"]:
            for n_nodes in activation_spaces[activation_name]["nodes"]:
                lr = _get_eta_for_config(depth_eta, activation_name, n_layers, n_nodes)
                for trial in range(n_trials):
                    jobs.append(
                        (
                            activation_name,
                            activation_class,
                            n_layers,
                            n_nodes,
                            lr,
                            SEED + trial,
                        )
                    )

    def _worker(args):
        (
            activation_name,
            activation_class,
            n_layers,
            n_nodes,
            learning_rate,
            trial_seed,
        ) = args
        return _train_single_model(
            activation_name,
            activation_class,
            n_layers,
            n_nodes,
            learning_rate,
            trial_seed,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            use_batch_norm=use_batch_norm,
            track_per_layer=track_per_layer,
            epochs=epochs,
            batch_size=batch_size,
        )

    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_worker)(job) for job in jobs)
    df_trials = pd.DataFrame(results)
    df_trials["noise_floor"] = NOISE_FLOOR
    agg = (
        df_trials.groupby(["activation", "n_layers", "n_nodes"])
        .agg(
            n_params=("n_params", "first"),
            train_mse_mean=("train_mse", "mean"),
            train_mse_std=("train_mse", "std"),
            val_mse_mean=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            learning_rate=("learning_rate", "first"),
            n_trials=("train_mse", "count"),
            grad_norm_final_mean=("grad_norm_final", "mean"),
            grad_norm_final_std=("grad_norm_final", "std"),
            grad_norm_mean_mean=("grad_norm_mean", "mean"),
            grad_norm_mean_std=("grad_norm_mean", "std"),
            batch_norm=("batch_norm", "first"),
        )
        .reset_index()
    )
    agg["overfitting_gap"] = agg["val_mse_mean"] - agg["train_mse_mean"]
    agg["overfitting_ratio"] = agg["val_mse_mean"] / agg["train_mse_mean"]
    return pd.concat([agg, df_trials], ignore_index=True, sort=False)


# -----------------------------------------------------------------------------
# Split execution
# -----------------------------------------------------------------------------


def _run_single_split(
    split_seed: int, use_batch_norm: bool, epochs: int, batch_size: int
) -> pd.DataFrame:
    bn_label = "WITH_BN" if use_batch_norm else "NO_BN"
    print("=" * 80)
    print(f"Running split seed {split_seed} ({bn_label})")
    print("=" * 80)
    data = _prepare_data(split_seed)
    depth_eta = _calibrate_learning_rates(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        use_batch_norm=use_batch_norm,
    )
    activation_spaces = {
        name: {"nodes": N_NODES_DEFAULT, "layers": N_LAYERS_DEFAULT}
        for name in ACTIVATIONS.keys()
    }

    track_layers = TRACK_PER_LAYER
    if track_layers:
        print(
            f"\n=== Running architecture search with per-layer tracking ({bn_label}) ==="
        )
    else:
        print(f"\n=== Running architecture search ({bn_label}) ===")

    n_jobs = max(
        1, (os.cpu_count() - 1) if os.cpu_count() and os.cpu_count() > 2 else 1
    )
    df_combined = _run_architecture_search(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        depth_eta,
        activation_spaces,
        N_TRIALS,
        n_jobs,
        use_batch_norm=use_batch_norm,
        track_per_layer=track_layers,
        epochs=epochs,
        batch_size=batch_size,
    )
    df_combined["noise_floor"] = NOISE_FLOOR

    seed_dir = RESULTS_DIR / f"seed_{split_seed}_{bn_label}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = seed_dir / "results.csv"

    df_to_save = df_combined.drop(
        columns=[
            "per_layer_grad_history",
            "per_layer_grad_std_history",
            "per_layer_dead_history",
        ],
        errors="ignore",
    )
    df_to_save.to_csv(csv_path, index=False)

    if "per_layer_grad_history" in df_combined.columns:
        per_layer_data = df_combined[
            df_combined["per_layer_grad_history"].notna()
        ].copy()
    else:
        per_layer_data = pd.DataFrame()

    if not per_layer_data.empty:
        import pickle

        per_layer_path = seed_dir / "per_layer_gradients.pkl"
        with open(per_layer_path, "wb") as f:
            pickle.dump(
                per_layer_data[
                    [
                        "activation",
                        "n_layers",
                        "n_nodes",
                        "trial_seed",
                        "per_layer_grad_history",
                        "per_layer_grad_std_history",
                        "per_layer_dead_history",
                    ]
                ].to_dict("records"),
                f,
            )
        print(f"Saved per-layer gradient data to {per_layer_path}")

    return df_combined


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------


def load_part_d_results(
    use_batch_norm: bool,
    settings: PlotSettings = PLOT_SETTINGS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    bn_label = "WITH_BN" if use_batch_norm else "NO_BN"
    combined_path = RESULTS_DIR / f"results_combined_{bn_label}.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Combined results not found at {combined_path}. Run main_d() first."
        )
    df_raw = pd.read_csv(combined_path)
    df_plot = aggregate_results_for_plotting(df_raw, settings)

    import pickle

    per_layer_records: List[dict] = []
    pattern = f"seed_*_{bn_label}"
    for seed_dir in RESULTS_DIR.glob(pattern):
        per_layer_path = seed_dir / "per_layer_gradients.pkl"
        if per_layer_path.exists():
            with open(per_layer_path, "rb") as f:
                per_layer_records.extend(pickle.load(f))
    return df_raw, df_plot, per_layer_records


def generate_part_d_plots_from_disk(
    use_batch_norm: bool,
    settings: PlotSettings = PLOT_SETTINGS,
    per_layer_width: int | None = None,
    per_layer_epochs: int | None = None,
    learning_curve_configs: Dict[str, Tuple[int, int]] | None = None,
    show_figures: bool = False,
) -> None:
    df_raw, df_plot, per_layer_records = load_part_d_results(
        use_batch_norm=use_batch_norm, settings=settings
    )
    bn_label = "WITH_BN" if use_batch_norm else "NO_BN"
    fig_dir = FIGURES_DIR / f"combined_{bn_label}"
    generate_part_d_plots(
        df_plot,
        df_raw,
        per_layer_records,
        fig_dir,
        settings,
        use_batch_norm,
        per_layer_width=per_layer_width,
        per_layer_epochs=per_layer_epochs,
        learning_curve_configs=learning_curve_configs,
        show_figures=show_figures,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def main_d(
    seeds: Sequence[int] | None = None,
    use_batch_norm: bool | None = None,
    skip_plots: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
    show_plots: bool = False,
    learning_curve_configs: Dict[str, Tuple[int, int]] | None = None,
) -> Path:
    seeds = list(seeds) if seeds is not None else DEFAULT_SPLIT_SEEDS
    use_bn = USE_BATCH_NORM if use_batch_norm is None else use_batch_norm
    epochs_val = epochs if epochs is not None else EPOCHS
    batch_val = batch_size if batch_size is not None else BATCH_SIZE
    bn_label = "WITH_BN" if use_bn else "NO_BN"

    combined_results: List[pd.DataFrame] = []
    for split_seed in seeds:
        df_seed = _run_single_split(
            split_seed,
            use_batch_norm=use_bn,
            epochs=epochs_val,
            batch_size=batch_val,
        )
        combined_results.append(df_seed)

    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_path = RESULTS_DIR / f"results_combined_{bn_label}.csv"

    df_to_save = combined_df.drop(columns=["per_layer_grad_history"], errors="ignore")
    df_to_save.to_csv(combined_path, index=False)

    import pickle

    per_layer_records = []
    for split_seed in seeds:
        seed_dir = RESULTS_DIR / f"seed_{split_seed}_{bn_label}"
        per_layer_path = seed_dir / "per_layer_gradients.pkl"
        if per_layer_path.exists():
            with open(per_layer_path, "rb") as f:
                per_layer_records.extend(pickle.load(f))

    if not skip_plots:
        combined_fig_dir = FIGURES_DIR / f"combined_{bn_label}"
        df_plot_combined = aggregate_results_for_plotting(combined_df, PLOT_SETTINGS)
        generate_part_d_plots(
            df_plot_combined,
            combined_df,
            per_layer_records,
            combined_fig_dir,
            PLOT_SETTINGS,
            use_bn,
            per_layer_width=None,
            per_layer_epochs=None,
            learning_curve_configs=learning_curve_configs,
            show_figures=show_plots,
        )

    print(
        f"\nAll splits completed ({bn_label}). Combined results saved to {combined_path}"
    )
    return combined_path


if __name__ == "__main__":
    main_d()
