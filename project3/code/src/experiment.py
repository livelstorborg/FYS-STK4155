import jax.nn as jnn
from src.nn.train import train_pinn
from src.nn.evaluation import compute_error_metrics

import jax.nn as jnn
import pandas as pd
from src.nn.train import train_pinn
from src.nn.evaluation import compute_error_metrics


def run_architecture_sweep(
    hidden_widths,
    num_hidden_layers,
    activation_fns,
    *,
    T=0.5,
    steps=5000,
    lr=1e-3,
    seeds=(0,),
    lambda_ic=10.0,
    lambda_bc=10.0,
    Nx_eval=100,
    Nt_eval=100,
):
    """
    Run a general architecture sweep over:
      - hidden_widths:        list[int]
      - num_hidden_layers:    list[int]
      - activation_fns:       dict[str, callable]

    Optional keyword arguments allow further control.

    Returns:
      A list of dictionaries with architecture specs and error metrics.
    """

    results = []

    for act_name, act_fn in activation_fns.items():
        for L in num_hidden_layers:
            for W in hidden_widths:
                layers = [2] + [W] * L + [1]
                activations = [act_fn] * (len(layers) - 2)

                L2_all = []
                Linf_all = []

                for seed in seeds:
                    model, losses = train_pinn(
                        layers=layers,
                        activations=activations,
                        steps=steps,
                        T=T,
                        lr=lr,
                        seed=seed,
                        lambda_ic=lambda_ic,
                        lambda_bc=lambda_bc,
                    )

                    L2, Linf = compute_error_metrics(
                        model,
                        Nx=Nx_eval,
                        Nt=Nt_eval,
                        T=T,
                    )
                    L2_all.append(L2)
                    Linf_all.append(Linf)

                result = {
                    "activation": act_name,
                    "hidden_layers": L,
                    "width": W,
                    "L2_rel_mean": float(sum(L2_all) / len(L2_all)),
                    "Linf_mean": float(sum(Linf_all) / len(Linf_all)),
                }

                print(
                    f"[act={act_name:6s}  L={L}  W={W:3d}]  "
                    f"L2_rel={result['L2_rel_mean']:.3e}, "
                    f"Linf={result['Linf_mean']:.3e}"
                )

                results.append(result)
    results = pd.DataFrame(results)

    return results
