from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .utils import (
    analytical_solution,
    gradient_descent,  
    gd_momentum,
    gd_adagrad,
    gd_rmsprop,
    gd_adam
)


class RegressionAnalysis:
    """
    Compact regression runner for OLS/Ridge/Lasso across multiple optimizers.

    Access results via:
      - ra.runs[('ols','gd')]['theta'] / ['history'] / ['metrics'] / ['y_pred_test'] / ['y_pred_train']
      - ra.get_metric('ols','gd','test_mse')   # convenient helper
      - ra.summary()                           # quick overview dict
    """

    def __init__(
        self, 
        data, 
        degree, 
        lam=0.0, 
        eta=0.01, 
        num_iters=1000, 
        full_dataset=False,
        # Add stochastic parameters
        batch_size=32,
        n_epochs=50,
        random_state=None
    ):
        if full_dataset:
            # data = [X_full, y_full, y_mean] (already centered)
            self.X_full, self.y_full, self.y_mean = data
            self.X_train = self.X_test = self.X_full
            self.y_train = self.y_test = self.y_full
            self.y_offset = self.y_mean
        else:
            # data = [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]
            self.X_train, self.X_test, self.y_train, self.y_test, *_rest = data
            self.y_offset = _rest[-1]  # y_mean

        self.degree = degree
        self.lam = lam
        self.eta = eta
        self.num_iters = num_iters
        # Add stochastic parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Where everything lands: runs[(model, opt)] -> dict of artifacts
        self.runs: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Available models/opts - add stochastic optimizers
        self._models = ("ols", "ridge", "lasso")
        self._opt_fns = {
            # Full-batch optimizers
            "gd": gradient_descent,
            "momentum": gd_momentum,
            "adagrad": gd_adagrad,
            "rmsprop": gd_rmsprop,
            "adam": gd_adam,
            # Stochastic optimizers (use same functions with stochastic=True)
            "sgd": gradient_descent,
            "sgd_momentum": gd_momentum,
            "sgd_adagrad": gd_adagrad,
            "sgd_rmsprop": gd_rmsprop,
            "sgd_adam": gd_adam,
        }
        
        # Track which optimizers are stochastic
        self._stochastic_opts = {"sgd", "sgd_momentum", "sgd_adagrad", "sgd_rmsprop", "sgd_adam"}

    # ----------------- internals -----------------

    def _predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return X @ theta

    def _compute_metrics(self, theta: np.ndarray) -> Dict[str, Any]:
        # test
        y_test_true = self.y_test + self.y_offset
        y_pred_test = self._predict(self.X_test, theta) + self.y_offset
        test_mse = mean_squared_error(y_test_true, y_pred_test)
        test_r2 = r2_score(y_test_true, y_pred_test)
        # train
        y_train_true = self.y_train + self.y_offset
        y_pred_train = self._predict(self.X_train, theta) + self.y_offset
        train_mse = mean_squared_error(y_train_true, y_pred_train)
        train_r2 = r2_score(y_train_true, y_pred_train)
        return {
            "test_mse": test_mse,
            "test_r2": test_r2,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "y_pred_test": y_pred_test,
            "y_pred_train": y_pred_train,
        }

    def _valid_combo(self, model: str, opt: str) -> bool:
        if model not in self._models:
            return False
        if opt == "analytical":
            return model in ("ols", "ridge")  # no analytical lasso
        return opt in self._opt_fns or opt in self._stochastic_opts

    # ----------------- public API -----------------

    def fit(
        self,
        models: Union[str, Tuple[str, ...], None] = None,
        opts: Union[str, Tuple[str, ...], None] = None,
        **opt_kwargs,
    ) -> None:
        """
        Fit model(s) with optimizer(s). Handles both single and multiple combinations.
        
        Parameters
        ----------
        models : str or tuple of str, optional
            Single model or tuple of models in {'ols','ridge','lasso'}
            If None, uses all available models
        opts : str or tuple of str, optional  
            Single optimizer or tuple of optimizers in 
            {'analytical','gd','momentum','adagrad','rmsprop','adam',
            'sgd','sgd_momentum','sgd_adagrad','sgd_rmsprop','sgd_adam'}
            If None, uses default optimizers
        **opt_kwargs
            Optimizer-specific parameters (e.g., beta=0.9, eps=1e-8, batch_size=64)
        """
        # Handle defaults
        if models is None:
            models = self._models
        if opts is None:
            opts = ("analytical", "gd", "momentum", "adagrad", "rmsprop", "adam")
        
        # Convert single strings to tuples for uniform processing
        if isinstance(models, str):
            models = (models,)
        if isinstance(opts, str):
            opts = (opts,)
        
        # Fit all combinations
        for model in models:
            for opt in opts:
                if not self._valid_combo(model, opt):
                    continue
                    
                # Solve for theta
                if opt == "analytical":
                    theta = analytical_solution(
                        self.X_train,
                        self.y_train,
                        method=model,
                        lam=(self.lam if model == "ridge" else 0.0),
                    )
                    history = None
                elif opt in self._stochastic_opts:
                    # Stochastic optimizer - use n_epochs instead of num_iters
                    if self.eta is None or self.n_epochs is None:
                        raise ValueError("eta and n_epochs required for stochastic methods")
                    
                    lam_arg = self.lam if model in ("ridge", "lasso") else None
                    theta, history = self._opt_fns[opt](
                        self.X_train,
                        self.y_train,
                        eta=self.eta,
                        num_iters=self.n_epochs,  # Pass n_epochs as num_iters
                        method=model,
                        lam=lam_arg,
                        stochastic=True,  # Enable stochastic mode
                        batch_size=opt_kwargs.get("batch_size", self.batch_size),
                        **{k: v for k, v in opt_kwargs.items() if k != "batch_size"}
                    )
                else:
                    # Full-batch optimizer - original parameter structure  
                    if self.eta is None or self.num_iters is None:
                        raise ValueError("eta and num_iters required for gradient-based methods")
                    lam_arg = self.lam if model in ("ridge", "lasso") else 0.0
                    theta, history = self._opt_fns[opt](
                        self.X_train,
                        self.y_train,
                        self.eta,
                        self.num_iters,
                        method=model,
                        lam=lam_arg,
                        **opt_kwargs,
                    )

                # Metrics + predictions
                m = self._compute_metrics(theta)
                key = (model, opt)
                self.runs[key] = {
                    "theta": theta,
                    "history": history,
                    "metrics": {
                        "train_mse": m["train_mse"],
                        "train_r2": m["train_r2"],
                        "test_mse": m["test_mse"],
                        "test_r2": m["test_r2"],
                    },
                    "y_pred_test": m["y_pred_test"],
                    "y_pred_train": m["y_pred_train"],
                }

    def get_metric(self, model: str, opt: str, name: str) -> float:
        """
        Get a specific metric for a model/optimizer combination.
        
        Parameters
        ----------
        model : str
            Model name in {'ols', 'ridge', 'lasso'}
        opt : str
            Optimizer name
        name : str
            Metric name in {'train_mse','train_r2','test_mse','test_r2'}
        
        Returns
        -------
        float
            The requested metric value
        """
        key = (model, opt)
        if key not in self.runs:
            raise ValueError(f"Not fitted: {key}")
        return self.runs[key]["metrics"][name]

    def get_theta(self, model: str, opt: str) -> np.ndarray:
        """
        Get the fitted parameters for a model/optimizer combination.
        
        Parameters
        ----------
        model : str
            Model name in {'ols', 'ridge', 'lasso'}
        opt : str
            Optimizer name
            
        Returns
        -------
        np.ndarray
            The fitted parameter vector
        """
        key = (model, opt)
        if key not in self.runs:
            raise ValueError(f"Not fitted: {key}")
        return self.runs[key]["theta"]

    def fitted(self):
        """
        Get list of fitted (model, optimizer) combinations.
        
        Returns
        -------
        list
            List of (model, optimizer) tuples that have been fitted
        """
        return list(self.runs.keys())

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all fitted models with key metrics.
        
        Returns
        -------
        dict
            Dictionary with fitted model summaries
        """
        out: Dict[str, Dict[str, Any]] = {}
        for (model, opt), pack in self.runs.items():
            name = f"{model}+{opt}"
            theta = pack["theta"]
            out[name] = {
                "train_mse": pack["metrics"]["train_mse"],
                "test_mse": pack["metrics"]["test_mse"],
                "train_r2": pack["metrics"]["train_r2"],
                "test_r2": pack["metrics"]["test_r2"],
                "theta_shape": None if theta is None else tuple(theta.shape),
                "has_history": pack["history"] is not None,
            }
        return out

    def __repr__(self) -> str:
        fitted = ", ".join([f"{m}+{o}" for (m, o) in self.runs]) or "None"
        return f"RegressionAnalysis(degree={self.degree}, fitted=[{fitted}])"