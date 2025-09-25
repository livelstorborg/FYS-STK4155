from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .utils import (
    analytical_solution,
    gradient_descent,  
    gd_momentum,
    gd_adagrad,
    gd_rmsprop,
    gd_adam,
    # Add stochastic imports
    stochastic_gd,
    stochastic_gd_momentum,
    stochastic_gd_adagrad,
    stochastic_gd_rmsprop,
    stochastic_gd_adam,
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
            # Stochastic optimizers
            "sgd": stochastic_gd,
            "sgd_momentum": stochastic_gd_momentum,
            "sgd_adagrad": stochastic_gd_adagrad,
            "sgd_rmsprop": stochastic_gd_rmsprop,
            "sgd_adam": stochastic_gd_adam,
        }
        
        # Track which optimizers are stochastic
        self._stochastic_opts = {
            "sgd", "sgd_momentum", "sgd_adagrad", "sgd_rmsprop", "sgd_adam"
        }

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
        return opt in self._opt_fns

    # ----------------- public API -----------------

    def fit_one(self, model: str, opt: str, **opt_kwargs) -> None:
        """
        Fit a single (model, opt) combo.
        model in {'ols','ridge','lasso'}
        opt   in {'analytical','gd','momentum','adagrad','rmsprop','adam',
                  'sgd','sgd_momentum','sgd_adagrad','sgd_rmsprop','sgd_adam'}
        Extra optimizer-specific kwargs can be passed via **opt_kwargs.
        """
        if not self._valid_combo(model, opt):
            raise ValueError(f"Invalid combo: ({model}, {opt})")

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
            # Stochastic optimizer - different parameter structure
            if self.eta is None or self.n_epochs is None:
                raise ValueError("eta and n_epochs required for stochastic methods")
            
            lam_arg = self.lam if model in ("ridge", "lasso") else None
            theta, history = self._opt_fns[opt](
                self.X_train,
                self.y_train,
                method=model,
                lam=lam_arg,
                eta=self.eta,
                n_epochs=self.n_epochs,
                batch_size=opt_kwargs.get("batch_size", self.batch_size),
                random_state=opt_kwargs.get("random_state", self.random_state),
                **{k: v for k, v in opt_kwargs.items() 
                   if k not in ["batch_size", "random_state"]}
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

    def fit_many(
        self,
        models: Optional[Tuple[str, ...]] = None,
        opts: Optional[Tuple[str, ...]] = None,
        **opt_kwargs,
    ) -> None:
        """
        Fit a grid of (models x opts).
        Example:
            ra.fit_many(models=('ols','ridge','lasso'),
                        opts=('analytical','gd','adam','sgd','sgd_adam'),
                        beta=0.9, eps=1e-8, batch_size=64)
        Any optimizer-specific parameters can be included in **opt_kwargs; they'll be ignored by
        optimizers that don't use them.
        """
        models = models or self._models
        opts = opts or ("analytical", "gd", "momentum", "adagrad", "rmsprop", "adam")
        for model in models:
            for opt in opts:
                if self._valid_combo(model, opt):
                    self.fit_one(model, opt, **opt_kwargs)

    def get_metric(self, model: str, opt: str, name: str) -> float:
        """
        name in {'train_mse','train_r2','test_mse','test_r2'}
        """
        key = (model, opt)
        if key not in self.runs:
            raise ValueError(f"Not fitted: {key}")
        return self.runs[key]["metrics"][name]

    def get_theta(self, model: str, opt: str) -> np.ndarray:
        key = (model, opt)
        if key not in self.runs:
            raise ValueError(f"Not fitted: {key}")
        return self.runs[key]["theta"]

    def fitted(self):
        return list(self.runs.keys())

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns dict keyed by 'model+opt' like 'ols+gd' with core metrics + theta shape.
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