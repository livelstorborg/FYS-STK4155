from sklearn.metrics import mean_squared_error, r2_score
from .utils import Ridge_parameters, OLS_parameters, gd_OLS, gd_Ridge
from typing import Optional, Dict, Any
import numpy as np


class RegressionAnalysis:
    """
    Clean regression analysis class with lazy initialization.
    No more massive variable declarations!
    """

    def __init__(self, data, degree, lam=None, eta=None, num_iters=None):
        # Unpack data
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.x_train,
            self.x_test,
            self.y_mean,
        ) = data

        # Store parameters
        self.degree = degree
        self.lam = lam
        self.eta = eta
        self.num_iters = num_iters

        # Internal storage for computed results (lazy initialization)
        self._results: Dict[str, Any] = {}
        self._fitted_methods = set()

    def _store_result(self, key: str, value: Any) -> None:
        """Store a computed result"""
        self._results[key] = value

    def _get_result(self, key: str, default=None) -> Any:
        """Get a stored result, return default if not found"""
        return self._results.get(key, default)

    def _predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Helper method for predictions"""
        return X @ theta

    def _calculate_metrics_for_method(
        self, method_name: str, theta: np.ndarray
    ) -> None:
        """Calculate and store metrics for a given method"""
        if theta is None:
            return

        # Test predictions and metrics
        y_test_true = self.y_test + self.y_mean
        y_pred_test = self._predict(self.X_test, theta) + self.y_mean
        mse_test = mean_squared_error(y_test_true, y_pred_test)
        r2_test = r2_score(y_test_true, y_pred_test)

        # Train predictions and metrics
        y_train_true = self.y_train + self.y_mean
        y_pred_train = self._predict(self.X_train, theta) + self.y_mean
        mse_train = mean_squared_error(y_train_true, y_pred_train)
        r2_train = r2_score(y_train_true, y_pred_train)

        # Store all results
        self._store_result(f"y_pred_{method_name}", y_pred_test)
        self._store_result(f"mse_{method_name}", mse_test)
        self._store_result(f"r2_{method_name}", r2_test)
        self._store_result(f"train_mse_{method_name}", mse_train)
        self._store_result(f"train_r2_{method_name}", r2_train)

    # ================= FITTING METHODS =================
    def fit_analytical(self):
        """Fit analytical solutions"""
        # OLS
        theta_ols = OLS_parameters(self.X_train, self.y_train)
        self._store_result("theta_ols_analytical", theta_ols)
        self._fitted_methods.add("ols_analytical")

        # Ridge (if lambda provided)
        if self.lam is not None and self.lam > 0:
            theta_ridge = Ridge_parameters(self.X_train, self.y_train, self.lam)
            self._store_result("theta_ridge_analytical", theta_ridge)
            self._fitted_methods.add("ridge_analytical")

    def fit_gradient_descent(self):
        """Fit gradient descent solutions"""
        if self.eta is None or self.num_iters is None:
            raise ValueError("eta and num_iters required for gradient descent")

        # OLS GD
        theta_ols_gd, history_ols = gd_OLS(
            self.X_train, self.y_train, self.eta, self.num_iters
        )
        self._store_result("theta_ols_gd", theta_ols_gd)
        self._store_result("ols_mse_history", history_ols)
        self._fitted_methods.add("ols_gd")

        # Ridge GD (if lambda provided)
        if self.lam is not None and self.lam > 0:
            theta_ridge_gd, history_ridge = gd_Ridge(
                self.X_train, self.y_train, self.eta, self.lam, self.num_iters
            )
            self._store_result("theta_ridge_gd", theta_ridge_gd)
            self._store_result("ridge_mse_history", history_ridge)
            self._fitted_methods.add("ridge_gd")

    def fit_lasso(self):
        """Fit Lasso regression (placeholder for now)"""
        if self.eta is None or self.num_iters is None or self.lam is None:
            raise ValueError("eta, num_iters, and lam required for Lasso")

        # TODO: Implement actual Lasso GD. For now, using Ridge as placeholder
        theta_lasso_gd, history_lasso = gd_Ridge(
            self.X_train, self.y_train, self.eta, self.num_iters, self.lam
        )
        self._store_result("theta_lasso_gd", theta_lasso_gd)
        self._store_result("lasso_mse_history", history_lasso)
        self._fitted_methods.add("lasso_gd")

    def predict(self):
        """Generate predictions for all fitted methods"""
        for method in self._fitted_methods:
            theta_key = f"theta_{method}"
            theta = self._get_result(theta_key)
            if theta is not None:
                self._calculate_metrics_for_method(method, theta)

    def calculate_metrics(self):
        """Calculate metrics (same as predict for backward compatibility)"""
        self.predict()

    def fit_all(self):
        """Fit all available methods"""
        self.fit_analytical()
        if self.eta is not None and self.num_iters is not None:
            self.fit_gradient_descent()
        if self.lam is not None:
            try:
                self.fit_lasso()
            except ValueError:
                pass  # Skip if parameters missing
        self.predict()

    # ================= PROPERTIES FOR BACKWARD COMPATIBILITY =================
    # These properties provide the same interface as the old class

    # Theta properties
    @property
    def theta_ols_analytical(self) -> Optional[np.ndarray]:
        return self._get_result("theta_ols_analytical")

    @theta_ols_analytical.setter
    def theta_ols_analytical(self, value):
        if value is not None:
            self._store_result("theta_ols_analytical", value)
            self._fitted_methods.add("ols_analytical")

    @property
    def theta_ridge_analytical(self) -> Optional[np.ndarray]:
        return self._get_result("theta_ridge_analytical")

    @theta_ridge_analytical.setter
    def theta_ridge_analytical(self, value):
        if value is not None:
            self._store_result("theta_ridge_analytical", value)
            self._fitted_methods.add("ridge_analytical")

    @property
    def theta_ols_gd(self) -> Optional[np.ndarray]:
        return self._get_result("theta_ols_gd")

    @theta_ols_gd.setter
    def theta_ols_gd(self, value):
        if value is not None:
            self._store_result("theta_ols_gd", value)
            self._fitted_methods.add("ols_gd")

    @property
    def theta_ridge_gd(self) -> Optional[np.ndarray]:
        return self._get_result("theta_ridge_gd")

    @theta_ridge_gd.setter
    def theta_ridge_gd(self, value):
        if value is not None:
            self._store_result("theta_ridge_gd", value)
            self._fitted_methods.add("ridge_gd")

    @property
    def theta_lasso_gd(self) -> Optional[np.ndarray]:
        return self._get_result("theta_lasso_gd")

    @theta_lasso_gd.setter
    def theta_lasso_gd(self, value):
        if value is not None:
            self._store_result("theta_lasso_gd", value)
            self._fitted_methods.add("lasso_gd")

    # Prediction properties
    @property
    def y_pred_ols_analytical(self) -> Optional[np.ndarray]:
        return self._get_result("y_pred_ols_analytical")

    @property
    def y_pred_ridge_analytical(self) -> Optional[np.ndarray]:
        return self._get_result("y_pred_ridge_analytical")

    @property
    def y_pred_ols_gd(self) -> Optional[np.ndarray]:
        return self._get_result("y_pred_ols_gd")

    @property
    def y_pred_ridge_gd(self) -> Optional[np.ndarray]:
        return self._get_result("y_pred_ridge_gd")

    @property
    def y_pred_lasso_gd(self) -> Optional[np.ndarray]:
        return self._get_result("y_pred_lasso_gd")

    # MSE properties
    @property
    def mse_ols_analytical(self) -> Optional[float]:
        return self._get_result("mse_ols_analytical")

    @property
    def mse_ridge_analytical(self) -> Optional[float]:
        return self._get_result("mse_ridge_analytical")

    @property
    def mse_ols_gd(self) -> Optional[float]:
        return self._get_result("mse_ols_gd")

    @property
    def mse_ridge_gd(self) -> Optional[float]:
        return self._get_result("mse_ridge_gd")

    @property
    def mse_lasso_gd(self) -> Optional[float]:
        return self._get_result("mse_lasso_gd")

    # R2 properties
    @property
    def r2_ols_analytical(self) -> Optional[float]:
        return self._get_result("r2_ols_analytical")

    @property
    def r2_ridge_analytical(self) -> Optional[float]:
        return self._get_result("r2_ridge_analytical")

    @property
    def r2_ols_gd(self) -> Optional[float]:
        return self._get_result("r2_ols_gd")

    @property
    def r2_ridge_gd(self) -> Optional[float]:
        return self._get_result("r2_ridge_gd")

    @property
    def r2_lasso_gd(self) -> Optional[float]:
        return self._get_result("r2_lasso_gd")

    # Training MSE properties
    @property
    def train_mse_ols_analytical(self) -> Optional[float]:
        return self._get_result("train_mse_ols_analytical")

    @property
    def train_mse_ridge_analytical(self) -> Optional[float]:
        return self._get_result("train_mse_ridge_analytical")

    @property
    def train_mse_ols_gd(self) -> Optional[float]:
        return self._get_result("train_mse_ols_gd")

    @property
    def train_mse_ridge_gd(self) -> Optional[float]:
        return self._get_result("train_mse_ridge_gd")

    @property
    def train_mse_lasso_gd(self) -> Optional[float]:
        return self._get_result("train_mse_lasso_gd")

    # Training R2 properties
    @property
    def train_r2_ols_analytical(self) -> Optional[float]:
        return self._get_result("train_r2_ols_analytical")

    @property
    def train_r2_ridge_analytical(self) -> Optional[float]:
        return self._get_result("train_r2_ridge_analytical")

    @property
    def train_r2_ols_gd(self) -> Optional[float]:
        return self._get_result("train_r2_ols_gd")

    @property
    def train_r2_ridge_gd(self) -> Optional[float]:
        return self._get_result("train_r2_ridge_gd")

    @property
    def train_r2_lasso_gd(self) -> Optional[float]:
        return self._get_result("train_r2_lasso_gd")

    # History properties
    @property
    def ols_mse_history(self) -> Optional[list]:
        return self._get_result("ols_mse_history")

    @property
    def ridge_mse_history(self) -> Optional[list]:
        return self._get_result("ridge_mse_history")

    @property
    def lasso_mse_history(self) -> Optional[list]:
        return self._get_result("lasso_mse_history")

    # ================= UTILITY METHODS =================
    def get_train_mse(self, method="ols_analytical") -> float:
        """Get training MSE for a specific method"""
        key = f"train_mse_{method}"
        result = self._get_result(key)
        if result is None:
            raise ValueError(f"{method} not fitted or metrics not calculated")
        return result

    def get_train_r2(self, method="ols_analytical") -> float:
        """Get training R2 for a specific method"""
        key = f"train_r2_{method}"
        result = self._get_result(key)
        if result is None:
            raise ValueError(f"{method} not fitted or metrics not calculated")
        return result

    def get_fitted_methods(self) -> list:
        """Get list of fitted methods"""
        return list(self._fitted_methods)

    def is_fitted(self, method: str) -> bool:
        """Check if a specific method is fitted"""
        return method in self._fitted_methods

    def get_results_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all results"""
        summary = {}
        for method in self._fitted_methods:
            summary[method] = {
                "train_mse": self._get_result(f"train_mse_{method}"),
                "test_mse": self._get_result(f"mse_{method}"),
                "train_r2": self._get_result(f"train_r2_{method}"),
                "test_r2": self._get_result(f"r2_{method}"),
                "theta_shape": self._get_result(f"theta_{method}").shape
                if self._get_result(f"theta_{method}") is not None
                else None,
            }
        return summary

    def __repr__(self) -> str:
        """String representation showing fitted methods"""
        fitted = ", ".join(self._fitted_methods) if self._fitted_methods else "None"
        return f"RegressionAnalysis(degree={self.degree}, fitted_methods=[{fitted}])"
