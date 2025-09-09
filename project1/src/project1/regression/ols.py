import numpy as np


class OLS:
    """Ordinary Least Squares regression.

    Provides both a simple estimator API (fit/predict) and static utility
    methods that mirror the previous function-based API.
    """

    def __init__(
        self,
        *,
        method: str = "closed_form",
        eta: float | None = None,
        n_iter: int | None = None,
    ):
        self.method = method
        self.eta = eta
        self.n_iter = n_iter
        self.theta_: np.ndarray | None = None

    @staticmethod
    def parameters(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Closed-form OLS coefficients (may raise LinAlgError if singular)."""
        return np.linalg.inv(X.T @ X) @ X.T @ y

    @staticmethod
    def gradient_descent(
        X: np.ndarray, y: np.ndarray, eta: float, n_iter: int
    ) -> np.ndarray:
        """Estimate OLS coefficients using gradient descent."""
        n, m = X.shape
        theta = np.zeros(m)
        for _ in range(n_iter):
            gradient = (2 / n) * X.T @ (X @ theta - y)
            theta -= eta * gradient
        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLS":
        """Fit the model and store coefficients in theta_."""
        if self.method == "closed_form":
            self.theta_ = OLS.parameters(X, y)
        elif self.method == "gd":
            if self.eta is None or self.n_iter is None:
                raise ValueError("eta and n_iter must be set for method='gd'")
            self.theta_ = OLS.gradient_descent(X, y, eta=self.eta, n_iter=self.n_iter)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return X @ self.theta_


# Backward-compatible function wrappers
def OLS_parameters(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return OLS.parameters(X, y)


def gradient_descent_ols(
    X: np.ndarray, y: np.ndarray, eta: float, n_iter: int
) -> np.ndarray:
    return OLS.gradient_descent(X, y, eta=eta, n_iter=n_iter)






