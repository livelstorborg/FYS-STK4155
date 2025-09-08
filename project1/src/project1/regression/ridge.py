import numpy as np


class Ridge:
    """Ridge regression with closed-form and gradient-descent options."""

    def __init__(
        self,
        *,
        lam: float = 1.0,
        method: str = "closed_form",
        eta: float | None = None,
        n_iter: int | None = None,
    ):
        self.lam = lam
        self.method = method
        self.eta = eta
        self.n_iter = n_iter
        self.theta_: np.ndarray | None = None

    @staticmethod
    def parameters(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
        n_features = X.shape[1]
        I = np.eye(n_features)
        return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y

    @staticmethod
    def gradient_descent(
        X: np.ndarray, y: np.ndarray, lam: float, eta: float, n_iter: int
    ) -> np.ndarray:
        n, m = X.shape
        theta = np.zeros(m)
        for _ in range(n_iter):
            gradient = (2 / n) * (X.T @ (X @ theta - y)) + 2 * lam * theta
            theta -= eta * gradient
        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        if self.method == "closed_form":
            self.theta_ = Ridge.parameters(X, y, self.lam)
        elif self.method == "gd":
            if self.eta is None or self.n_iter is None:
                raise ValueError("eta and n_iter must be set for method='gd'")
            self.theta_ = Ridge.gradient_descent(
                X, y, lam=self.lam, eta=self.eta, n_iter=self.n_iter
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return X @ self.theta_


# Backward-compatible function wrappers
def ridge_parameters(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    return Ridge.parameters(X, y, lambda_)


def gradient_descent_ridge(
    X: np.ndarray, y: np.ndarray, lam: float, eta: float, n_iter: int
) -> np.ndarray:
    return Ridge.gradient_descent(X, y, lam=lam, eta=eta, n_iter=n_iter)
