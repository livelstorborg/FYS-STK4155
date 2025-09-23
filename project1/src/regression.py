import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from .utils import Ridge_parameters, OLS_parameters, gd_OLS, gd_Ridge


class RegressionAnalysis:
    """
    A class to perform regression analysis on polynomial features.

    Each instance represents one complete regression analysis with specific parameters:
    - polynomial degree
    - regularization parameter (lambda)
    - learning rate (eta)
    - number of iterations for gradient descent

    The class stores both analytical and gradient descent solutions,
    along with predictions and performance metrics.
    """

    def __init__(self, data, degree, lam=None, eta=None, num_iters=None):
        """
        Initialize regression analysis instance.

        Parameters
        ----------
        data: list
            List containing [X_train, X_test, y_train, y_test, x_train, x_test, y_mean]
        degree : int
            Polynomial degree used
        lam : float
            Regularization parameter for Ridge regression
        eta : float
            Learning rate for gradient descent
        num_iters : int
            Number of iterations for gradient descent
        """

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.x_train,
            self.x_test,
            self.y_mean,
        ) = data

        self.degree = degree
        self.lam = lam
        self.eta = eta
        self.num_iters = num_iters

        # ---------- Optimal parameters ---------
        # Analytical
        self.theta_ols_analytical = None
        self.theta_ridge_analytical = None

        # Gradient Descent
        self.theta_ols_gd = None
        self.theta_ridge_gd = None

        # Gradient Descent with momentum
        self.theta_ols_gd_mom = None
        self.theta_ridge_gd_mom = None

        # Gradient Descent - ADAgrad
        self.theta_ols_gd_adagrad = None
        self.theta_ridge_gd_adagrad = None

        # Gradient Descent - RMSprop
        self.theta_ols_gd_rmsprop = None
        self.theta_ridge_gd_rmsprop = None

        # Gradient Descent - ADAM
        self.theta_ols_gd_adam = None
        self.theta_ridge_gd_adam = None

        # ---------- Predictions ----------
        self.y_pred_ols_analytical = None
        self.y_pred_ridge_analytical = None

        self.y_pred_ols_gd = None
        self.y_pred_ridge_gd = None

        self.y_pred_ols_gd_mom = None
        self.y_pred_ridge_gd_mom = None

        self.y_pred_ols_gd_adagrad = None
        self.y_pred_ridge_gd_adagrad = None

        self.y_pred_ols_gd_rmsprop = None
        self.y_pred_ridge_gd_rmsprop = None

        self.y_pred_ols_gd_adam = None
        self.y_pred_ridge_gd_adam = None

        # ---------- Performance Metrics ----------
        self.mse_ols_analytical = None
        self.mse_ridge_analytical = None
        self.mse_ols_gd = None
        self.mse_ridge_gd = None
        self.r2_ols_analytical = None
        self.r2_ridge_analytical = None
        self.r2_ols_gd = None
        self.r2_ridge_gd = None

        # ---------- Training Metrics ----------
        self.train_mse_ols_analytical = None
        self.train_r2_ols_analytical = None
        self.train_mse_ridge_analytical = None
        self.train_r2_ridge_analytical = None
        self.train_mse_ols_gd = None
        self.train_r2_ols_gd = None
        self.train_mse_ridge_gd = None
        self.train_r2_ridge_gd = None

        # ---------- MSE History ----------
        self.ols_mse_history = None
        self.ridge_mse_history = None

        self.ols_mse_history_mom = None
        self.ridge_mse_history_mom = None

        self.ols_mse_history_adagrad = None
        self.ridge_mse_history_adagrad = None

        self.ols_mse_history_rmsprop = None
        self.ridge_mse_history_rmsprop = None

        self.ols_mse_history_adam = None
        self.ridge_mse_history_adam = None

    def fit_analytical(self):
        """Fit both OLS and Ridge regression using analytical solutions."""
        # OLS analytical solution (always calculate)
        self.theta_ols_analytical = OLS_parameters(self.X_train, self.y_train)

        # Ridge analytical solution (only if lambda is set and > 0)
        if self.lam is not None and self.lam > 0:
            self.theta_ridge_analytical = Ridge_parameters(
                self.X_train, self.y_train, self.lam
            )
        elif self.lam is not None and self.lam == 0:
            print("Warning: lam=0 is equivalent to OLS. Use OLS methods instead.")
        elif self.lam is None:
            print("Warning: lam not set. Skipping Ridge analytical solution.")

    def fit_gradient_descent(self):
        """Fit both OLS and Ridge regression using gradient descent."""
        if self.eta is None or self.num_iters is None:
            print("Warning: eta or num_iters not set. Skipping gradient descent.")
            return

        # Fit OLS with gradient descent
        self.theta_ols_gd, self.ols_mse_history = gd_OLS(
            self.X_train, self.y_train, self.eta, self.num_iters
        )

        # Fit Ridge with gradient descent (only if lambda is set and > 0)
        if self.lam is not None and self.lam > 0:
            self.theta_ridge_gd, self.ridge_mse_history = gd_Ridge(
                self.X_train,
                self.y_train,
                self.eta,
                self.num_iters,
                self.lam,
                stopping_param=1e-6,
            )
        elif self.lam is not None and self.lam == 0:
            print("Warning: lam=0 for Ridge GD is equivalent to OLS GD.")
        elif self.lam is None:
            print("Warning: lam not set. Skipping Ridge gradient descent.")

    def predict(self):
        """Generate predictions on test set for all fitted models."""
        if self.theta_ols_analytical is not None:
            self.y_pred_ols_analytical = (
                self.X_test @ self.theta_ols_analytical + self.y_mean
            )

        if self.theta_ridge_analytical is not None:
            self.y_pred_ridge_analytical = (
                self.X_test @ self.theta_ridge_analytical + self.y_mean
            )

        if self.theta_ols_gd is not None:
            self.y_pred_ols_gd = self.X_test @ self.theta_ols_gd + self.y_mean

        if self.theta_ridge_gd is not None:
            self.y_pred_ridge_gd = self.X_test @ self.theta_ridge_gd + self.y_mean

    def calculate_metrics(self):
        """Calculate MSE and R² scores for all predictions (both train and test)."""
        # True test values (unscaled)
        y_test_true = self.y_test + self.y_mean
        # True train values (unscaled)
        y_train_true = self.y_train + self.y_mean

        # OLS Analytical
        if self.y_pred_ols_analytical is not None:
            # Test metrics
            self.mse_ols_analytical = mean_squared_error(
                y_test_true, self.y_pred_ols_analytical
            )
            self.r2_ols_analytical = r2_score(y_test_true, self.y_pred_ols_analytical)
            # Train metrics
            y_train_pred_ols = self.X_train @ self.theta_ols_analytical + self.y_mean
            self.train_mse_ols_analytical = mean_squared_error(
                y_train_true, y_train_pred_ols
            )
            self.train_r2_ols_analytical = r2_score(y_train_true, y_train_pred_ols)

        # Ridge Analytical
        if self.y_pred_ridge_analytical is not None:
            # Test metrics
            self.mse_ridge_analytical = mean_squared_error(
                y_test_true, self.y_pred_ridge_analytical
            )
            self.r2_ridge_analytical = r2_score(
                y_test_true, self.y_pred_ridge_analytical
            )
            # Train metrics
            y_train_pred_ridge = (
                self.X_train @ self.theta_ridge_analytical + self.y_mean
            )
            self.train_mse_ridge_analytical = mean_squared_error(
                y_train_true, y_train_pred_ridge
            )
            self.train_r2_ridge_analytical = r2_score(y_train_true, y_train_pred_ridge)

        # OLS Gradient Descent
        if self.y_pred_ols_gd is not None:
            # Test metrics
            self.mse_ols_gd = mean_squared_error(y_test_true, self.y_pred_ols_gd)
            self.r2_ols_gd = r2_score(y_test_true, self.y_pred_ols_gd)
            # Train metrics
            y_train_pred_ols_gd = self.X_train @ self.theta_ols_gd + self.y_mean
            self.train_mse_ols_gd = mean_squared_error(
                y_train_true, y_train_pred_ols_gd
            )
            self.train_r2_ols_gd = r2_score(y_train_true, y_train_pred_ols_gd)

        # Ridge Gradient Descent
        if self.y_pred_ridge_gd is not None:
            # Test metrics
            self.mse_ridge_gd = mean_squared_error(y_test_true, self.y_pred_ridge_gd)
            self.r2_ridge_gd = r2_score(y_test_true, self.y_pred_ridge_gd)
            # Train metrics
            y_train_pred_ridge_gd = self.X_train @ self.theta_ridge_gd + self.y_mean
            self.train_mse_ridge_gd = mean_squared_error(
                y_train_true, y_train_pred_ridge_gd
            )
            self.train_r2_ridge_gd = r2_score(y_train_true, y_train_pred_ridge_gd)

    def fit_all(self):
        """Convenience method to fit all models and calculate all metrics."""
        self.fit_analytical()
        self.fit_gradient_descent()
        self.predict()
        self.calculate_metrics()

    def print_results(self):
        """Print MSE and R² results in a formatted table."""
        print(f"\nResults for Degree {self.degree}, λ={self.lam}, η={self.eta}")
        print("=" * 60)
        print(f"{'Method':<20} {'MSE':<12} {'R²':<12}")
        print("-" * 60)

        if self.mse_ols_analytical is not None:
            print(
                f"{'OLS (Analytical)':<20} {self.mse_ols_analytical:<12.6f} {self.r2_ols_analytical:<12.6f}"
            )
        if self.mse_ols_gd is not None:
            print(
                f"{'OLS (Grad. Desc.)':<20} {self.mse_ols_gd:<12.6f} {self.r2_ols_gd:<12.6f}"
            )
        if self.mse_ridge_analytical is not None:
            print(
                f"{'Ridge (Analytical)':<20} {self.mse_ridge_analytical:<12.6f} {self.r2_ridge_analytical:<12.6f}"
            )
        if self.mse_ridge_gd is not None:
            print(
                f"{'Ridge (Grad. Desc.)':<20} {self.mse_ridge_gd:<12.6f} {self.r2_ridge_gd:<12.6f}"
            )

    def get_train_mse(self, method="ols_analytical"):
        """
        Calculate training MSE for a specific method.

        Parameters
        ----------
        method : str
            Method to calculate MSE for ('ols_analytical', 'ridge_analytical',
            'ols_gd', 'ridge_gd')

        Returns
        -------
        float
            Training MSE
        """

        if method == "ols_analytical":
            if self.theta_ols_analytical is None:
                raise ValueError("OLS analytical solution not fitted")
            y_train_pred = self.X_train @ self.theta_ols_analytical + self.y_mean

        elif method == "ridge_analytical":
            if self.theta_ridge_analytical is None:
                raise ValueError("Ridge analytical solution not fitted")
            y_train_pred = self.X_train @ self.theta_ridge_analytical + self.y_mean

        elif method == "ols_gd":
            if self.theta_ols_gd is None:
                raise ValueError("OLS gradient descent solution not fitted")
            y_train_pred = self.X_train @ self.theta_ols_gd + self.y_mean

        elif method == "ridge_gd":
            if self.theta_ridge_gd is None:
                raise ValueError("Ridge gradient descent solution not fitted")
            y_train_pred = self.X_train @ self.theta_ridge_gd + self.y_mean

        else:
            raise ValueError(f"Unknown method: {method}")

        # True training values (unscaled)
        y_train_true = self.y_train + self.y_mean

        return mean_squared_error(y_train_true, y_train_pred)

    def get_train_r2(self, method="ols_analytical"):
        """Calculate training R² for a specific method."""

        if method == "ols_analytical":
            if self.theta_ols_analytical is None:
                raise ValueError("OLS analytical solution not fitted")
            y_train_pred = self.X_train @ self.theta_ols_analytical + self.y_mean

        elif method == "ridge_analytical":
            if self.theta_ridge_analytical is None:
                raise ValueError("Ridge analytical solution not fitted")
            y_train_pred = self.X_train @ self.theta_ridge_analytical + self.y_mean

        y_train_true = self.y_train + self.y_mean
        return r2_score(y_train_true, y_train_pred)
