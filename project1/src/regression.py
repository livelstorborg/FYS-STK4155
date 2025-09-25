from sklearn.metrics import mean_squared_error, r2_score
from .utils import Ridge_parameters, OLS_parameters, gd_OLS, gd_Ridge


class RegressionAnalysis:
    def __init__(self, data, degree, lam=None, eta=None, num_iters=None):
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

        # Parameters
        self.theta_ols_analytical = None
        self.theta_ridge_analytical = None
        self.theta_ols_gd = None
        self.theta_ridge_gd = None

        # (Placeholders kept for compatibility)
        self.theta_ols_gd_mom = None
        self.theta_ridge_gd_mom = None
        self.theta_ols_gd_adagrad = None
        self.theta_ridge_gd_adagrad = None
        self.theta_ols_gd_rmsprop = None
        self.theta_ridge_gd_rmsprop = None
        self.theta_ols_gd_adam = None
        self.theta_ridge_gd_adam = None

        # Predictions
        self.y_pred_ols_analytical = None
        self.y_pred_ridge_analytical = None
        self.y_pred_ols_gd = None
        self.y_pred_ridge_gd = None

        # Metrics
        self.mse_ols_analytical = None
        self.r2_ols_analytical = None
        self.train_mse_ols_analytical = None
        self.train_r2_ols_analytical = None

        self.mse_ridge_analytical = None
        self.r2_ridge_analytical = None
        self.train_mse_ridge_analytical = None
        self.train_r2_ridge_analytical = None

        self.mse_ols_gd = None
        self.r2_ols_gd = None
        self.train_mse_ols_gd = None
        self.train_r2_ols_gd = None

        self.mse_ridge_gd = None
        self.r2_ridge_gd = None
        self.train_mse_ridge_gd = None
        self.train_r2_ridge_gd = None

        # Histories (kept)
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

        # Internal maps to cut duplication
        self._METHODS = {
            "ols_analytical": {
                "theta": "theta_ols_analytical",
                "y_pred": "y_pred_ols_analytical",
                "mse": "mse_ols_analytical",
                "r2": "r2_ols_analytical",
                "train_mse": "train_mse_ols_analytical",
                "train_r2": "train_r2_ols_analytical",
            },
            "ridge_analytical": {
                "theta": "theta_ridge_analytical",
                "y_pred": "y_pred_ridge_analytical",
                "mse": "mse_ridge_analytical",
                "r2": "r2_ridge_analytical",
                "train_mse": "train_mse_ridge_analytical",
                "train_r2": "train_r2_ridge_analytical",
            },
            "ols_gd": {
                "theta": "theta_ols_gd",
                "y_pred": "y_pred_ols_gd",
                "mse": "mse_ols_gd",
                "r2": "r2_ols_gd",
                "train_mse": "train_mse_ols_gd",
                "train_r2": "train_r2_ols_gd",
            },
            "ridge_gd": {
                "theta": "theta_ridge_gd",
                "y_pred": "y_pred_ridge_gd",
                "mse": "mse_ridge_gd",
                "r2": "r2_ridge_gd",
                "train_mse": "train_mse_ridge_gd",
                "train_r2": "train_r2_ridge_gd",
            },
        }

    # ---------- fitting ----------
    def fit_analytical(self):
        self.theta_ols_analytical = OLS_parameters(self.X_train, self.y_train)
        if self.lam is not None and self.lam > 0:
            self.theta_ridge_analytical = Ridge_parameters(
                self.X_train, self.y_train, self.lam
            )

    def fit_gradient_descent(self):
        if self.eta is None or self.num_iters is None:
            return
        self.theta_ols_gd, self.ols_mse_history = gd_OLS(
            self.X_train, self.y_train, self.eta, self.num_iters
        )
        if self.lam is not None and self.lam > 0:
            self.theta_ridge_gd, self.ridge_mse_history = gd_Ridge(
                self.X_train, self.y_train, self.eta, self.lam, self.num_iters
            )

    # ---------- predictions & metrics ----------
    def _predict(self, X, theta):
        return X @ theta + self.y_mean

    def predict(self):
        for m, names in self._METHODS.items():
            theta = getattr(self, names["theta"])
            if theta is not None:
                setattr(self, names["y_pred"], self._predict(self.X_test, theta))

    def calculate_metrics(self):
        y_test_true = self.y_test + self.y_mean
        y_train_true = self.y_train + self.y_mean

        for m, names in self._METHODS.items():
            theta = getattr(self, names["theta"])
            if theta is None:
                continue
            # test
            y_pred_test = getattr(self, names["y_pred"])
            if y_pred_test is None:
                y_pred_test = self._predict(self.X_test, theta)
                setattr(self, names["y_pred"], y_pred_test)
            setattr(self, names["mse"], mean_squared_error(y_test_true, y_pred_test))
            setattr(self, names["r2"], r2_score(y_test_true, y_pred_test))
            # train
            y_pred_train = self._predict(self.X_train, theta)
            setattr(
                self, names["train_mse"], mean_squared_error(y_train_true, y_pred_train)
            )
            setattr(self, names["train_r2"], r2_score(y_train_true, y_pred_train))

    def fit_all(self):
        self.fit_analytical()
        self.fit_gradient_descent()
        self.predict()
        self.calculate_metrics()

    # ---------- simple getters ----------
    def _train_metric(self, method, metric):
        if method not in self._METHODS:
            raise ValueError(f"Unknown method: {method}")
        theta_attr = self._METHODS[method]["theta"]
        theta = getattr(self, theta_attr)
        if theta is None:
            raise ValueError(f"{method} solution not fitted")
        y_true = self.y_train + self.y_mean
        y_pred = self._predict(self.X_train, theta)
        if metric == "mse":
            return mean_squared_error(y_true, y_pred)
        if metric == "r2":
            return r2_score(y_true, y_pred)
        raise ValueError(f"Unknown metric: {metric}")

    def get_train_mse(self, method="ols_analytical"):
        return self._train_metric(method, "mse")

    def get_train_r2(self, method="ols_analytical"):
        return self._train_metric(method, "r2")
