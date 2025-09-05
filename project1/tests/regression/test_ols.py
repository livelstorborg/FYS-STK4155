import numpy as np

from project1.regression import ols


def make_linear_data(n=50, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), np.linspace(-1, 1, n)]
    beta = np.array([2.0, -3.0])
    y = X @ beta + noise * rng.normal(size=n)
    return X, y, beta


def test_ols_parameters_closed_form_recovery_no_noise():
    X, y, beta_true = make_linear_data(n=10, noise=0.0)
    beta_hat = ols.OLS_parameters(X, y)
    assert beta_hat.shape == beta_true.shape
    assert np.allclose(beta_hat, beta_true)


def test_ols_parameters_singular_matrix_raises():
    # Singular X^T X when X has duplicate columns
    X = np.c_[np.ones(5), np.ones(5)]
    y = np.arange(5.0)
    try:
        _ = ols.OLS_parameters(X, y)
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    assert raised, "Expected LinAlgError for singular matrix"


def test_gradient_descent_ols_converges_small_noise():
    X, y, beta_true = make_linear_data(n=50, noise=0.01, seed=42)
    beta_gd = ols.gradient_descent_ols(X, y, eta=0.1, n_iter=2000)
    # Should be close to closed-form solution
    beta_cf = ols.OLS_parameters(X, y)
    assert np.allclose(beta_gd, beta_cf, atol=1e-2)


def test_gradient_descent_ols_zero_iterations_returns_zeros():
    X, y, _ = make_linear_data(n=5, noise=0.0)
    beta = ols.gradient_descent_ols(X, y, eta=0.1, n_iter=0)
    assert np.allclose(beta, np.zeros(X.shape[1]))


def test_gradient_descent_ols_shapes_and_types():
    X, y, _ = make_linear_data(n=7)
    beta = ols.gradient_descent_ols(X, y, eta=0.05, n_iter=10)
    assert beta.shape == (X.shape[1],)
