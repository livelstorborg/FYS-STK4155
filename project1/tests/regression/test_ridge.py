import numpy as np

from project1.regression import ridge


def make_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    X = np.c_[np.ones(n), x, x**2]
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + 0.05 * rng.normal(size=n)
    return X, y, beta


def test_ridge_parameters_matches_closed_form_lambda_zero_reduces_to_ols():
    X, y, _ = make_data(n=20, seed=1)
    beta_ridge = ridge.ridge_parameters(X, y, lambda_=0.0)
    # Compare against OLS formula (no reg)
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    assert np.allclose(beta_ridge, beta_ols)


def test_ridge_parameters_regularization_shrinks_coeffs():
    X, y, _ = make_data(n=50, seed=2)
    beta_l0 = ridge.ridge_parameters(X, y, lambda_=0.0)
    beta_l10 = ridge.ridge_parameters(X, y, lambda_=10.0)
    # Larger lambda should shrink coefficients towards zero
    assert np.linalg.norm(beta_l10) < np.linalg.norm(beta_l0)


def test_ridge_parameters_shapes():
    X, y, _ = make_data(n=7)
    beta = ridge.ridge_parameters(X, y, lambda_=1.0)
    assert beta.shape == (X.shape[1],)


def test_gradient_descent_ridge_converges_to_closed_form_small_eta():
    X, y, _ = make_data(n=40, seed=3)
    lam = 0.5
    beta_cf = ridge.ridge_parameters(X, y, lambda_=lam)
    # The GD implementation optimizes (1/n)||y-Xb||^2 + lam'||b||^2.
    # To match closed-form which uses ||y-Xb||^2 + lam||b||^2, set lam' = lam/n.
    lam_prime = lam / X.shape[0]
    beta_gd = ridge.gradient_descent_ridge(X, y, lam=lam_prime, eta=0.05, n_iter=4000)
    assert np.allclose(beta_gd, beta_cf, atol=1e-2)


def test_gradient_descent_ridge_zero_iter_returns_zeros():
    X, y, _ = make_data(n=5)
    beta = ridge.gradient_descent_ridge(X, y, lam=1.0, eta=0.1, n_iter=0)
    assert np.allclose(beta, np.zeros(X.shape[1]))
