import numpy as np

from project1.regression import features as feats


def test_polynomial_features_no_intercept_basic():
    x = np.array([0.0, 1.0, 2.0])
    X = feats.polynomial_features(x, p=3, intercept=False)
    # Columns: x^1, x^2, x^3
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 4.0, 8.0],
        ]
    )
    assert X.shape == (3, 3)
    assert np.allclose(X, expected)


def test_polynomial_features_with_intercept():
    x = np.array([1.0, 2.0])
    X = feats.polynomial_features(x, p=2, intercept=True)
    # Columns: 1, x^1, x^2
    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0],
        ]
    )
    assert X.shape == (2, 3)
    assert np.allclose(X, expected)


def test_polynomial_features_degree_zero():
    x = np.array([3.0, 4.0])
    X_no_intercept = feats.polynomial_features(x, p=0, intercept=False)
    assert X_no_intercept.shape == (2, 0)

    X_with_intercept = feats.polynomial_features(x, p=0, intercept=True)
    assert np.allclose(X_with_intercept, np.ones((2, 1)))


def test_polynomial_features_empty_input():
    x = np.array([])
    X = feats.polynomial_features(x, p=3, intercept=True)
    assert X.shape == (0, 4)
