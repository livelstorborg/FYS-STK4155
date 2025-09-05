import numpy as np

from project1.regression import functions as fns


def test_runge_scalar_values():
    assert fns.runge(0.0) == 1.0
    assert np.isclose(fns.runge(1.0), 1 / 26)


def test_runge_vectorized_and_symmetry():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = fns.runge(x)
    assert y.shape == x.shape
    # Even function: f(x) == f(-x)
    assert np.allclose(y[0], y[-1])
    assert np.allclose(y[1], y[-2])


def test_runge_large_values_underflow_to_zero():
    # For very large |x|, value tends to 0
    x = 1e9
    assert np.isclose(fns.runge(x), 0.0)
