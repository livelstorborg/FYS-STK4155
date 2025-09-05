import sys

import numpy as np

from project1.regression import polynomial_features

print(sys.executable)

x = np.arange(10)
p = 3
feats = polynomial_features(x, p)
assert feats is not None
