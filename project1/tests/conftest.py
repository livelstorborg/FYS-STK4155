import os

import numpy as np


def pytest_configure(config):
    # Make numeric diffs stable and readable
    np.set_printoptions(precision=8, suppress=True)
    # Ensure deterministic behavior where RNG used
    seed = int(os.environ.get("PYTEST_GLOBAL_SEED", 12345))
    np.random.seed(seed)
