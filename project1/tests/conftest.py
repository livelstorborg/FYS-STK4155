import os
import sys

import numpy as np


def pytest_configure(config):
    # Make numeric diffs stable and readable
    np.set_printoptions(precision=8, suppress=True)
    # Ensure deterministic behavior where RNG used
    seed = int(os.environ.get("PYTEST_GLOBAL_SEED", 12345))
    np.random.seed(seed)
    # Ensure src/ is on sys.path for imports like 'project1.*'
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
