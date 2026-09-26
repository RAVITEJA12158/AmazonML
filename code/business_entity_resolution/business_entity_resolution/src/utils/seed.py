import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set every source of randomness we touch. Call this once at the start
    of every script, before anything else runs."""
    random.seed(seed)
    np.random.seed(seed)
