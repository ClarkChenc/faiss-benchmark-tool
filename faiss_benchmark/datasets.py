import numpy as np

def create_random_dataset(nb: int, d: int):
    """Creates a random dataset of a given size and dimension."""
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nb // 10, d)).astype('float32')
    return xb, xq