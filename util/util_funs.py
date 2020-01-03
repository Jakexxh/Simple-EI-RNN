import numpy as np


def rectify(x):
    return x*(x > 0)


def spectral_radius(x):
    return np.max(abs(np.linalg.eigvals(x)))
