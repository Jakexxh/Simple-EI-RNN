import numpy as np


def rectify(x):
    return x*(x > 0)


def spectral_radius(x):
    return np.max(abs(np.linalg.eigvals(x)))

# k=np.array([1,1,1,1,1,1,1,1,-1,-1])
#
# exc, = np.where(k > 0)
# inh, = np.where(k < 0)
#
# C = np.zeros((10, 10))
# for i in exc:
#     C[i, exc] = 1
#     C[i, i] = 0
#     C[i, inh] = 1
#     C[i, inh] *= np.sum(C[i, exc]) / np.sum(C[i, inh])
# for i in inh:
#     C[i, exc] = 1
#     C[i, inh] = 1
#     C[i, i] = 0
#     C[i, inh] *= np.sum(C[i, exc]) / np.sum(C[i, inh])