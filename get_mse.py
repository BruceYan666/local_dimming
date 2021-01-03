def get_mse(X, Y):
    import numpy as np
    h = X.shape[0]
    w = X.shape[1]
    tmp = (X - Y) ** 2
    return np.sum(tmp) / (h * w)


