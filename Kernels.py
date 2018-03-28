import numpy as _np

class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y, dx=0, dy=0):
        exp_mat = _np.exp(-self.gamma * (_np.tile(_np.sum(x ** 2, axis=1), (len(y), 1)).T +
                                         _np.tile(_np.sum(y ** 2, axis=1), (len(x), 1)) - 2 * x.dot(y.T)))
        if dx == dy:
            if dx == 0:
                return exp_mat
            else:
                return -2.0 * self.gamma * exp_mat * (2.0 * self.gamma * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1]) ** 2 - 1)
        elif dx == 0:
            return -2.0 * self.gamma * exp_mat * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1])
        elif dy == 0:
            return 2.0 * self.gamma * exp_mat * _np.subtract.outer(x[:, dx - 1].T, y[:, dx - 1])
        else:
            return -4.0 * self.gamma**2 * exp_mat * _np.subtract.outer(x[:, dx - 1].T, y[:, dx - 1]) \
                   * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1])
