import numpy as _np
import scipy.spatial.distance as _spdist


class RBF:
    def __init__(self, gamma=0.1, amplitude=1.):
        self.gamma = gamma
        self.amplitude = amplitude

    def __call__(self, x, y, dx=0, dy=0, dp=0):
        # dp derivative of the parameters is used for the GPR
        # in case of GPR the gamma have to be redefined outside to gamma = 1/(2*exp(length scale)) because length scale
        # is the hyper parameter of interest
        # mat = _np.tile(_np.sum(x ** 2, axis=1), (len(y), 1)).T + _np.tile(_np.sum(y ** 2, axis=1),
        #                                                                   (len(x), 1)) - 2 * x.dot(y.T)
        mat = _spdist.cdist(x ,y, 'sqeuclidean')
        exp_mat = self.amplitude * _np.exp(-self.gamma*mat)
        if dp == 0:
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

        elif dp == 1:
            # derivative of the length scale --> gamma = 1/(2*exp(length scale))
            return exp_mat*mat

        elif dp == 2:
            # derivative of the amplitude (variance)
            return exp_mat*2.

