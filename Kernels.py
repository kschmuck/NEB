import numpy as _np
import scipy.spatial.distance as _spdist


class RBF:
    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def __call__(self, x, y, dx=0, dy=0, dp=0):
        # dp derivative of the parameters is used for the GPR
        # in case of GPR the gamma have to be redefined outside to gamma = 1/(2*exp(length scale)) because length scale
        # is the hyper parameter of interest
        # mat = _np.tile(_np.sum(x ** 2, axis=1), (len(y), 1)).T + _np.tile(_np.sum(y ** 2, axis=1),
        #                                                                   (len(x), 1)) - 2 * x.dot(y.T)
        mat = _spdist.cdist(x, y, 'sqeuclidean')
        exp_mat = _np.exp(-self.gamma * mat)
        if dp == 0:
            if dx == dy:
                if dx == 0:
                    return exp_mat
                else:
                    return -2.0 * self.gamma * exp_mat * (
                        2.0 * self.gamma * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1]) ** 2 - 1)
            elif dx == 0:
                return -2.0 * self.gamma * exp_mat * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1])
            elif dy == 0:
                return 2.0 * self.gamma * exp_mat * _np.subtract.outer(x[:, dx - 1].T, y[:, dx - 1])
            else:
                return -4.0 * self.gamma ** 2 * exp_mat * _np.subtract.outer(x[:, dx - 1].T, y[:, dx - 1]) \
                       * _np.subtract.outer(x[:, dy - 1].T, y[:, dy - 1])

        elif dp == 1:
            # derivative of the length scale --> gamma = 1/(2*exp(length scale))
            return exp_mat * mat

            # elif dp == 2:
            #     # derivative of the amplitude (variance)
            #     return exp_mat*2.


class newRBFGrad:
    def __init__(self, signal_variance=0.,length_scale=0., bias=0.,
                 bounds=[(10 ** -5, 10**5.), (10 ** -5, 10**5),  (10 ** -5, 10**5)]):
        # standard implementation isotropic space
        self.hyper_parameter = None
        self.set_hyper_parameters(_np.array([signal_variance, length_scale, bias])) # length_scale_grad
        # self.hyper_parameter = [signal_variance, length_scale, length_scale_grad, bias]
        self.bounds = bounds

    def add_hyper_parameter(self, hyper_parameter):
        # used for adding the gradient length scale
        self.hyper_parameter = _np.concatenate([self.hyper_parameter, hyper_parameter])

    def set_hyper_parameters(self, hyper_parameters):
        self.hyper_parameter = []
        for element in hyper_parameters:
            self.hyper_parameter.append(_np.exp(element))
        self.hyper_parameter = _np.array(self.hyper_parameter)

    def get_hyper_parameters(self):
        return_array = []
        for element in self.hyper_parameter:
            return_array.append(_np.log(element))
        return _np.array(return_array)

    def __call__(self, x, y, dx=0, dy=0, dp=0):
        if dp > len(self.hyper_parameter):
            raise ValueError(
                'There are only ' + str(len(self.hyper_parameter)) + ' hyper parameters')

        if len(self.hyper_parameter) == 3:
            signal_variance =(self.hyper_parameter[0]) # amplitude
            length_scale = (self.hyper_parameter[1])
            bias = 0#self.hyper_parameter[2]
            distance = _spdist.cdist(x/length_scale, y/length_scale, metric='sqeuclidean')

            exp_mat_func = _np.exp(-0.5*distance)

            if dp == 0:
                if dx == dy:
                    if dx == 0:
                        return signal_variance * exp_mat_func + bias
                    else:
                        return signal_variance / length_scale ** 2 * exp_mat_func \
                                    * (1 - _np.subtract.outer(x[:, dx - 1], y[:, dx - 1])**2 / length_scale**2)
                elif dx == 0:
                    return -signal_variance * exp_mat_func * _np.subtract.outer(x[:, dy-1], y[:, dy-1])/length_scale**2
                elif dy == 0: # dx == i
                    return signal_variance * exp_mat_func * _np.subtract.outer(x[:, dx-1], y[:, dx-1])/length_scale**2
                else:
                    return -signal_variance / length_scale**4 * _np.subtract.outer(x[:, dx-1], y[:, dx-1])\
                           * _np.subtract.outer(x[:, dy-1], y[:, dy-1]) * exp_mat_func

            elif dp == 1:  # derivative of the signal variance --> derivative is same  since signal var = exp(parameter)
                return self(x, y, dx=dx, dy=dy, dp=0)
                # if dx == dy:
                #     if dx == 0:
                #         return signal_variance*_np.exp(-0.5 * distance)
                #     else:
                #         return self(x, y, dx=dx, dy=dy, dp=0)
                # else:
                #     return self(x, y, dx=dx, dy=dy, dp=0)

            elif dp == 2:
                if dx == dy:
                    if dx == 0:
                        return signal_variance*_np.exp(-0.5*distance)*distance
                    else:
                        return signal_variance / length_scale ** 2 * exp_mat_func * (2 * _np.subtract.outer(x[:, dx - 1],
                                          y[:, dx - 1])**2/length_scale**2 - 1 + (1 - _np.subtract.outer(x[:, dx - 1],
                                                                       y[:, dx - 1])**2 / length_scale**2) * distance)
                elif dy == 0:
                    # dx == i
                    return self(x, y, dx=dx, dy=0, dp=0) * (distance - 1)
                elif dx == 0:
                    return self(x, y, dx=dx, dy=0, dp=0) * (distance - 1)
                else:
                    return self(x, y, dx=dx, dy=dy, dp=0) * (2 - distance)

            elif dp == 3:
                if dx == 0 and dy == 0:
                    return bias*_np.ones_like(distance)
                else:
                    return _np.zeros_like(distance)
                # if dx == dy:
                #     if dx == 0:
                #         return bias*_np.ones_like(distance)
                #     else:
                #         return _np.zeros_like(distance)
                # else:
                #     return _np.zeros_like(distance)

            else:
                raise ValueError('no more parameters')

