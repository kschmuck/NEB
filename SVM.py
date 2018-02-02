import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import scipy.optimize._minimize as spmin


def euclidean_distance(x, y):
    xx = np.sum(x**2, axis=1).reshape(-1, 1)
    yy= np.sum(yy, axis=1).reshape(1, -1)
    xy = 2.0*np.dot(x, y.T)
    squared_distance = xx + yy - xy
    return squared_distance


class SVM:
    def __init__(self, epsilon=0.1, kernel='rbf'):
        self.epsilon = epsilon
        self.b = None
        self.alpha = None

        self.target = None
        self.input_value = None

        if kernel == 'rbf':
            self.kernel = RBF()

    def fit_scipy(self, input_value, target_value, max_steps=100, derivative_target=None, C=1.0):
        self.target = target_value
        self.input_value = input_value
        kernel = self.kernel.kernel(input_value)
        n_samples = len(target_value)
        # inequalities have to be func >= 0
        constrains = ({'type': 'eq', 'fun': lambda x: np.array(np.sum(x[::2]-x[1::2]))},
                      {'type': 'ineq', 'fun': lambda x: np.array(x)},
                      {'type': 'ineq', 'fun': lambda x: np.array(-x + C)})
        # 'jac': lambda x: np.array(np.sum(1.0-x[1::2]), np.sum(x[::2]-1.0)),
        # 'jac': lambda x: np.ones(n_samples*2)},
        # 'jac': lambda x: -np.ones(n_samples*2)})

        def dual_func(x):
            # x = [every even is alpha, and every odd is alpha star]
            term_a = -0.5 * np.dot(np.dot(np.transpose(x[::2] - x[1::2]), kernel), x[::2] - x[1::2])
            term_b = self.epsilon * np.sum(x[::2] + x[1::2]) + np.dot(self.target.T, (x[::2] - x[1::2]))
            return -(term_a + term_b)

        res = spmin.minimize(dual_func, np.zeros(n_samples*2), method='SLSQP', constraints=constrains)

        self.alpha = res.x[::2]-res.x[1::2]
        # TODO calculating b --> is not relevant for the NEB run
        self.b = 0.0

    def predict_scipy(self, predict_value):
        kernel = self.kernel.kernel(self.input_value, predict_value)
        prediction = np.dot(self.alpha, kernel) + self.b
        return prediction


class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = -gamma

    def kernel(self, x, y=None):

        if y is None:
            y = x
        squared_distance = euclidean_distance(x, y)
        kernel = np.exp(self.gamma*squared_distance)
        return kernel


