import numpy as np
import Kernels
import scipy.optimize._minimize as spmin
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import cho_solve, cholesky
import copy as cp
import warnings
import scipy.optimize as sp_opt
import scipy.spatial.distance as spdist


np.set_printoptions(linewidth=320)
debug_flag = True


class ML:
    """ Parent class for the surface fitting
    :param kernel is the mapping of the points to the feature space, kernel returns the pair distances of given points
                    and derivatives
    """
    def __init__(self, kernel, reg_value, reg_derivative):
        self.x_train = None
        self.x_prime_train = None
        self.y_train = None
        self.y_prime_train = None
        self.n_samples = None
        self.n_samples_prime = None
        self.n_dim = None

        self.kernel = kernel

        self._intercept = None
        self._alpha = None
        self._support_index_alpha = None
        self._beta = None
        self._support_index_beta = None

        self._reg_value = reg_value
        self._reg_derivative = reg_derivative

        self._is_fitted = False

    def _fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        """
        saves the necessary fitting data
        :param x_train: function input pattern  shape = [N_samples, N_features]
        :param y_train: function values to the pattern shape = [N_samples, 1]
        :param x_prime_train: derivative input pattern shape = [N_samples, N_features]
        :param y_prime_train: derivatives values shape = [N_samples, N_features]
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_prime_train = x_prime_train
        self.y_prime_train = y_prime_train

        self.n_samples = len(x_train)
        if y_prime_train is None:
            self.n_samples_prime = 0
            self.n_dim = len(x_train[0])
            self.y_prime_train = np.array([])
        else:
            self.n_samples_prime, self.n_dim = y_prime_train.shape
            self.y_prime_train = self.y_prime_train.flatten('F')

    def predict(self, x):
        """
        function value prediction method for a fitted model
        :param x: prediction pattern shape = [N_samples, N_features]
        :return: predicted function value
        """
        if self._is_fitted:
            if self.n_samples != 0 and self.n_samples_prime != 0:
                return self._alpha[self._support_index_alpha].dot(self.kernel(
                    self.x_train[self._support_index_alpha], x)) + sum(
                    self._beta[self._support_index_beta[ii], ii].dot(self.kernel(
                        self.x_prime_train[self._support_index_beta[ii]], x, dx=ii+1)) for ii in range(self.n_dim)) \
                       + self._intercept

            elif self.n_samples != 0:
                return self._alpha[self._support_index_alpha].dot(
                    self.kernel(self.x_train[self._support_index_alpha], x)) + self._intercept

            else:
                return sum(self._beta[self._support_index_beta[ii], ii].dot(self.kernel(
                    self.x_prime_train[self._support_index_beta[ii]], x, dx=ii+1)) for ii in range(self.n_dim))

        else:
            raise ValueError('not fitted yet')

    def predict_derivative(self, x):
        """
        derivative prediction for a fitted model
        :param x: prediction pattern shape = [N_samples, N_features]
        :return: derivative prediction
        """
        if self._is_fitted:
            ret_mat = np.zeros((len(x), self.n_dim))
            if self.n_samples_prime != 0 and self.n_samples != 0:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = self._alpha[self._support_index_alpha].dot(self.kernel(
                        self.x_train[self._support_index_alpha], x, dy=ii+1)) \
                                     + sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, dy=ii+1, dx=jj+1)) for jj in
                        range(self.n_dim)])

            elif self.n_samples != 0:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = self._alpha[self._support_index_alpha].dot(self.kernel(
                        self.x_train[self._support_index_alpha], x, dy=ii+1))

            else:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, dy=ii+1, dx=jj+1)) for jj in
                        range(self.n_dim)])
            return ret_mat
        else:
            raise ValueError('not fitted yet')

    def predict_val_der(self, x, *args):
        """
        function to allow to use the implemented NEB method
        :param x: prediction pattern, for derivative and function value the same shape = [N_samples, N_features]
        :return: function value prediction, derivative prediction
        """
        x = x.reshape(-1, self.n_dim)
        return self.predict(x), self.predict_derivative(x).reshape(-1), None

    # def _invert_mat(self, mat, cholesky_inversion=False):
    #     if cholesky_inversion:
    #         try:
    #             L = cholesky(mat)
    #         except np.linalg.LinAlgError:
    #             check_matrix(mat)
    #             # implement automatic increasing noise
    #             raise Exception('failed')
    #         return L
    #
    #     else:
    #         # mat = [[A B]      output = [[W X]
    #         #        [C D]]               [Y Z]]
    #         inv_mat = np.zeros(mat.shape)
    #         a = mat[:self.n_samples, :self.n_samples]
    #         b = mat[:self.n_samples, self.n_samples:]
    #         c = mat[self.n_samples:, :self.n_samples]
    #         d_inv = np.linalg.inv(mat[self.n_samples:, self.n_samples:])
    #
    #         w = np.linalg.inv(a - b.dot(d_inv.dot(c)))
    #         x = -w.dot(b.dot(d_inv))
    #
    #         inv_mat[:self.n_samples, :self.n_samples] = w
    #         inv_mat[:self.n_samples, self.n_samples:] = x
    #         inv_mat[self.n_samples:, :self.n_samples] = -d_inv.dot(c.dot(w))
    #         inv_mat[self.n_samples:, self.n_samples:] = d_inv + d_inv.dot(c.dot(w))
    #
    #     return inv_mat


class TSVR(ML):
    def __init__(self, kernel, C1=1, C2=1):
        ML.__init__(kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)



class IRWLS(ML):
    """
    Implementation of the iterative reweighed least square support vector machine for the simultaneous learning of the
    function value and derivatives
    See:
    """

    def __init__(self, kernel, C1=1., C2=1., epsilon=1e-3, epsilon_prime=1e-3, max_iter=1e4):
        self._mat = None
        self.mat_size = None
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        self.max_iter = max_iter

        ML.__init__(self, kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, eps=1e-6):
        if debug_flag:
            self.debug_plotting = (list(), list(), list())
            debug_idx_a = []
            debug_idx_s = []

        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        kernel_mat = create_mat(self.kernel, x_train, x_train, x1_prime=x_prime_train, x2_prime=x_prime_train,
                                dx_max=self.n_dim, dy_max=self.n_dim)
        self.mat_size = self.n_dim * self.n_samples_prime + self.n_samples + 1
        self._mat = np.zeros([self.mat_size, self.mat_size])
        self._mat[:-1, :-1] = kernel_mat
        self._mat[:self.n_samples, -1] = 1.
        self._mat[-1, :self.n_samples] = 1.

        a = np.zeros(self.n_samples)
        a_star = np.zeros(self.n_samples)
        a[0::2] = self._reg_value # even
        a_star[1::2] = self._reg_value # odd

        s = np.zeros(self.n_samples_prime * self.n_dim)
        s_star = np.zeros(self.n_samples_prime*self.n_dim)
        s[0::2] = self._reg_derivative # even
        s_star[1::2] = self._reg_derivative # odd

        weight = np.zeros(self.mat_size)
        val_error, val_error_star, grad_error, grad_error_star = self.get_error(self._mat, weight)

        lagrangian = []
        lagrangian.append(self.get_lagrangian(self._mat, weight, val_error + val_error_star,
                                              grad_error + grad_error_star, flag=debug_flag))

        converged = False
        step = 0
        while not converged:
            a = self.error_weight(self._reg_value, val_error, eps)
            a_star = self.error_weight(self._reg_value, val_error_star, eps)
            s = self.error_weight(self._reg_derivative, grad_error, eps)
            s_star = self.error_weight(self._reg_derivative, grad_error_star, eps)

            idx_a, idx_s, idx_weight = self._get_index(a, a_star, s, s_star)

            if debug_flag:
                debug_idx_a.append(sum(idx_a))
                debug_idx_s.append(sum(idx_s))
            target_vector = self._get_target(idx_a, idx_s, a, a_star, s, s_star)

            mat = self._get_mat(idx_weight)

            mat[:sum(idx_a), :sum(idx_a)] += np.eye(sum(idx_a)) / (a[idx_a] + a_star[idx_a])
            mat[sum(idx_a):-1, sum(idx_a):-1] += np.eye(sum(idx_s)) / (s[idx_s] + s_star[idx_s])

            new_weight = np.zeros(self.mat_size)

            new_weight[idx_weight] = np.linalg.solve(mat, target_vector)

            mu = 1#*1e0
            old_weight = cp.deepcopy(weight)
            weight = old_weight * (1 - mu) + mu*new_weight
            val_error, val_error_star, grad_error, grad_error_star = self.get_error(self._mat, weight) #[error_mat_idx[idx_weight], :]
            lagrangian.append(self.get_lagrangian(self._mat, weight, val_error + val_error_star, grad_error + grad_error_star))
            # lagrangian.append(self.get_lagrangian(self._get_mat(idx_weight), weight[idx_weight], val_error + val_error_star,
            #                                       grad_error + grad_error_star))

            if abs((lagrangian[-1] - lagrangian[-2])/lagrangian[-2]) <= eps:
                converged = True
                self.get_lagrangian(self._mat, weight, val_error + val_error_star,
                                     grad_error + grad_error_star)
                print('Lagrangian converged in step = ' + str(step))

            else:
                pond = 0.5
                conv = False
                iter_step = 0
                while not conv:
                    dummy_weight = weight*pond + old_weight*(1 - pond)
                    val_error, val_error_star, grad_error, grad_error_star = self.get_error(self._mat, dummy_weight)
                    cost = self.get_lagrangian(self._mat, dummy_weight, val_error
                                               + val_error_star, grad_error + grad_error_star)
                    # cost = self.get_lagrangian(self._get_mat(idx_weight), dummy_weight[idx_weight], val_error
                    #                            + val_error_star, grad_error + grad_error_star)

                    if (cost <= lagrangian[-2]*(1 + 1e-7)):
                        conv = True
                    else:
                        pond *= 0.5
                    iter_step += 1
                    if iter_step >= 1e2:
                        conv = True

                if debug_flag:
                    # self.get_lagrangian(self._mat, dummy_weight, val_error + val_error_star, grad_error + grad_error_star,
                    #                 flag=debug_flag)
                    self.get_lagrangian(self._get_mat(idx_weight), dummy_weight[idx_weight], val_error
                                        + val_error_star, grad_error + grad_error_star, flag=debug_flag)
                weight = cp.copy(dummy_weight)
                lagrangian[-1] = cost

            step += 1
            if step > self.max_iter:
                print('not converged')
                converged = True

        self._alpha = weight[:self.n_samples]
        beta = weight[self.n_samples:-1]

        self._support_index_alpha = np.arange(0, self.n_samples)[idx_a]
        beta_idx = idx_s.reshape(self.n_dim, -1).T
        self._support_index_beta = []
        self._beta = beta.reshape(self.n_dim, -1).T
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime)[beta_idx[:, ii]])

        self._intercept = weight[-1]
        self._is_fitted = True
        if debug_flag:
            print(self._alpha[self._support_index_alpha].shape)
            for ii in range(self.n_dim):
                print(self._beta[self._support_index_beta[ii], ii].shape)
            print(sum(idx_a))
            print(sum(idx_s))
            print(debug_idx_a[:10])
            print(debug_idx_s[:10])
            fig = plt.figure()
            fig.add_subplot(221)
            plt.plot(lagrangian)
            plt.title('Lagrangian')
            fig.add_subplot(222)
            plt.plot(self.debug_plotting[0])
            plt.title('weight reg')
            fig.add_subplot(223)
            plt.plot(self.debug_plotting[1])
            plt.title('val error')
            fig.add_subplot(224)
            plt.plot(self.debug_plotting[2])
            plt.title('grad error')

    def error_weight(self, constant, error, epsilon):
        weight = np.zeros_like(error)
        weight[error > epsilon] = 2 * constant * (error[error >= epsilon] - epsilon) / error[error >= epsilon]
        weight[np.logical_and(error < epsilon, error > 0.)] = constant / epsilon
        weight[weight > 1 / epsilon] = 1 / epsilon
        return weight

    def _get_index(self, a, a_star, s, s_star):
        idx_a = np.logical_or(a > 0., a_star > 0.)
        idx_s = np.logical_or(s > 0., s_star > 0.)
        idx_weight = np.concatenate([idx_a, idx_s, np.ones(1)])
        idx_weight = np.ndarray.astype(idx_weight, dtype=bool)
        return idx_a, idx_s, idx_weight

    def _get_target(self, idx_a, idx_s, a, a_star, s, s_star):
        return np.concatenate([self.y_train[idx_a] + (a[idx_a] - a_star[idx_a]) / (a[idx_a] + a_star[idx_a]) * self.epsilon,
                               self.y_prime_train[idx_s] + (s[idx_s] - s_star[idx_s]) / (s[idx_s] + s_star[idx_s])
                               * self.epsilon_prime, np.zeros([1])])

    def _get_mat(self, idx_weight):
        idx_mat = np.logical_and(np.tile(idx_weight, self.mat_size).reshape(self.mat_size, self.mat_size),
                                 np.tile(idx_weight, self.mat_size).reshape(self.mat_size, self.mat_size).T)
        mat = cp.copy(self._mat)
        return mat[idx_mat].reshape(np.sum(idx_weight, dtype=int), np.sum(idx_weight, dtype=int))

    def get_error(self, mat, weight):
        prediction = mat.T.dot(weight)

        val_err = prediction[:self.n_samples]
        val_error = val_err - self.y_train - self.epsilon
        val_error_star = -val_err + self.y_train - self.epsilon
        val_error[val_error < 0.] = 0.
        val_error_star[val_error_star < 0.] = 0

        grad_err = prediction[self.n_samples:-1]
        grad_error = grad_err - self.y_prime_train.flatten('F') - self.epsilon_prime
        grad_error_star = -grad_err + self.y_prime_train.flatten('F') - self.epsilon_prime
        grad_error[grad_error < 0.] = 0.
        grad_error_star[grad_error_star < 0.] = 0.

        return val_error, val_error_star, grad_error, grad_error_star

    def get_lagrangian(self, mat,  weight_vector, val_error, grad_error, flag=False):
        weight_regularization = weight_vector.T.dot(mat.dot(weight_vector))/2.
        val_error = self._reg_value * sum(self._error_func(val_error, self.epsilon))
        grad_error = self._reg_derivative * sum(self._error_func(grad_error, self.epsilon_prime))

        if flag:
            self.debug_plotting[0].append(weight_regularization)
            self.debug_plotting[1].append(val_error)
            self.debug_plotting[2].append(grad_error)
        return weight_regularization + val_error + grad_error

    def _error_func(self, error, epsilon, approximate_function='L2'):
        l_value = np.zeros_like(error)
        idx = error >= epsilon
        if approximate_function == 'L2':
            l_value[idx] = error[idx] ** 2 - 2*error[idx]*epsilon + epsilon**2
        elif approximate_function =='L1':
            l_value = error[idx]
        return l_value


class RLS(ML):
    """
    Implementation of the regularized least square method for simultaneous fitting of function value and derivatives
    See:
    """
    def __init__(self, kernel, C1=1., C2=1., min_intercept=False):
        self.min_intercept = min_intercept
        ML.__init__(self, kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        """
        Fitting a new model to a given training pattern
        :param x_train: function input pattern shape = [N_samples, N_features]
        :param y_train: function values shape = [N_samples, 1]
        :param x_prime_train: derivative input pattern shape = [N_samples, N_features]
        :param y_prime_train: derivative values shape = [N_samples, N_features]
        :param C1: penalty parameter to the function error
        :param C2: penalty parameter to the derivative error
        :param minimze_b: changes the behaviour to minimize the weights and the bias, default False minimizee only weights
        :return:
        """

        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        if self.n_samples_prime == 0:
            kernel_mat = create_mat(self.kernel, x_train, x_train, x_prime_train, x_prime_train, eval_gradient=False)
            kernel_mat[:self.n_samples, :self.n_samples] = kernel_mat[:self.n_samples, :self.n_samples]\
                                                           + np.eye(self.n_samples) / self._reg_value
        else:
            kernel_mat = create_mat(self.kernel, x_train, x_train, x_prime_train, x_prime_train, dx_max=self.n_dim,
                                                                                                    dy_max=self.n_dim)
            kernel_mat[:self.n_samples, :self.n_samples] = kernel_mat[:self.n_samples, :self.n_samples] \
                                                           + np.eye(self.n_samples)/self._reg_value
            kernel_mat[self.n_samples:, self.n_samples:] = kernel_mat[self.n_samples:, self.n_samples:] \
                                                           + np.eye(self.n_samples_prime*self.n_dim) / self._reg_derivative

        if not self.min_intercept:
            mat_size = self.n_samples + self.n_samples_prime * self.n_dim + 1
            mat = np.zeros([mat_size, mat_size])
            mat[:-1, :-1] = kernel_mat

            if self.n_samples == 0:
                mat[-1, -1] = 1
            else:
                mat[:self.n_samples, -1] = 1
                mat[-1, :self.n_samples] = 1

            vec = np.concatenate([self.y_train, self.y_prime_train.flatten('F'), np.zeros([1])])

        else:
            vec = np.concatenate([self.y_train, self.y_prime_train.flatten('F')])
            mat = kernel_mat

        weight = np.linalg.solve(mat, vec)

        self._alpha = weight[0:self.n_samples].reshape(-1).T
        if not self.min_intercept:
            self._intercept = weight[-1]
            self._beta = weight[self.n_samples:-1].reshape(self.n_dim, -1).T
        else:
            self._intercept = sum(self._alpha)
            self._beta = weight[self.n_samples:].reshape(self.n_dim, -1).T

        self._support_index_alpha = np.arange(0, self.n_samples, 1)
        self._support_index_beta = []
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

        self._is_fitted = True


class GPR(ML):
    def __init__(self, kernel, noise_value=1e-10, noise_derivative=1e-10, opt_method='LBFGS_B', opt_parameter=True,
                 opt_restarts=0, normalize_y=False):
        # RBF kernel only supported and a constant mean over all training points
        # cholesky matrix
        # noise is added to the main diagonal of the kernel matrix
        # this is done in the log marginal likelihood
        self.L = None
        self.y_mean = None

        self._target_vector = None

        self._opt_flag = True
        self.opt_method = opt_method

        self.opt_parameter = opt_parameter
        self.opt_restarts = opt_restarts
        self._normalize_y = normalize_y

        ML.__init__(self, kernel, reg_value=noise_value, reg_derivative=noise_derivative)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)

        if self._normalize_y:
            self.y_mean = np.mean(y_train)
        else:
            self.y_mean = 0.
        if self.x_prime_train is None:
            self._target_vector = np.concatenate([self.y_train - self.y_mean])
        else:
            # self.y_prime_mean = np.mean(y_prime_train, axis=0)
            self._target_vector = np.concatenate([self.y_train - self.y_mean, self.y_prime_train.flatten('F')])

        if self.opt_restarts > 0:
            self.opt_parameter = True

        if self.opt_parameter:
            initial_hyper_parameters = self.get_hyper_parameter()
            opt_hyper_parameter = []
            value = []
            opt = self._opt_routine(initial_hyper_parameters)

            opt_hyper_parameter.append(opt[0])
            value.append(opt[1])

            for ii in range(self.opt_restarts):
                initial_hyper_parameters = []
                bounds = self.kernel.bounds
                for element in bounds:
                    initial_hyper_parameters.append(np.random.uniform(element[0], element[1], 1))
                initial_hyper_parameters = np.array(initial_hyper_parameters)
                opt = self._opt_routine(initial_hyper_parameters)
                opt_hyper_parameter.append(opt[0])
                value.append(opt[1])

            min_idx = np.argmin(value)
            self.set_hyper_parameter(opt_hyper_parameter[min_idx])

        if self.n_samples_prime == 0:
            K = create_mat(self.kernel, self.x_train, self.x_train)
            K += np.eye(K.shape[0]) * self._reg_value
            self.L, alpha = self._cholesky(K)
            self._alpha = alpha
        else:
            k_mat = create_mat(self.kernel, self.x_train, self.x_train, x1_prime=self.x_prime_train,
                               x2_prime=self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim)
            k_mat[:self.n_samples, :self.n_samples] += self._reg_value * np.eye(self.n_samples)
            k_mat[self.n_samples:, self.n_samples:] += np.eye(self.n_samples_prime * self.n_dim) * self._reg_derivative

            self.L, alpha = self._cholesky(k_mat)
            self._alpha = alpha[:self.n_samples]
            self._beta = alpha[self.n_samples:].reshape(self.n_dim, -1).T

        self._is_fitted = True
        self._intercept = self.y_mean

    def get_hyper_parameter(self):
        return self.kernel.theta

    def set_hyper_parameter(self, hyper_parameter):
        self.kernel.theta = hyper_parameter

    def get_bounds(self):
        return self.kernel.bounds
        #

    def optimize(self, hyper_parameter):
        self.set_hyper_parameter(hyper_parameter)
        # print(hyper_parameter)
        log_marginal_likelihood, d_log_marginal_likelihood = self.log_marginal_likelihood(derivative=self._opt_flag)

        return -log_marginal_likelihood, -d_log_marginal_likelihood

    def log_marginal_likelihood(self, derivative=False):
        # gives vale of log marginal likelihood with the gradient
        if self.n_samples_prime == 0:
            k_mat, k_grad = create_mat(self.kernel, self.x_train, self.x_train, eval_gradient=True)
            k_mat += np.eye(k_mat.shape[0]) * self._reg_value
        else:
            k_mat, k_grad = create_mat(self.kernel, self.x_train, self.x_train, self.x_prime_train,
                                        self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim, eval_gradient=True)
            k_mat[:self.n_samples, :self.n_samples] += np.eye(self.n_samples) * self._reg_value
            k_mat[self.n_samples:, self.n_samples:] += np.eye(self.n_samples_prime*self.n_dim)*self._reg_derivative
        L, alpha = self._cholesky(k_mat)
        log_mag_likelihood = -0.5*self._target_vector.dot(alpha) - np.log(np.diag(L)).sum() - L.shape[0] / 2. * np.log(2 * np.pi)

        if not derivative:
            return log_mag_likelihood

        temp = (np.multiply.outer(alpha, alpha) - cho_solve((L, True), np.eye(L.shape[0])))[:, :, np.newaxis]
        d_log_mag_likelihood = 0.5 * np.einsum("ijl,ijk->kl", temp, k_grad)
        d_log_mag_likelihood = d_log_mag_likelihood.sum(-1)

        return log_mag_likelihood, d_log_mag_likelihood

    def _cholesky(self, kernel):
        try:
            L = cholesky(kernel, lower=True)
        except np.linalg.LinAlgError:
            check_matrix(kernel)
            print(self.get_hyper_parameter())
            # implement automatic increasing noise
            raise Exception('failed')
        alpha = cho_solve((L, True), self._target_vector)
        return L, alpha

    def _opt_routine(self, initial_hyper_parameter):
        if self.opt_method == 'LBFGS_B':
            self._opt_flag = True
            opt_hyper_parameter, value, opt_dict = sp_opt.fmin_l_bfgs_b(self.optimize, initial_hyper_parameter,
                                                                    bounds=self.get_bounds())
            if opt_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: %s" % opt_dict)
        elif self.opt_method == 'BFGS':
            raise NotImplementedError('Implementation is not finished.')
            # TODO return function value
            # TODO implementation
            self._opt_flag = False
            opt_hyper_parameter = sp_opt.fmin_bfgs(self.optimize, initial_hyper_parameter)
            value = 0
        else:
            raise NotImplementedError('Method is not implemented use method=LBFGS_B.')

        return opt_hyper_parameter, value

    def predict(self, x, error_estimate=False):
        # print(self.get_hyper_parameter())
        if self.n_samples_prime == 0:
            predictive_mean = (self.kernel(x, self.x_train).dot(self._alpha)) + self._intercept

            if error_estimate:
                v = cho_solve((self.L, True), self.kernel(self.x_train, x))

                mat = self.kernel(x, x)
                predictive_covariance = mat - self.kernel(self.x_train, x).T.dot(v)
                predictive_covariance[predictive_covariance[np.diag_indices(len(predictive_covariance))] < 0.0] = 0.0
                predictive_variance = np.diag(predictive_covariance)
                return predictive_mean, predictive_variance
            else:
                return predictive_mean

        else:
            predictive_mean = self._alpha.dot(self.kernel(self.x_train, x)) + self._intercept \
                       + sum(self._beta[:, ii].dot(self.kernel(self.x_prime_train, x, dx=ii+1)) for ii in range(self.n_dim))
            if error_estimate:
                n, d = x.shape
                k_mat = create_mat(self.kernel, self.x_train, x, x1_prime=self.x_prime_train, x2_prime=x,
                                   dx_max=self.n_dim, dy_max=self.n_dim)
                mat = create_mat(self.kernel, x, x, x1_prime=x, x2_prime=x, dx_max=self.n_dim, dy_max=self.n_dim)
                v = cho_solve((self.L, True), k_mat)
                predictive_covariance = mat - k_mat.T.dot(v)
                predictive_covariance[predictive_covariance[np.diag_indices(len(predictive_covariance))] < 0.0] = 0.0
                predictive_variance = np.diag(predictive_covariance)[:n]
                return predictive_mean, predictive_variance
            else:
                return predictive_mean

    def predict_derivative(self, x, error_estimate=False):
        predictive_derivative = np.zeros((len(x), self.n_dim))
        if self.n_samples_prime == 0:
            for ii in range(self.n_dim):
                predictive_derivative[:, ii] = self._alpha.dot(self.kernel(self.x_train, x, dy=ii + 1))
            return predictive_derivative
        else:
            for ii in range(self.n_dim):
                predictive_derivative[:, ii] = self._alpha.dot(self.kernel(self.x_train, x, dy=ii + 1)) \
                                     + sum([self._beta[:, jj].dot(self.kernel(self.x_prime_train, x, dy=ii + 1, dx=jj + 1))
                                            for jj in range(self.n_dim)])
            return predictive_derivative


def check_matrix(mat):
    print('symmertric = ' + str(np.allclose(mat, mat.T, 10**-32)))
    if not np.allclose(mat, mat.T, 10**-32):
        raise ValueError('not symmetric')
    print('derterminante = ' + str(np.linalg.det(mat)))
    eig_val, eig_vec = np.linalg.eigh(mat)
    print('eigenvalues = ' + str(eig_val))
    print('dimension = ' + str(mat.shape))
    print('----------------')


def create_mat(kernel, x1, x2, x1_prime=None, x2_prime=None, dx_max=0, dy_max=0, eval_gradient=False):
    # creates the kernel matrix
    # if x1_prime is None then no derivative elements are calculated.
    # derivative elements are given in the manner of [dx1, dx2, dx3, ...., dxn]
    n, d = x1.shape
    if x1_prime is None:
        return kernel(x1, x2, dx=0, dy=0, eval_gradient=eval_gradient)
    else:
        m, f = x2.shape
        if not eval_gradient:
            kernel_mat = np.zeros([n * (1 + d), m * (1 + f)])
            for jj in range(dx_max + 1):
                for ii in range(dy_max + 1):
                    if ii == 0 and jj == 0:
                        kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(x1, x2, dx=ii, dy=jj,
                                                                                      eval_gradient=eval_gradient)
                    elif (ii == 0 and jj != 0):
                        kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(x1, x2_prime, dx=ii, dy=jj,
                                                                                      eval_gradient=eval_gradient)
                    elif (jj == 0 and ii != 0):
                        kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(x1_prime, x2, dx=ii, dy=jj,
                                                                                      eval_gradient=eval_gradient)
                    else:
                        kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(x1_prime, x2_prime, dx=ii, dy=jj,
                                                                                      eval_gradient=eval_gradient)
            return kernel_mat
        else:
            num_theta = len(kernel.theta)
            kernel_derivative = np.zeros([n * (1 + d), m * (1 + f), num_theta])
            kernel_mat = np.zeros([n * (1 + d), m * (1 + f)])
            for ii in range(dx_max + 1):
                for jj in range(dy_max + 1):
                    if ii == 0 and jj == 0:
                       k_mat, deriv_mat = kernel(x1, x2, dx=ii, dy=jj, eval_gradient=eval_gradient)
                    elif (ii == 0 and jj != 0):
                        k_mat, deriv_mat = kernel(x1, x2_prime, dx=ii, dy=jj, eval_gradient=eval_gradient)
                    elif (jj == 0 and ii != 0):
                        k_mat, deriv_mat = kernel(x1_prime, x2, dx=ii, dy=jj, eval_gradient=eval_gradient)
                    else:
                        k_mat, deriv_mat = kernel(x1_prime, x2_prime, dx=ii, dy=jj, eval_gradient=eval_gradient)

                    kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = k_mat
                    kernel_derivative[n * ii:n * (ii + 1), m * jj:m * (jj + 1), :] = deriv_mat
            return kernel_mat, kernel_derivative
