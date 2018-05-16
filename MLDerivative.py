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

# Todo IRWLS or Support vector machine algorithm

class ML:
    """ Parent class for the surface fitting
    :param kernel is the mapping of the points to the feature space, kernel returns the pair distances of given points
                    and derivatives
    """
    def __init__(self, kernel):
        self.x_train = None
        self.x_prime_train = None
        self.y_train = None
        self.y_prime_train = None
        self.n_samples = None
        self.n_samples_prime = None
        self.n_dim = None

        self.y_debug = None

        self._is_fitted = False

        self.kernel = kernel

        self._intercept = None
        self._alpha = None
        self._support_index_alpha = None
        self._beta = None
        self._support_index_beta = None

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

    def _invert_mat(self, mat):
        # mat = [[A B]      output = [[W X]
        #        [C D]]               [Y Z]]
        inv_mat = np.zeros(mat.shape)
        a = mat[:self.n_samples, :self.n_samples]
        b = mat[:self.n_samples, self.n_samples:]
        c = mat[self.n_samples:, :self.n_samples]
        d_inv = np.linalg.inv(mat[self.n_samples:, self.n_samples:])

        w = np.linalg.inv(a - b.dot(d_inv.dot(c)))
        x = -w.dot(b.dot(d_inv))

        inv_mat[:self.n_samples, :self.n_samples] = w
        inv_mat[:self.n_samples, self.n_samples:] = x
        inv_mat[self.n_samples:, :self.n_samples] = -d_inv.dot(c.dot(w))
        inv_mat[self.n_samples:, self.n_samples:] = d_inv + d_inv.dot(c.dot(w))

        return inv_mat


class IRWLS(ML):
    # Todo implementation!!!!!!!
    """
    Implementation of the iterative reweighed least square support vector machine for the simultaneous learning of the
    function value and derivatives
    See:
    """

    def __init__(self, kernel):
        self._y = None
        self._mat = None
        ML.__init__(self, kernel)

    def fitNEW(self, x_train, y_train, x_prime_train=None, y_prime_train=None, C=1., error_cap=10**-8, epsilon=0.01,
            max_iter=10**3, eps=10**-4):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self._y = np.concatenate([self.y_train, self.y_prime_train.flatten()])

        size_mat = self.n_samples + self.n_samples_prime * self.n_dim# + 1
        # mat = np.zeros([size_mat, size_mat])
        self._mat = self.create_mat(self.x_train, self.x_train, self.x_prime_train, self.x_prime_train)
        mat = self._mat
        # mat[:self.n_samples, -1] = 1.
        # mat[-1, :self.n_samples] = 1.
        # self._mat = mat

        a = np.zeros(size_mat)
        a[0::2] = 2*C
        a[1::2] = -2*C

        weight = np.zeros(size_mat)
        lagrangian = []
        error = self._errorNEW(weight)
        lagrangian.append(self._lagrangianNEW(weight, error, epsilon, C))

        converged = False
        step = 0
        while not converged:

            ind_a = a != 0
            d_a = 1./a[ind_a] * np.eye(sum(ind_a))
            # ind_a = np.concatenate([ind_a, np.array([1])])
            ind_mat = np.tile(ind_a, size_mat).reshape(size_mat, size_mat)
            calc_mat = mat[np.logical_and(ind_mat, ind_mat.T).reshape(size_mat,size_mat)].reshape(sum(ind_a), sum(ind_a))
            calc_mat[:, :] += d_a

            weight_s = np.zeros(size_mat)
            weight_s[ind_a] = np.linalg.inv(calc_mat).dot(self._y[ind_a])

            nu = 0.1
            weight_b = weight * (1 - nu) + nu * weight_s
            error = self._errorNEW(weight_b)
            l = self._lagrangianNEW(weight, error, epsilon, C)

            if (abs(lagrangian[-1]-l)/lagrangian[-1]) < eps:
                converged = True
            else:
                con = True
                m = .5
                s = 0
                while con:
                    w = weight * (1-m) + weight_s * m
                    error = self._errorNEW(weight_b)
                    l = self._lagrangianNEW(weight, error, epsilon, C)
                    # print(sum(error))
                    if l <= (lagrangian[-1]*(1 + 10**-3)):
                        con = False
                    else:
                        m /= 2
                        s += 1
                    if s > 1000:
                        print(w)
                        print(error)
                        print(l)
                        print(lagrangian)
                        raise ValueError('IS NOT CONVERGED')
                weight_b = w.copy()
            lagrangian.append(l)
            weight = weight_b.copy()

            error = self._errorNEW(weight)
            a = np.zeros(size_mat)
            ind = error >= epsilon
            a[ind] = 2*C*(error[ind] - epsilon)/error[ind]
            ind = error == 0
            a[ind] = error_cap
            a[a > error_cap] = error_cap

            step += 1
            if step > max_iter:
                raise UserWarning('Not converged?')
                break

        self._alpha = weight[:self.n_samples]
        self._beta = weight[self.n_samples:].reshape(-1,self.n_dim)
        self._intercept = 0# weight[-1]
        self._support_index_beta = np.array([np.arange(1, self.n_samples_prime)]*self.n_dim)
        self._support_index_alpha = np.arange(1, self.n_samples, 1)
        self._is_fitted = True

    def _lagrangianNEW(self, weight, error, epsilon, C):
        def loss(error, epsilon):
            return error**2 + epsilon**2 - 2*error*epsilon

        return 0.5*weight.T.dot(self._mat.dot(weight)) + C*np.sum(loss(error, epsilon))

    def _errorNEW(self, weight):
        error = weight.dot(self._mat) - self._y
        return np.sqrt(error**2)


    def _error_weight(self, error, constant, error_cap):
        """
        maximum error function with the cut at error_cap to avoid dividing by zero
        :param error:
        :param constant:
        :param error_cap:
        :return: error weights
        """
        weight = np.minimum(np.maximum(0., constant/error), constant/error_cap)
        return weight

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, C1=1., C2=1., error_cap=10**-8, epsilon=0.01,
            max_iter=10**3, eps=10**-4):
        """
        Fit the model with given trainin data
        :param x_train: function input pattern shape = [N_samples, N_features]
        :param y_train: function values shape = [N_samples, 1]
        :param x_prime_train: derivative input pattern shape = [N_samples, N_features]
        :param y_prime_train: derivative values shape = [N_samples, N_features]
        :param C1: penalty parameter to the function error
        :param C2: penalty parameter to the derivative error
        :param error_cap: cuts the maximum function
        :param epsilon: parameter for the insensitive region around the curve
        :param max_iter: maximum iterations
        :param eps: used for convergence, if norm of the weights changes from on iteration to the next less than eps
        :return:
        """
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self.y_prime_train = self.y_prime_train.flatten()

        size_mat = self.n_samples + self.n_samples_prime * self.n_dim + 1
        mat = np.zeros([size_mat, size_mat])
        self._mat = self.create_mat(self.x_train, self.x_train, self.x_prime_train, self.x_prime_train)
        mat[:-1, :-1] = self._mat
        mat[:self.n_samples, -1] = 1.
        mat[-1, :self.n_samples] = 1.

        lagrangian = []

        # init error weights
        a = np.zeros(self.n_samples * 2)
        s = np.zeros(self.n_samples_prime * self.n_dim * 2)
        a[0:self.n_samples:2] = C1
        a[self.n_samples + 1::2] = C1
        s[0:self.n_samples_prime * self.n_dim:2] = C2
        s[self.n_samples_prime * self.n_dim + 1::2] = C2

        support_index_alpha = np.arange(0, self.n_samples, 1)
        support_index_beta = np.arange(0, self.n_samples_prime * self.n_dim, 1)

        step = 0
        converged = False
        alpha = np.zeros(self.n_samples)
        beta = np.zeros(self.n_samples_prime* self.n_dim)
        b = 0.

        f_error, g_error = self._error_function(alpha, beta, b, epsilon)
        a = self._error_weight(f_error, C1, error_cap)
        s = self._error_weight(g_error, C2, error_cap)

        lagrangian.append(self._lagrangian(alpha, beta, np.array([f_error[a > 0.], g_error[s > 0.]]), C1, C2))

        while not converged:

            index_a = np.logical_or(a[:self.n_samples] > 0., a[self.n_samples:] > 0.)
            index_s = np.logical_or(s[:self.n_samples_prime * self.n_dim] > 0., s[self.n_samples_prime * self.n_dim:] > 0.)

            a_ = a[:self.n_samples][index_a]
            a_star_ = a[self.n_samples:][index_a]
            s_ = s[:self.n_samples_prime * self.n_dim][index_s]
            s_star_ = s[self.n_samples_prime * self.n_dim:][index_s]

            self._support_index_alpha = support_index_alpha[index_a]
            self._support_index_beta = support_index_beta[index_s]
            #
            n_support_alpha = len(self._support_index_alpha)
            n_support_beta = len(self._support_index_beta)

            d_a = (np.eye(n_support_alpha) / (a_ - a_star_))
            d_s = (np.eye(n_support_beta) / (s_ - s_star_))

            weight_index = np.concatenate([index_a, index_s, np.array([1])])

            index = np.logical_and(np.tile(weight_index, size_mat).reshape(size_mat, size_mat),
                                   np.tile(weight_index, size_mat).reshape(size_mat, size_mat).T)

            calc_mat = mat[index].reshape(n_support_alpha + n_support_beta + 1, n_support_alpha + n_support_beta + 1)

            if n_support_alpha == 0:
                # to avoid the singularity if just derivatives occur as support vectors
                calc_mat[-1, -1] = 1
                print('avoid singularity ' + str(step))

            calc_mat[:n_support_alpha, :n_support_alpha] += d_a
            calc_mat[n_support_alpha:-1, n_support_alpha:-1] += d_s

            target_vec = np.concatenate([self.y_train[self._support_index_alpha] + ((a_ - a_star_) / (a_ + a_star_)) * epsilon,
                                 self.y_prime_train.flatten()[self._support_index_beta]+ ((s_ - s_star_) / (s_ + s_star_)) * epsilon,
                                 np.array([0])])

            weight = np.linalg.inv(calc_mat).dot(target_vec)


            alpha_s = np.zeros(self.n_samples)
            alpha_s[self._support_index_alpha] = weight[:n_support_alpha]
            beta_s = np.zeros(self.n_samples_prime * self.n_dim)
            beta_s[self._support_index_beta] = weight[n_support_alpha:-1]
            b_s = weight[-1]

            # B old *(1-mu) + mu *B new
            eta = 0.1
            alpha_new = alpha * (1-eta) + eta * alpha_s
            beta_new = beta *(1 - eta) + eta * beta_s
            b_new = b*(1 - eta) + eta * b_s

            f_error, g_error = self._error_function(alpha_new, beta_new, b_new, epsilon)
            a = self._error_weight(f_error, C1, error_cap)
            s = self._error_weight(g_error, C2, error_cap)
            l = self._lagrangian(alpha_new, beta_new, np.array([f_error[a > 0.], g_error[s > 0.]]), C1, C2)
            if (abs(l - lagrangian[-1])/lagrangian[-1]) < 10**-8:
                lagrangian.append(l)
                alpha = alpha_new.copy()
                beta = beta_new.copy()
                b = b_new
                converged = True
            else:
                m = 0.5
                con = True
                while con:
                    alpha_n = alpha_new + alpha_s * (1-m)
                    beta_n = beta_new + beta_s* (1 - m)
                    b_n = b_new + b_s * (1 - m)
                    f_error, g_error = self._error_function(alpha_n, beta_n, b_n, epsilon)
                    a = self._error_weight(f_error, C1, error_cap)
                    s = self._error_weight(g_error, C2, error_cap)
                    l = self._lagrangian(alpha_new, beta_new, np.array([f_error[a > 0.], g_error[s > 0.]]), C1, C2)

                    if (l  <= lagrangian[-1]*(1 + 10**-3)):
                        con = False
                    else:
                        m = m/2.

                lagrangian.append(l)
                alpha = alpha_n.copy()
                beta = beta_n.copy()
                b = b_n


            # f_error, g_error = self._error_function(alpha, beta, b, epsilon)
            # f_error_s, g_error_s = self._error_function(alpha_s, beta_s, b_s, epsilon)
            #
            #
            # alpha += dir_alpha * eta
            # beta += dir_beta * eta
            # b += dir_b * eta
            # alpha = alpha_s.copy()
            # beta = beta_s.copy()
            # b = b_s

            f_error, g_error = self._error_function(alpha, beta, b, epsilon)
            a = self._error_weight(f_error, C1, error_cap)
            s = self._error_weight(g_error, C2, error_cap)

            # lagrangian.append(self._lagrangian(alpha, beta, np.array([f_error[a > 0.], g_error[s > 0.]]), C1, C2))


            # if step > 1:
            #     if np.linalg.norm(alpha - alpha_old) < eps and np.linalg.norm(beta-beta_old) < eps and\
            #             abs(b-b_old) < eps:
            #         converged = True
            #         print('parameter converged ' + str(step))
            #
            # if step > max_iter:
            #     print('iteration is not converged ' + str(step))
            #     converged = True
            # step += 1

        self._alpha = alpha.copy()
        self._support_index_alpha = np.arange(1, self.n_samples, 1)
        self._support_index_beta = np.array([np.arange(1, self.n_samples_prime)]*self.n_dim)
        idx_beta = index_s.reshape(-1, self.n_dim)
        self._support_index_beta = []
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0,self.n_samples_prime, 1))#[idx_beta[:, ii]])
        self._beta = beta.copy().reshape(-1, self.n_dim)
        self._intercept = b
        self.y_prime_train = self.y_prime_train.reshape(-1,self.n_dim)
        self._is_fitted = True

        plt.figure()
        plt.plot(lagrangian)

    def _error_function(self, alpha, beta, b, epsilon):
        """
        error calculation to the given training points in fit
        :return: function error, derivative error
        """
        k = self._mat[:self.n_samples, :self.n_samples]
        g = self._mat[:self.n_samples, self.n_samples:]
        k_prime = self._mat[self.n_samples:, :self.n_samples]
        j = self._mat[self.n_samples:, self.n_samples:]

        func_error = np.zeros(self.n_samples * 2)
        grad_error = np.zeros(self.n_samples_prime * self.n_dim * 2)
        if self.n_samples != 0:
            func_prediction = alpha[self._support_index_alpha].dot(k[self._support_index_alpha, :]) +\
                              beta[self._support_index_beta].dot(k_prime[self._support_index_beta, :]) + b
            func_error[:self.n_samples] = func_prediction - self.y_train - epsilon
            func_error[self.n_samples:] = -func_prediction + self.y_train - epsilon
        if self.n_samples_prime != 0:
            grad_prediction = alpha[self._support_index_alpha].dot(g[self._support_index_alpha, :]) +\
                              beta[self._support_index_beta].dot(j[self._support_index_beta, :])
            grad_error[:self.n_samples_prime*self.n_dim] = grad_prediction - self.y_prime_train-epsilon
            grad_error[self.n_samples_prime * self.n_dim:] = -grad_prediction + self.y_prime_train - epsilon

        return func_error, grad_error

    def _lagrangian(self, alpha, beta, error, C1, C2):
        # only for function value
        k = self._mat[:self.n_samples, :self.n_samples]
        g = self._mat[:self.n_samples, self.n_samples:]
        k_prime = self._mat[self.n_samples:, :self.n_samples]
        j = self._mat[self.n_samples:, self.n_samples:]

        if self.n_samples_prime != 0:
            return 0.5 * (alpha.T.dot(k.dot(alpha)) + alpha.T.dot(g.dot(beta)) + beta.T.dot(k_prime.dot(alpha))
               + beta.T.dot(j.dot(beta))) + C1 * sum(error[0]) + C2 * sum(error[1])
        else:
            return 0.5 * (alpha.T.dot(k.dot(alpha))) + C1 * sum(error[0])


class RLS(ML):
    """
    Implementation of the regularized least square method for simultaneous fitting of function value and derivatives
    See:
    """
    def __init__(self, kernel, C1=None, C2=None):
        ML.__init__(self, kernel)
        self.C1 = C1
        self.C2 = C2

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, minimze_b=False, C1=1., C2=1.):
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

        if self.C1 is None:
            self.C1 = C1
        else:
            C1 = self.C1
        if self.C2 is None:
            self.C2 = C2
        else:
            C2 = self.C2

        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        if self.n_samples_prime == 0:
            kernel_mat = create_mat(self.kernel, x_train, x_train, x_prime_train, x_prime_train, eval_gradient=False)
            kernel_mat[:self.n_samples, :self.n_samples] = kernel_mat[:self.n_samples, :self.n_samples]\
                                                           + np.eye(self.n_samples) / C1
        else:
            kernel_mat = create_mat(self.kernel, x_train, x_train, x_prime_train, x_prime_train, dx_max=self.n_dim,
                                                                                                    dy_max=self.n_dim)
            kernel_mat[:self.n_samples, :self.n_samples] = kernel_mat[:self.n_samples, :self.n_samples] \
                                                           + np.eye(self.n_samples)/C1
            kernel_mat[self.n_samples:, self.n_samples:] = kernel_mat[self.n_samples:, self.n_samples:] \
                                                           + np.eye(self.n_samples_prime*self.n_dim) / C2

        if not minimze_b:
            mat_size = self.n_samples + self.n_samples_prime * self.n_dim + 1
            mat = np.zeros([mat_size, mat_size])
            mat[:-1, :-1] = kernel_mat

            if self.n_samples == 0:
                mat[-1, -1] = 1
            else:
                mat[:self.n_samples, -1] = 1
                mat[-1, :self.n_samples] = 1
            # Todo implement inverting scheme
            vec = np.concatenate([self.y_train, self.y_prime_train.flatten('F'), np.zeros([1])])
            self.mat = mat
            mat = np.linalg.inv(mat)

        else:
            vec = np.concatenate([self.y_train, self.y_prime_train.flatten('F')])

            mat = kernel_mat
            self.mat = mat
            mat = np.linalg.inv(mat)

        weight = mat.dot(vec)
        self._alpha = weight[0:self.n_samples].reshape(-1).T
        self.y_debug = vec

        if not minimze_b:
            self._intercept = weight[-1]
            self._beta = weight[self.n_samples:-1].reshape(self.n_dim, -1).T
        else:
            self._intercept = sum(self._alpha)
            self._beta = weight[self.n_samples:].reshape(self.n_dim, -1).T

        self._support_index_alpha = np.arange(0, self.n_samples, 1)
        self._support_index_beta = []
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

        self.debug = kernel_mat
        self._is_fitted = True


class GPR(ML):
    # Todo derivatives fails--> probably kernel is wrong?
    def __init__(self, kernel):
        # RBF kernel only supported and a constant mean over all training points
        # cholesky matrix
        self.L = None
        self.y_mean = None
        self.y_prime_mean = None

        self._y_vec = None
        # noise is added to the main diagonal of the kernel matrix
        # this is done in the log marginal likelihood

        self.noise = None
        self.noise_prime = None
        self._opt_flag = True

        ML.__init__(self, kernel)

    def get_hyper_parameter(self):
        return self.kernel.theta

    def set_hyper_parameter(self, hyper_parameter):
        self.kernel.theta = hyper_parameter

    def get_bounds(self):
        return self.kernel.bounds
        #

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, noise=10**-10, noise_prime=10**-10,
            restarts=0, fit_method='LBFGS_B', optimize_hyperparameter=True, normalize_y=False):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self.noise = noise
        self.noise_prime = noise_prime
        if normalize_y:
            self.y_mean = np.mean(y_train)
        else:
            self.y_mean = 0.
        if self.x_prime_train is None:
            self._y_vec = np.concatenate([self.y_train - self.y_mean])
        else:
            self.y_prime_mean = np.mean(y_prime_train, axis=0)
            self._y_vec = np.concatenate([self.y_train - self.y_mean, self.y_prime_train.flatten('F')])

            # self._y_vec = np.concatenate([self.y_train, self.y_prime_train.flatten()])

        if restarts > 0:
            optimize_hyperparameter = True
        if optimize_hyperparameter:
            # self.optimize(self.get_hyper_parameter())
            initial_hyper_parameters = self.get_hyper_parameter()
            opt_hyper_parameter = []
            value = []
            opt = self._opt_routine(initial_hyper_parameters, method=fit_method)

            opt_hyper_parameter.append(opt[0])
            value.append(opt[1])

            for ii in range(restarts):
                initial_hyper_parameters = []
                bounds = self.kernel.bounds
                for element in bounds:
                    initial_hyper_parameters.append(np.random.uniform(element[0], element[1], 1))
                initial_hyper_parameters = np.array(initial_hyper_parameters)
                opt = self._opt_routine(initial_hyper_parameters, method=fit_method)
                opt_hyper_parameter.append(opt[0])
                value.append(opt[1])

            min_ind = np.argmin(value)
            self.set_hyper_parameter(opt_hyper_parameter[min_ind])

        # print('opt hyper parameter = ' + str(self.get_hyper_parameter()))

        if self.n_samples_prime == 0:
            K = create_mat(self.kernel, self.x_train, self.x_train)
            K += np.eye(K.shape[0]) * self.noise
            self.L, alpha = self._cholesky(K)
            self._alpha = alpha
        else:
            k_mat = create_mat(self.kernel, self.x_train, self.x_train, x1_prime=self.x_prime_train,
                               x2_prime=self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim)
            k_mat[:self.n_samples, :self.n_samples] += self.noise*np.eye(self.n_samples)
            k_mat[self.n_samples:, self.n_samples:] += np.eye(self.n_samples_prime * self.n_dim) * self.noise_prime
            # k_mat += self.noise * np.eye(k_mat.shape[0])

            self.L, alpha = self._cholesky(k_mat)
            self._alpha = alpha[:self.n_samples]
            self._beta = alpha[self.n_samples:].reshape(self.n_dim, -1).T

        self._is_fitted = True
        self._intercept = self.y_mean

    def optimize(self, hyper_parameter):
        self.set_hyper_parameter(hyper_parameter)
        # print(hyper_parameter)
        log_marginal_likelihood, d_log_marginal_likelihood = self.log_marginal_likelihood(derivative=self._opt_flag)

        return -log_marginal_likelihood, -d_log_marginal_likelihood

    def log_marginal_likelihood(self, derivative=False):
        # gives vale of log marginal likelihood with the gradient
        if self.n_samples_prime == 0:
            k_mat, k_grad = create_mat(self.kernel, self.x_train, self.x_train, eval_gradient=True)
            k_mat += np.eye(k_mat.shape[0]) * self.noise
        else:
            k_mat, k_grad = create_mat(self.kernel, self.x_train, self.x_train, self.x_prime_train,
                                        self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim, eval_gradient=True)
            k_mat[:self.n_samples, :self.n_samples] += np.eye(self.n_samples) * self.noise
            k_mat[self.n_samples:, self.n_samples:] += np.eye(self.n_samples_prime*self.n_dim)*self.noise_prime
        L, alpha = self._cholesky(k_mat)
        log_mag_likelihood = -0.5*self._y_vec.dot(alpha) - np.log(np.diag(L)).sum() - L.shape[0]/2.*np.log(2*np.pi)

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
        alpha = cho_solve((L, True), self._y_vec)
        return L, alpha

    def _opt_routine(self, initial_hyper_parameter, method='LBFGS_B'):
        if method == 'LBFGS_B':
            self._opt_flag = True
            opt_hyper_parameter, value, opt_dict = sp_opt.fmin_l_bfgs_b(self.optimize, initial_hyper_parameter,
                                                                    bounds=self.get_bounds())
            if opt_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: %s" % opt_dict)
        elif method == 'BFGS':
            # TODO return function value
            self._opt_flag = False
            opt_hyper_parameter = sp_opt.fmin_bfgs(self.optimize, initial_hyper_parameter)
            value = 0

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
