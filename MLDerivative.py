import numpy as np
import scipy.optimize._minimize as spmin
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as sp_opt
import scipy.spatial.distance as spdist

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
                        self.x_prime_train[self._support_index_beta[ii]], x, dy=ii+1)) for ii in range(self.n_dim)) \
                       + self._intercept

            elif self.n_samples != 0:
                return self._alpha[self._support_index_alpha].dot(
                    self.kernel(self.x_train[self._support_index_alpha], x)) + self._intercept

            else:
                return sum(self._beta[self._support_index_beta[ii], ii].dot(self.kernel(
                    self.x_prime_train[self._support_index_beta[ii]], x, dy=ii+1)) for ii in range(self.n_dim))

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
                        self.x_train[self._support_index_alpha], x, dx=ii+1)) \
                                     + sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, dx=ii+1, dy=jj+1)) for jj in
                        range(self.n_dim)])

            elif self.n_samples != 0:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = self._alpha[self._support_index_alpha].dot(self.kernel(
                        self.x_train[self._support_index_alpha], x, dx=ii+1))

            else:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, dx=ii+1, dy=jj+1)) for jj in
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
        x = x.reshape(-1,2)
        return self.predict(x), self.predict_derivative(x).reshape(-1), None

    def _create_mat(self):
        """
        creates the matrix to the lagrangian
        See
        :return: matrix elements k, g, k_prime, j
        """
        # [[k,        g]
        #  [k_prime,  j]]

        if self.n_samples != 0:
            k = self._create_mat_part(self.x_train, self.x_train, dxy=0)
        else:
            k = np.zeros([self.n_samples, self.n_samples])

        if self.n_samples_prime != 0:
            j = self._create_mat_part(self.x_prime_train, self.x_prime_train, dxy=2)
        else:
            j = np.zeros([self.n_samples_prime*self.n_dim, self.n_samples_prime*self.n_dim])

        if self.n_samples != 0 and self.n_samples_prime != 0:
            g, k_prime = self._create_mat_part(self.x_train, self.x_prime_train, dxy=1)
        else:
            g = np.zeros([self.n_samples, self.n_samples_prime*self.n_dim])
            k_prime = g.T

        return k, g, k_prime, j

    def _create_mat_part(self, x, y, dxy=0):
        """
        creates parts of the matrix
        caution if dxy = 0 x, y are both function pattern
                if dxy = 1 x is function pattern and y is the prime pattern
                if dxy = 2 x, y are both the prime pattern
        :param x: first pattern
        :param y: second pattern
        :param dxy: determines which part is created. default: dxy=0 the simple kernel is created;
                                                               dxy=1 creates the first order derivative matrix
                                                               dxy=2 creates the second order derivative matrix
        :return: in case of dxy = 1 two matrix's are returned else just one
        """
        if dxy == 0:
            return self.kernel(x, y)

        elif dxy == 1:
            g = np.zeros([len(x), len(y)*len(y[0])])
            # k_prime = np.zeros([len(y)*len(y[0]), len(x)])
            for ii in range(0, len(y[0])):
                ind1 = [ii * len(y), (ii + 1) * len(y)]
                ind2 = [0, len(x)]
                g[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel(y, x, dx=ii+1, dy=0)
                # k_prime[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel(y, x, dy=0, dx=ii+1)

            return g, g.T

        else:
            j = np.zeros([len(x)*len(x[0]), len(y)*len(y[0])])
            for ii in range(0, len(x[0])):
                for jj in range(0, len(y[0])):
                    ind1 = [ii * self.n_samples_prime, (ii + 1) * self.n_samples_prime]
                    ind2 = [jj * self.n_samples_prime, (jj + 1) * self.n_samples_prime]
                    j[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel(x, y, dx=ii + 1, dy=jj + 1)
            return j

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
    """
    Implementation of the iterative reweighed least square support vector machine for the simultaneous learning of the
    function value and derivatives
    See:
    """
    # Todo lagrangian function

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
        k, g, k_prime, j = self._create_mat()

        a = np.zeros(self.n_samples * 2)
        s = np.zeros(self.n_samples_prime * self.n_dim * 2)

        a[1:self.n_samples:2] = C1
        a[self.n_samples::2] = C1
        s[1:self.n_samples_prime * self.n_dim:2] = C2
        s[self.n_samples_prime * self.n_dim::2] = C2

        support_index_alpha = np.arange(0, self.n_samples, 1)
        support_index_beta = np.arange(0, self.n_samples_prime * self.n_dim, 1)

        size_mat = self.n_samples + self.n_samples_prime * self.n_dim + 1
        mat = np.zeros([size_mat, size_mat])

        mat[:self.n_samples, :self.n_samples] = k
        mat[:self.n_samples, self.n_samples:-1] = g
        mat[self.n_samples:-1, :self.n_samples] = k_prime
        mat[self.n_samples:-1, self.n_samples:-1] = j
        mat[:self.n_samples, -1] = 1.
        mat[-1, :self.n_samples] = 1.

        step = 0
        converged = False
        alpha = np.zeros(self.n_samples)
        beta = np.zeros(self.n_samples_prime * self.n_dim)
        b = 0.

        alpha_old = 0
        beta_old = 0
        b_old = 0

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

            d_a = (np.eye(n_support_alpha) / (a_ + a_star_))
            d_s = (np.eye(n_support_beta) / (s_ + s_star_))

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

            target_vec = np.concatenate([self.y_train[self._support_index_alpha],# + ((a_ - a_star_) / (a_ + a_star_)) * epsilon,
                                 self.y_prime_train.flatten()[self._support_index_beta],# + ((s_ - s_star_) / (s_ + s_star_)) * epsilon,
                                 np.array([0])])

            weight = np.linalg.inv(calc_mat).dot(target_vec)

            alpha_s = np.zeros(self.n_samples)
            alpha_s[self._support_index_alpha] = weight[:n_support_alpha]
            beta_s = np.zeros(self.n_samples_prime * self.n_dim)
            beta_s[self._support_index_beta] = weight[n_support_alpha:-1]
            b_s = weight[-1]

            dir_alpha = alpha - alpha_s
            dir_beta = beta - beta_s
            dir_b = b - b_s

            f_error, g_error = self._error_function(alpha, beta, b, k, g, k_prime, j, epsilon)
            f_error_s, g_error_s = self._error_function(alpha_s, beta_s, b_s, k, g, k_prime, j, epsilon)

            f_s = np.logical_and(f_error < 0., f_error_s > 0.)
            g_s = np.logical_and(g_error < 0., g_error_s > 0.)
            eta = 1.
            if f_s.any() or g_s.any():
                eta = np.min(np.concatenate([f_error[f_s]/(f_error-f_error_s)[f_s], g_error[g_s]/(g_error-g_error_s)[g_s]]))

            alpha += dir_alpha * eta
            beta += dir_beta * eta
            b += dir_b * eta
            alpha = alpha_s.copy()
            beta = beta_s.copy()
            b = b_s

            f_error, g_error = self._error_function(alpha, beta, b, k, g, k_prime, j, epsilon)
            a = self._error_weight(f_error, C1, error_cap)
            s = self._error_weight(g_error, C2, error_cap)

            if step > 1:
                if np.linalg.norm(alpha - alpha_old) < eps and np.linalg.norm(beta-beta_old) < eps and\
                        abs(b-b_old) < eps:
                    converged = True
                    print('parameter converged ' + str(step))

            if step > max_iter:
                print('iteration is not converged ' + str(step))
                converged = True
            step += 1

            alpha_old = alpha.copy()
            beta_old = beta.copy()
            b_old = b

        self._alpha = alpha.copy()
        idx_beta = index_s.reshape(-1, self.n_dim)
        self._support_index_beta = []
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0,self.n_samples_prime, 1)[idx_beta[:, ii]])
        self._beta = beta.copy().reshape(-1, self.n_dim)
        self._intercept = b
        self.y_prime_train = self.y_prime_train.reshape(-1,self.n_dim)
        self._is_fitted = True

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

    def _error_function(self, alpha, beta, b, k, g, k_prime, j, epsilon):
        """
        error calculation to the given training points in fit
        :return: function error, derivative error
        """
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

    def _lagrangian(self, alpha_, beta_, b_):
        pass


class RLS(ML):
    """
    Implementation of the regularized least square method for simultaneous fitting of function value and derivatives
    See:
    """

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
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)

        k, g, k_prime, j = self._create_mat()
        k = k + np.eye(self.n_samples)/C1
        j = j + np.eye(self.n_samples_prime*self.n_dim)/C2
        if not minimze_b:
            mat_size = self.n_samples + self.n_samples_prime * self.n_dim + 1
            mat = np.zeros([mat_size, mat_size])
            mat[:self.n_samples, :self.n_samples] = k #+ np.eye(self.n_samples)/C1
            mat[:self.n_samples, self.n_samples:-1] = g
            mat[self.n_samples:-1, :self.n_samples] = k_prime
            mat[self.n_samples:-1, self.n_samples:-1] = j #+ np.eye(self.n_samples_prime*self.n_dim)/C2

            if self.n_samples == 0:
                mat[-1, -1] = 1
            else:
                mat[:self.n_samples, -1] = 1
                mat[-1, :self.n_samples] = 1
            # Todo implement inverting scheme
            vec = np.concatenate([self.y_train, self.y_prime_train.flatten(), np.zeros([1])])
            mat = np.linalg.inv(mat)

        else:
            vec = np.concatenate([self.y_train, self.y_prime_train.flatten()])
            mat_size = self.n_samples + self.n_samples_prime * self.n_dim
            mat = np.zeros([mat_size, mat_size])

            if self.n_dim == 1:
                # inverting matrix with the scheme for one dimension of the paper ..
                k = np.linalg.inv(k + 1)

                v = np.linalg.inv(j - k_prime.dot(k.dot(g)))
                u = -k.dot(g.dot(v))
                w = -v.dot(k_prime.dot(k))
                t = k - k.dot(g.dot(w))

                mat[:self.n_samples, :self.n_samples] = t
                mat[self.n_samples:, :self.n_samples] = w
                mat[:self.n_samples, self.n_samples:] = u
                mat[self.n_samples:, self.n_samples:] = v
            else:
                mat[:self.n_samples, :self.n_samples] = k
                mat[:self.n_samples, self.n_samples:-1] = g
                mat[self.n_samples:-1, :self.n_samples] = k_prime
                mat[self.n_samples:-1, self.n_samples:-1] = j
                mat = np.linalg.inv(mat)

        weight = mat.dot(vec)

        self._alpha = weight[0:self.n_samples].reshape(-1).T

        if not minimze_b:
            self._intercept = weight[-1]
            self._beta = weight[self.n_samples:-1].reshape(-1, self.n_dim)
        else:
            self._intercept = sum(self._alpha)
            self._beta = weight[self.n_samples:].reshape(-1, self.n_dim)

        self._support_index_alpha = np.arange(0, self.n_samples, 1)
        self._support_index_beta = []
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

        self._is_fitted = True


class GPR(ML):
    # Todo optimize hyperparameters (minimizing them with respect to mean+liklihood+covariance
    # Todo use of hyperparameters
    # Todo covariance matrix --> RBF two hyperparameter
    # Todo liklihood --> gaussian one hyperparamter
    # Todo mean function --> constant mean --> mean for all is [mean(y), mean(y_prime)]
    def __init__(self, kernel):
        # RBF kernel only supported and a constant mean over all training points
        # 0:1 covariance [length scale, signal variance], 2 likelihood [variance], 3 mean [constant]
        # self.hyper_parameter = np.array([0, 0., np.log(.1), 0.0])
        # 1:2 covariance [length scale, signal variance], 3 likelihood [variance], 0 mean [constant]
        self.hyper_parameter = np.array([0.0, 0.0, 0.0, np.log(.1)])

        # chlesky matrix
        self.L = None
        self.noise_percsion = None
        self.y = None

        ML.__init__(self, RBF(gamma=0., amplitude=0.))
        # self.set_hyper_parameters_rbf_kernel()

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.n_samples = len(x_train)
        # self.hyper_parameter[3] = np.mean(y_train)
        self.hyper_parameter[0] = np.mean(y_train)

        hyper_parameter = self.hyper_parameter
        testarray = np.array([3.27256261, 1.62061382, 1.25046715,-9.44355913])
        debugging = self.optimize(testarray)
        debugging = self.derivative_optimize(testarray)
        opt = sp_opt.fmin_cg(self.optimize, hyper_parameter, self.derivative_optimize, full_output=True, disp=True, maxiter=1000)
        hyper_parameter = opt[0]
        likelihood_value = opt[1]
        # print(opt[1])
        # if opt[4] == 2:
        #     search_range = 4# -5, 5
        #     hyper_rand = hyper_parameter
        #     for ii in range(30):
        #         hyper_rand[1:4] = (np.random.randn(3)-0.5)*search_range
        #         opt = sp_opt.fmin_cg(self.optimize, hyper_rand, self.derivative_optimize, full_output=True, disp=False)
        #         print(opt[1])
        #         if opt[1] < likelihood_value:
        #             likelihood_value = opt[1]
        #             hyper_parameter = hyper_rand
        self.hyper_parameter = hyper_parameter
        self.log_marginal_likelihood()

    def optimize(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter
        self.set_hyper_parameters_rbf_kernel()
        return -self.log_marginal_likelihood()

    def derivative_optimize(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter
        self.set_hyper_parameters_rbf_kernel()
        return self.derivative_log_margina_likelihood()

    def log_marginal_likelihood(self):
        m = self.hyper_parameter[0]*np.ones(self.n_samples)
        K = self.kernel(self.x_train, self.x_train)
        likelihood_noise_variance = np.exp(2*self.hyper_parameter[3])

        L = np.linalg.cholesky(K/likelihood_noise_variance+np.eye(self.n_samples)).T
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, self.y_train- m))/likelihood_noise_variance
        self._alpha = alpha
        self.noise_percsion = np.ones([self.n_samples, 1])/np.sqrt(likelihood_noise_variance)
        self.L = L

        nlz = -0.5*np.dot(self.y_train-m, alpha) - np.log(np.diag(L)).sum() - self.n_samples*np.log(2*np.pi*likelihood_noise_variance)/2.
        return nlz

    def derivative_log_margina_likelihood(self):
        likelihood_noise_variance = np.exp(2 * self.hyper_parameter[3])
        K = self.kernel(self.x_train, self.x_train)
        m = self.hyper_parameter[0]*np.ones(self.n_samples)

        L = np.linalg.cholesky(K / likelihood_noise_variance + np.eye(self.n_samples)).T
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, self.y_train - m))/likelihood_noise_variance

        dnlz = np.zeros(len(self.hyper_parameter))
        Q = 1./likelihood_noise_variance*np.linalg.solve(L, np.linalg.solve(L.T, np.eye(self.n_samples))) - np.dot(alpha.reshape(-1,1), alpha.reshape(-1,1).T)
        dnlz[1] = 0.5*(Q*self.kernel(self.x_train, self.x_train, dp=1)).sum()
        dnlz[2] = 0.5*(Q*self.kernel(self.x_train, self.x_train, dp=2)).sum()
        dnlz[0] = -np.sum(alpha)
        dnlz[3] = np.trace(Q)*likelihood_noise_variance

        return dnlz

    def set_hyper_parameters_rbf_kernel(self):
        self.kernel.gamma = 1./np.exp(0.5*self.hyper_parameter[1]**2) #1./np.exp(0.5*)
        self.kernel.amplitude = np.exp(2*self.hyper_parameter[2])#

    def predict(self, x):
        n = len(x)
        K = np.exp(2*self.hyper_parameter[2])*np.exp(-0.5*np.zeros([n, 1]))
        K_cross = self.kernel(self.x_train, x)
        m = self.hyper_parameter[0]*np.ones(n)

        Fmu = m + np.dot(K_cross.T, self._alpha)
        V = np.linalg.solve(self.L.T, np.tile(self.noise_percsion, (1, n)) * K_cross)
        fs2= K - np.array([(V * V).sum(axis=0)]).T
        fs2 = np.maximum(fs2, 0)  # remove numerical noise i.e. negative variances
        Fs2 = np.tile(fs2, (1, 1))
        Fs2 += np.exp(2*self.hyper_parameter[3])
        predict_variance = np.reshape(np.reshape(Fs2,(np.prod(Fs2.shape),1)).sum(axis=1), (-1,1))
        prediction = np.reshape( np.reshape(Fmu,(np.prod(Fmu.shape),1)).sum(axis=1) , (-1,1) )

        return prediction, predict_variance


class RBF:
    def __init__(self, gamma=0.0, amplitude=0.):
        self.gamma = gamma
        self.amplitude = amplitude

    def __call__(self, x, y, dx=0, dy=0, dp=0):
        # dp derivative of the parameters is used for the GPR
        # in case of GPR the gamma have to be redefined outside to gamma = 1/(2*exp(length scale)) because length scale
        # is the hyper parameter of interest
        length_scale = np.exp(self.gamma)
        signal_variance = np.exp(2*self.amplitude)
        mat = spdist.cdist(x/length_scale, y/length_scale, 'sqeuclidean')
        exp_mat = signal_variance * np.exp(-0.5* mat)
        if dp == 0:
            return exp_mat
        elif dp == 1:
            # derivative of the length scale --> gamma = 1/(2*exp(length scale))
            return exp_mat * mat

        elif dp == 2:
            # derivative of the amplitude (variance)
            return exp_mat * 2.

