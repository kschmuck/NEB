import numpy as np
import scipy.optimize._minimize as spmin
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, kernel='rbf', gamma=0.1):
        self.x_train = None
        self.x_prime_train = None
        self.y_train = None
        self.y_prime_train = None
        self.n_samples = None
        self.n_samples_prime = None
        self.n_dim = None

        if kernel =='rbf':
            self.kernel = RBF(gamma=gamma)

        self._is_fitted = False

        self._intercept = None
        self._alpha = None
        self._support_index_alpha = None
        self._beta = None
        self._support_index_beta = None

    def _fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_prime_train = x_prime_train
        self.y_prime_train = y_prime_train

        self.n_samples = len(y_train)
        self.n_samples_prime, self.n_dim = y_prime_train.shape

    def predict(self, x):
        if self._is_fitted:
            if self.n_samples != 0 and self.n_samples_prime != 0:
                return self._alpha[self._support_index_alpha].dot(self.kernel.kernel(
                    self.x_train[self._support_index_alpha], x)) + sum(
                    self._beta[self._support_index_beta[ii], ii].dot(self.kernel.kernel(
                        self.x_prime_train[self._support_index_beta[ii]], x, ny=ii)) for ii in range(self.n_dim)) \
                       + self._intercept

            elif self.n_samples != 0:
                return self.alpha[self._support_index_alpha].dot(
                    self.kernel.kernel(self.x_train[self._support_index_alpha], x)) + self._intercept

            else:
                return sum(self._beta[self._support_index_beta[ii], ii].dot(self.kernel.kernel(
                        self.x_prime_train[self._support_index_beta[ii]], x, ny=ii)) for ii in range(self.n_dim))

        else:
            raise ValueError('not fitted yet')

    def predict_derivative(self, x):
        if self._is_fitted:
            ret_mat = np.zeros((len(x), self.n_dim))
            if self.n_samples_prime != 0 and self.n_samples != 0:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = self._alpha[self._support_index_alpha].dot(self.kernel.kernel(
                        self.x_train[self._support_index_alpha], x, nx=ii)) \
                                     + sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, nx=ii, ny=jj)) for jj in
                        range(self.n_dim)])

            elif self.n_samples != 0:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = self._alpha[self._support_index_alpha].dot(self.kernel.kernel(
                        self.x_train[self._support_index_alpha], x, nx=ii))

            else:
                for ii in range(self.n_dim):
                    ret_mat[:, ii] = sum([self._beta[self._support_index_beta[jj], jj].dot(self.kernel.kernel(
                        self.x_prime_train[self._support_index_beta[jj]], x, nx=ii, ny=jj)) for jj in
                        range(self.n_dim)])
        else:
            raise ValueError('not fitted yet')

    def _create_mat(self, C1=None, C2=None):

        # [[k, g]
        #  [k_prime,  j]]

        k = np.zeros([self.n_samples, self.n_samples])
        g = np.zeros([self.n_samples, self.n_samples_prime*self.dim])
        k_prime = np.zeros([self.n_samples_prime*self.dim, self.n_samples])
        j = np.zeros([self.n_samples_prime*self.dim, self.n_samples_prime*self.dim])

        if self.n_samples_prime != 0:
            j = np.zeros([self.n_samples_prime * self.dim, self.n_samples_prime * self.dim])
            for nx in range(0, self.dim):
                ind = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]
                mat = self.kernel.kernel(self.x_prime, self.x_prime, nx=nx, ny=nx)
                if C2 is not None:
                    mat += np.eye(self.n_samples_prime) / C2
                j[ind[0]:ind[1], ind[0]:ind[1]] = mat

        if self.n_samples != 0:
            mat = self.kernel.kernel(self.x, self.x)
            if C1 is not None:
                mat += np.eye(self.n_samples)/C1
            k[:,:] = mat

        if self.n_samples != 0 and self.n_samples_prime != 0:
            for nx in range(-1, self.dim):
                for ny in range(nx + 1, self.dim):
                    if nx == -1:
                        ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
                        ind2 = [0, self.n_samples]
                    else:
                        ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
                        ind2 = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]

                    k_prime[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny)
                    g[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny).T

        return k, g, k_prime, j

class IRWLS(SVM):
    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, C1=1., C2=1., error_cap=10**-8,
                epsilon=0.1, max_iter=10**4, eps=10**-8):
        self.error_cap = error_cap
        self.epsilon = epsilon
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self.y_prime_train = self.y_prime_train.flatten()

        mat = np.zeros([self.n_samples + self.n_samples_prime * self.n_dim + 1,
                        self.n_samples + self.n_samples_prime * self.n_dim + 1])

        mat[-1, :self.n_samples] = 1.
        mat[:self.n_samples, -1] = 1.

        mat[:self.n_samples, :self.n_samples], mat[self.n_samples:-1, :self.n_samples], mat[:self.n_samples,
                                    self.n_samples:-1], mat[self.n_samples:-1, self.n_samples:-1] = self._create_mat()

        a = np.zeros(self.n_samples)
        a_star = np.zeros(self.n_samples)
        s = np.zeros(self.n_samples_prime * self.n_dim)
        s_star = np.zeros(self.n_samples_prime * self.n_dim)

        a[0::2] = C1
        a_star[1::2] = C1
        s[0::2] = C2
        s_star[1::2] = C2

        step = 0
        converged = False

        alpha = np.zeros(self.n_samples)
        beta = np.zeros(self.n_samples_prime*self.n_dim)
        b = 0.

        while not converged:
            alpha_index = np.logical_and(a > 0., a_star > 0.)
            beta_index = np.logical_and(s > 0., s_star > 0.)

            mat_index = np.logical_and.outer(np.concatenate([alpha_index, beta_index, np.ones(1, dtype=bool)]),
                                             np.concatenate([alpha_index, beta_index, np.ones(1, dtype=bool)]))

            self._support_index_alpha = np.arange(0, self.n_samples, 1)[alpha_index]
            self._support_index_beta = np.arange(0, self.n_samples_prime * self.dim, 1)[beta_index]

            num_alpha = len(self._support_index_alpha)
            num_beta = len(self._support_index_beta)

            d_a = np.linalg.inv(np.eye(num_alpha) * (a + a_star)[self._support_index_alpha])
            d_s = np.linalg.inv(np.eye(num_beta) * (s + s_star)[self._support_index_beta])

            mat_calc = mat[mat_index].reshape([num_alpha + num_beta + 1, num_alpha + num_beta + 1])

            if num_alpha == 0:
                mat_calc[-1, -1] = 1

            mat_calc[:num_alpha, :num_alpha] += d_a
            mat_calc[num_alpha:-1, num_alpha:-1] += d_s

            vec = np.concatenate([self.y_train[self._support_index_alpha] + self.epsilon *
                                 ((a - a_star)/(a + a_star))[self._support_index_alpha],
                                 self.y_prime_train[self._support_index_beta] + + self.epsilon *
                                 ((s - s_star)/(s + s_star))[self._support_index_beta], np.array([0])])

            weights = np.linalg.solve(mat_calc, vec)

            alpha_s = np.zeros(self.n_samples)
            beta_s = np.zeros(self.n_samples_prime*self.n_dim)

            alpha_s[self._support_index_alpha] = weights[:num_alpha]
            beta_s[self._support_index_beta] = weights[num_beta:-1]
            b_s = weights[-1]

            func_error, func_error_star, grad_error, grad_error_star = self.calc_error(alpha, beta, b)
            func_error_s, func_error_star_s, grad_error_s, grad_error_star_s = self.calc_error(alpha_s, beta_s, b_s)

            func_s = np.concatenate([np.logical_and(func_error < 0., func_error_s > 0.),
                          np.logical_and(func_error_star < 0., func_error_star_s > 0.)])
            grad_s = np.concatenate([np.logical_and(grad_error < 0., grad_error_s > 0.),
                          np.logical_and(grad_error_star < 0., grad_error_star_s > 0.)])

            if not (func_s.any() or grad_s.any()):
                alpha = alpha_s.copy()
                beta = beta_s.copy()
            else:
                func_error = np.concatenate([func_error, func_error_star])
                grad_error = np.concatenate([grad_error, grad_error_star])
                func_error_s = np.concatenate([func_error_s, func_error_star_s])
                grad_error_s = np.concatenate([grad_error_s, grad_error_star_s])

                eta = np.min([(func_error[func_s]/(func_error_s-func_error)[func_s])],
                             (grad_error[grad_s])/(grad_error_s-grad_error)[grad_s])

                alpha += eta*(alpha-alpha_s)
                beta += eta*(beta-beta_s)
            b = b_s

            func_error, func_error_star, grad_error, grad_error_star = self.calc_error(alpha, beta, b)
            a = self.error_weight(func_error, C1)
            a_star = self.error_weight(func_error_star, C1)
            s = self.error_weight(grad_error, C2)
            s_star = self.error_weight(grad_error_star, C2)

            if step > 0:
                if np.linalg.norm(alpha-alpha_old) < eps and np.linalg.norm(beta-beta_old) < eps and np.linalg.norm(b-b_old):
                    print('Converged after {} iterations'.format(step))
                    converged = True

            alpha_old = alpha.copy()
            beta_old = beta.copy()
            b_old = b
            step += 1
            if step >= max_iter:
                print('not converged')
                converged = True
        self.y_prime_train = self.y_prime_train.reshape(-1, self.n_dim)
        self._alpha = alpha
        self._beta = beta.reshape(-1, self.n_dim)
        self._intercept = b

        beta_index = beta_index.reshape(-1,self.n_dim)
        self._support_index_beta = []

        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime, 1)[beta_index[ii]])

    def error_weight(self, error, constant):
        weight = np.zeros(error.shape)
        weight[(error < self.error_cap) & (error >= 0)] = constant / self.error_cap
        weight[error > self.error_cap] = constant / error
        return weight

    def calc_error(self, alpha, beta, b, k, k_prime, g, j):

        if self.n_samples != 0:
            function_prediction = alpha.dot(k) + beta.dot(k_prime) + b
            function_error = function_prediction - self.y_train - self.epsilon
            function_error_star = -function_prediction + self.y_train - self.epsilon
        else:
            function_error = np.array([])
            function_error_star = np.array([])

        if self.n_samples_prime != 0:
            gradient_prediction = alpha.dot(g) + beta.dot(j)
            gradient_error = gradient_prediction - self.y_prime_train - self.epsilon
            gradient_error_star = -gradient_prediction - self.y_prime_train - self.epsilon
        else:
            gradient_error = np.array([])
            gradient_error_star = np.array([])

        return function_error, function_error_star, gradient_error, gradient_error_star

    def lagrangian(self, alpha, beta, b, k, k_prime, g, j, index_alpha, index_beta):
        pass

class RLS(SVM):
    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, minimze_b=False, C1=1., C2=1.):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)

        k, g, k_prime, j = self._create_mat(C1=C1, C2=C2)

        if minimze_b:
            mat_size = self.n_samples + self.n_samples_prime * self.dim + 1
            mat = np.zeros([mat_size, mat_size])
            mat[:self.n_samples, :self.n_samples] = k
            mat[:self.n_samples, self.n_samples:] = g
            mat[self.n_samples:, :self.n_samples] = k_prime
            mat[self.n_samples:, self.n_samples:] = j

            if self.n_samples == 0:
                mat[-1, -1] = 1
                vec = np.concatenate([self.y, self.y_prime.flatten(), np.zeros([1])])
            else:
                vec = np.concatenate([self.y, self.y_prime.flatten()])
                mat[:self.n_samples, -1] = 1
                mat[-1, :self.n_samples] = 1
            # Todo implement inverting scheme
            mat = np.linalg.inv(mat)

        else:
            vec = np.concatenate([self.y, self.y_prime.flatten()])
            mat_size = self.n_samples + self.n_samples_prime * self.dim
            mat = np.zeros([mat_size, mat_size])

            # inverting matrix with the scheme of the paper ..
            k = np.linalg.inv(k + 1)

            v = np.linalg.inv(j - k_prime.dot(k.dot(g)))
            u = -k.dot(g.dot(v))
            w = -v.dot(k_prime.dot(k))
            t = k - k.dot(g.dot(w))

            mat[:self.n_samples, :self.n_samples] = t
            mat[self.n_samples:, :self.n_samples] = w
            mat[:self.n_samples, self.n_samples:] = u
            mat[self.n_samples:, self.n_samples:] = v

        weight = mat.dot(vec)
        self._alpha = weight[0:self.n_samples].reshape(-1).T
        self._beta = weight[self.n_samples:].reshape((self.dim, -1)).T
        self._intercept = sum(self.alpha)
        self._support_index_alpha = np.arange(0, self.n_samples, 1)
        self._support_index_beta = []
        for ii in range(self.dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

        self._is_fitted = True

# class SVM:
#     def __init__(self, epsilon=0.1, epsilon_beta=0.1, kernel='rbf', gamma=0.1, method='rls'):
#         self.x = None
#         self.x_prime = None
#         self.y = None
#         self.y_prime = None
#
#         self.epsilon = epsilon
#         self.epsilon_beta = epsilon_beta
#
#         self.intercept = None
#         self.alpha = None
#         self.support_index_alpha = None
#
#         self.dim = None
#         self.n_samples = None
#         self.n_samples_prime = None
#
#         self.beta = None
#         self.support_index_beta = None
#
#         if kernel == 'rbf':
#             self.kernel = RBF(gamma=gamma)
#
#         self.method = method
#         self._is_fitted = False
#
#         self.debug_mat = None
#
#     def predict_derivative(self, x):
#         if not self._is_fitted:
#             raise ValueError('not fitted')
#         if self.method == 'simple':
#             raise ValueError('simple method has can not predict derivatives at the moment')
#         ret_mat = np.zeros((len(x), self.dim))
#         for ii in range(self.dim):
#             ret_mat[:, ii] = self.alpha[self.support_index_alpha].dot(self.kernel.kernel(
#                                 self.x[self.support_index_alpha], x, nx=ii))\
#                             + sum([self.beta[self.support_index_beta[jj], jj].dot(self.kernel.kernel(
#                                 self.x_prime[self.support_index_beta[jj]], x, nx=ii, ny=jj)) for jj in
#                                  range(self.dim)])
#         return ret_mat
#
#     def predict(self, x):
#         if not self._is_fitted:
#             raise ValueError('not fitted')
#
#         if self.method == 'irwls_function':
#             return self.alpha[self.support_index_alpha].dot(
#                     self.kernel.kernel(self.x[self.support_index_alpha], x)) + self.intercept
#
#         if self.n_samples_prime != 0:
#             return self.alpha[self.support_index_alpha].dot(self.kernel.kernel(self.x[self.support_index_alpha], x)) \
#                    + sum(self.beta[self.support_index_beta[ii], ii].dot(self.kernel.kernel(
#                        self.x_prime[self.support_index_beta[ii]], x, ny=ii)) for ii in range(self.dim)) + self.intercept
#         else:
#             return self.alpha[self.support_index_alpha].dot(
#                     self.kernel.kernel(self.x[self.support_index_alpha], x)) + self.intercept
#
#     def fit(self, x, y, x_prime=None, y_prime=None, C1=1.0, C2=1.0, max_iter=10**4):
#         self.x = x
#         self.y = y
#         self.dim = x.shape[1]
#
#         if x_prime is None:
#             self.x_prime = np.empty([0, self.dim])
#         else:
#             self.x_prime = x_prime
#
#         if y is None:
#             self.n_samples = 0
#         else:
#             self.n_samples = len(y)
#
#         if y_prime is None:
#             self.n_samples_prime = 0
#             self.y_prime = np.empty(0)
#         else:
#             self.y_prime = y_prime
#             self.n_samples_prime = len(y_prime)
#
#         if self.method == 'irwls':
#             self._fit_irwls(C1=C1, C2=C2)
#             self._is_fitted = True
#         elif self.method == 'rls':
#             self._fit_rls(C1=C1, C2=C2)
#             self._is_fitted = True
#         elif self.method == 'rls_b':
#             self._fit_rls_b(C1=C1, C2=C2)
#             self._is_fitted = True
#         elif self.method == 'simple':
#             self._fit_simple(C1=C1, C2=C2)
#             self._is_fitted = True
#         elif self.method =='irwls_function':
#             self.fit_irwls_new(C1=C1, C2=C2)
#             self._is_fitted = True
#         else:
#             raise NotImplementedError('Method is not implemented use irwls, rls or simple')
#
# # Todo irwls fit d_a and d_s inversion
# # Todo own error Function, maybe own class for the different methods?
#     # def fit_irwls_new(self, C1=1.0, C2=1.0, max_iter=10**2, error_cap=10**-4, eps=10**-8):
#     #     def weight_function(error, constant):
#     #         weight = np.zeros(error.shape)
#     #         weight[error >= 0] = constant / error[error >= 0]
#     #         return weight
#     #
#     #     # def error_function(_alpha, _beta, _b):
#     #     def error_function(_alpha, _b):
#     #         value_error = np.zeros(self.n_samples*2)
#     #         grad_error = np.zeros(num_derivatives*2)
#     #
#     #         # val_prediction = k[:, support_index_alpha].dot(_alpha) + k_prime[:, support_index_beta].dot(_beta) + _b
#     #         val_prediction = k[:, support_index_alpha].dot(_alpha[support_index_alpha]) + _b
#     #
#     #         value_error[:self.n_samples] = val_prediction - self.y - self.epsilon
#     #         value_error[self.n_samples:] = -val_prediction + self.y - self.epsilon
#     #
#     #         # grad_prediction = g[:, support_index_alpha].dot(_alpha) + j[:, support_index_beta].dot(beta)
#     #         # grad_error[:num_derivatives] = grad_prediction - self.y_prime.flatten() - self.epsilon_beta
#     #         # grad_error[num_derivatives:] = -grad_prediction + self.y_prime.flatten() - self.epsilon_beta
#     #         return value_error, grad_error
#     #
#     #     # def lagrangian(_alpha, _beta, _b, _k, _g, _k_prime, _j):
#     #     def lagrangian(_alpha, _b, _k):
#     #         # val_error, grad_error = error_function(_alpha, _beta, _b, _k, _g, _k_prime, _j)
#     #         val_error, grad_error = error_function(_alpha, _b)
#     #         return 1/2. * (_alpha[support_index_alpha].dot(_k.dot(_alpha[support_index_alpha])))\
#     #                + C1 *np.sum(val_error[val_error > 0.])
#     #         # return 1/2. * (alpha.dot(_k.dot(alpha)) + _beta.dot(_j.dot(_beta)) + _alpha.dot(_k_prime.dot(_beta))
#     #         #                + _beta.dot(_g.dot(_alpha))) + C1 * np.sum(val_error[val_error > 0.])\
#     #         #                + C2 * np.sum(grad_error[grad_error > 0.])
#     #
#     #     num_derivatives = self.n_samples_prime*self.dim
#     #     a = np.zeros(self.n_samples*2)
#     #     # s = np.zeros(num_derivatives)
#     #     a[:self.n_samples][1::2] = C1
#     #     a[self.n_samples:][0::2] = C1
#     #     # s[:num_derivatives][1::2] = C2
#     #     # s[num_derivatives:][0::2] = C2
#     #
#     #     k, g, k_prime, j = self._create_mat()
#     #
#     #     converged = False
#     #     step = 0
#     #     l_keep = []
#     #     while not converged:
#     #         a_index = np.logical_or(a[:self.n_samples] > 0., a[self.n_samples:] > 0.)
#     #         # s_index = np.logical_or(s[:num_derivatives] > 0., s[num_derivatives:] > 0.)
#     #
#     #         support_index_alpha = np.arange(0, self.n_samples, 1)[a_index]
#     #         # support_index_beta = np.arange(0, num_derivatives, 1)[s_index]
#     #         num_alpha = len(support_index_alpha)
#     #         # num_beta = len(support_index_beta)
#     #
#     #         add_a = (a[:self.n_samples]-a[self.n_samples:])/(a[self.n_samples:]+a[:self.n_samples])
#     #         # add_s = (s[:num_derivatives]-a[num_derivatives:])/(a[num_derivatives:]+a[:num_derivatives])
#     #
#     #         d_a = np.linalg.inv(np.eye(num_alpha)*(a[:self.n_samples][a_index]+a[self.n_samples:][a_index]))
#     #         # d_s = np.linalg.inv(np.eye(num_alpha)*(s[:num_derivatives][s_index]+s[num_derivatives:][s_index]))
#     #
#     #         # selection of the matrix elements useful for the lagrangian
#     #         k_i = k[:, support_index_alpha][support_index_alpha, :]
#     #         # k_prime_i = k_prime[:, support_index_alpha][support_index_beta, :]
#     #         # g_i = g[:, support_index_beta][support_index_alpha, :]
#     #         # j_i = j[:, support_index_beta][:, support_index_beta]
#     #
#     #         # calc_mat = np.zeros([num_alpha + num_beta + 1, num_alpha + num_beta + 1])
#     #         calc_mat = np.zeros([num_alpha + 1, num_alpha + 1])
#     #
#     #         calc_mat[:num_alpha, :num_alpha] = k_i + d_a
#     #         # calc_mat[num_alpha:-1, num_alpha:-1] = j_i + d_s
#     #         # calc_mat[:num_alpha, num_alpha:-1] = k_prime_i
#     #         # calc_mat[num_alpha:-1, :num_alpha] = g_i
#     #
#     #         # if num_alpha == 0:
#     #         #     calc_mat[-1, -1] = 1.
#     #         # else:
#     #         #     calc_mat[-1, :num_alpha] = 1.
#     #         #     calc_mat[:num_alpha: -1] = 1.
#     #         calc_mat[-1, :num_alpha] = 1.
#     #         calc_mat[:num_alpha, -1] = 1.
#     #
#     #         y = np.concatenate([self.y[support_index_alpha] + add_a[support_index_alpha], np.zeros(1)])
#     #         # y = np.concatenate([self.y[support_index_alpha] + add_a, self.y_prime.flatten()[support_index_beta] + add_s,
#     #         #                    np.zeros(1)])
#     #
#     #         vec_s = np.linalg.inv(calc_mat).dot(y)
#     #         alpha = np.zeros(self.n_samples)
#     #         # beta = np.zeros(num_derivatives)
#     #         alpha[support_index_alpha] = vec_s[:num_alpha]
#     #         # beta[support_index_beta] = vec_s[num_alpha:-1]
#     #         b = vec_s[-1]
#     #
#     #         # f_error, g_error = error_function(alpha, beta, b)
#     #         f_error, g_error = error_function(alpha, b)
#     #         a = weight_function(f_error, C1)
#     #         # s = weight_function(g_error, C2)
#     #         if step > 1:
#     #             if lagrangian(alpha, b, k_i) > l_keep[step-1]:
#     #                 converged = True
#     #                 print('converged')
#     #
#     #         l_keep.append(lagrangian(alpha, b, k_i))
#     #
#     #         step += 1
#     #         if step > max_iter:
#     #             print('not converged')
#     #             break
#     #     self.alpha = alpha.copy()
#     #     self.support_index_alpha = support_index_alpha.copy()
#     #     self.intercept = b
#
#     def _fit_irwls(self, C1=1.0, C2=1.0, max_iter=10**2, error_cap=10**-4, eps=10**-8):
#
#         def calc_weight(error, constant):
#             weight = np.zeros(error.shape)
#             weight[(error < error_cap) & (error >= 0.)] = constant/error_cap
#             weight[error >= error_cap] = constant / error[error >= error_cap]
#             return weight
#
#         def error_function(alpha_, beta_, b_):
#             func_error = np.zeros(self.n_samples*2)
#             grad_error = np.zeros(self.n_samples_prime*self.dim*2)
#             if self.n_samples != 0 and self.n_samples_prime != 0:
#                 func_error[:self.n_samples] = alpha_[idx_alpha].dot(k[idx_alpha, :]) +\
#                                               beta_[idx_beta].dot(k_prime[idx_beta, :]) + b_ - self.y - self.epsilon
#                 func_error[self.n_samples:] = -alpha_[idx_alpha].dot(k[idx_alpha, :]) - \
#                                               beta_[idx_beta].dot(k_prime[idx_beta, :]) - b_ + self.y - self.epsilon
#
#                 grad_error[self.n_samples_prime*self.dim:] = alpha_[idx_alpha].dot(g[idx_alpha, :]) \
#                                                              + beta_[idx_beta].dot(j[idx_beta, :])\
#                                                              - self.y_prime.flatten() - self.epsilon_beta
#                 grad_error[:self.n_samples_prime*self.dim] = -alpha_[idx_alpha].dot(g[idx_alpha, :]) \
#                                                              - beta_[idx_beta].dot(j[idx_beta, :]) \
#                                                              + self.y_prime.flatten() - self.epsilon_beta
#             elif self.n_samples_prime == 0:
#                 func_error[:self.n_samples] = alpha_[idx_alpha].dot(k[idx_alpha, :]) + b_ - self.y - self.epsilon
#                 func_error[self.n_samples:] = - alpha_[idx_alpha].dot(k[idx_alpha, :]) - b_ + self.y - self.epsilon
#
#             return func_error, grad_error
#
#         def lagrangian(alpha_, beta_, b_):
#             func_error, grad_error = error_function(alpha_, beta_, b_)
#             _k = k[:, idx_alpha]
#             _k_prime = k_prime[:, idx_alpha]
#             _g = g[:, idx_beta]
#             _j = j[:, idx_beta]
#
#             return 1 / 2. * (alpha_[idx_alpha].T.dot(_k[idx_alpha, :].dot(alpha_[idx_alpha]))
#                              + alpha_[idx_alpha].T.dot(_g[idx_alpha, :].dot(beta_[idx_beta]))
#                              + beta_[idx_beta].T.dot(_k_prime[idx_beta, :].dot(alpha_[idx_alpha]))
#                              + (beta_[idx_beta].T.dot(_j[idx_beta, :].dot(beta_[idx_beta])))) \
#                              + C1 * np.sum(func_error[func_error > 0.]) + C2 * np.sum(grad_error[grad_error > 0.])
#             # return  C1 * np.sum(func_error[func_error > 0.]) + C2 * np.sum(grad_error[grad_error > 0.])
#
#
#         k, g, k_prime, j = self._create_mat()
#
#         a = np.zeros(self.n_samples * 2)
#         s = np.zeros(self.n_samples_prime*self.dim*2)
#
#         a[1:self.n_samples:2] = C1
#         a[self.n_samples::2] = C1
#         s[1:self.n_samples_prime*self.dim:2] = C2
#         s[self.n_samples_prime*self.dim::2] = C2
#
#         support_index_alpha = np.arange(0, self.n_samples, 1)
#         support_index_beta = np.arange(0, self.n_samples_prime * self.dim, 1)
#
#         size_mat = self.n_samples + self.n_samples_prime*self.dim + 1
#         mat = np.zeros([size_mat, size_mat])
#
#         mat[:self.n_samples, :self.n_samples] = k
#         mat[:self.n_samples, self.n_samples:-1] = g
#         mat[self.n_samples:-1, :self.n_samples] = k_prime
#         mat[self.n_samples:-1, self.n_samples:-1] = j
#         mat[:self.n_samples, -1] = 1.
#         mat[-1, :self.n_samples] = 1.
#         # mat[-1, :-1] = 1.
#         # mat[:-1, -1] = 1.
#
#         step = 1
#         converged = False
#         alpha = np.zeros(self.n_samples)
#         beta = np.zeros(self.n_samples_prime * self.dim)
#         b = 0.
#         l_keep = []
#         alpha_keep = []
#         beta_keep = []
#         b_keep = []
#         f_error_keep = []
#         g_error_keep = []
#
#         x_predict = np.linspace(-11 * np.pi, 11 * np.pi, 1000).reshape(-1, 1)
#         rho = 0.8
#         l_diff = []
#         while not converged:
#
#             index_a = np.logical_or(a[:self.n_samples] > 0., a[self.n_samples:] > 0.)
#             index_s = np.logical_or(s[:self.n_samples_prime*self.dim] > 0., s[self.n_samples_prime*self.dim:] > 0.)
#
#             a_ = a[:self.n_samples][index_a]
#             a_star_ = a[self.n_samples:][index_a]
#             s_ = s[:self.n_samples_prime*self.dim][index_s]
#             s_star_ = s[self.n_samples_prime*self.dim:][index_s]
#
#             idx_alpha = support_index_alpha[index_a]
#             idx_beta = support_index_beta[index_s]
#
#             d_a = np.linalg.inv(np.eye(len(idx_alpha)) * (a_ + a_star_))  # *(e + e_star)[self.support_index_alpha]/C1
#             d_s = np.linalg.inv(np.eye(len(idx_beta)) * (s_ + s_star_))  # (d + d_star)[self.support_index_beta]/C2
#
#             index = np.logical_and(np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
#                                    .reshape(size_mat, size_mat),
#                                    np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
#                                    .reshape(size_mat, size_mat).T)
#
#             calc_mat = mat[index].reshape(len(idx_alpha)+len(idx_beta) + 1, len(idx_alpha)+len(idx_beta) + 1)
#
#             if len(idx_alpha) == 0:
#                 # to avoid the singularity if just derivatives occour as support vectors
#                 calc_mat[-1, -1] = 1
#                 print('avoid singularity ' + str(step))
#
#             calc_mat[:len(idx_alpha), :len(idx_alpha)] += d_a
#             calc_mat[len(idx_alpha):-1, len(idx_alpha):-1] += d_s
#
#             y_ = np.concatenate([self.y[idx_alpha] + ((a_ - a_star_) / (a_ + a_star_)) * self.epsilon,
#                                  self.y_prime.flatten()[idx_beta] + ((s_ - s_star_) / (s_ + s_star_)) * self.epsilon_beta,
#                                  np.array([0])])
#             vec_ = np.linalg.inv(calc_mat).dot(y_)
#
#             alpha_s = np.zeros(self.n_samples)
#             alpha_s[idx_alpha] = vec_[:len(idx_alpha)]
#             beta_s = np.zeros(self.n_samples_prime*self.dim)
#             beta_s[idx_beta] = vec_[len(idx_alpha):-1]
#             b_s = vec_[-1]
#
#             dir_alpha = -alpha + alpha_s
#             dir_beta = -beta + beta_s
#             dir_b = b_s - b
#             eta = 1.
#             l_new = lagrangian(alpha + eta * dir_alpha, beta + eta * dir_beta, b + eta *dir_b)
#             l_dif = lagrangian(alpha_s, beta_s, b_s) - lagrangian(alpha, beta, b)
#             if step > 1:
#                 ii = 0
#                 while l_old <= l_new:
#                     # print(ii)
#                     eta = rho * eta
#                     l_new = lagrangian(alpha + eta * dir_alpha, beta + eta * dir_beta, b + eta *dir_b)
#                     ii += 1
#                     if ii > 10**3:
#                         # converged = True
#                         break
#             print('-------------')
#             print(eta)
#             alpha += dir_alpha*eta
#             beta += dir_beta*eta
#             b += dir_b*eta
#
#             l_diff.append(l_dif)
#             # print(lagrangian(alpha_s, beta_s, b_s) - lagrangian(alpha, beta, b))
#             # f_error, g_error = error_function(alpha, beta, b)
#             # f_error_s, g_error_s = error_function(alpha_s, beta_s, b_s)
#
#             # index_eta_f = np.logical_and(f_error < 0., f_error_s > 0.)
#             # index_eta_g = np.logical_and(g_error < 0., g_error_s > 0.)
#
#             # eta_alpha = 1.
#             # eta_beta = 1.
#             # if step > 1:
#             #     index_eta_f = np.logical_and(f_error < 0., f_error_s > 0.)
#             #     index_eta_g = np.logical_and(g_error < 0., g_error_s > 0.)
#             #     if index_eta_f.any():
#             #         eta_alpha = np.min(f_error[index_eta_f]/(f_error[index_eta_f]-f_error_s[index_eta_f]))
#             #     if index_eta_g.any():
#             #         eta_beta = np.min(g_error[index_eta_g]/(g_error[index_eta_g]-g_error_s[index_eta_g]))
#             #     eta_alpha = np.min([1.0, eta_alpha])
#             #     eta_beta = np.min([1.0, eta_beta])
#
#
#             # eta = np.min([eta_alpha, eta_beta])
#             # alpha += eta_alpha*(alpha_s-alpha)
#             # beta += eta_beta*(beta_s-beta)
#             # b += 1.*(b_s-b)
#             # eta = np.min([eta_alpha, eta_beta])
#             # alpha += eta*(alpha_s-alpha)
#             # beta += eta*(beta_s-beta)
#             # b += eta*(b_s-b)
#
#             # f_error, g_error = error_function(alpha, beta, b)
#
#             # l = lagrangian(alpha, beta, b)
#             #
#             # if l > lagrangian(alpha_s, beta_s, b_s):
#             #     alpha = alpha_s.copy()
#             #     beta = beta_s.copy()
#             #     b = b_s
#
#             f_error, g_error = error_function(alpha, beta, b)
#             a = calc_weight(f_error, C1)
#             s = calc_weight(g_error, C2)
#
#             # convergence criteria
#
#             if step > 1:
#                 l = lagrangian(alpha, beta, b)
#
#                 if np.less(abs(alpha[idx_alpha] - self.alpha[idx_alpha]), eps).all() \
#                         and np.less(abs(beta[idx_beta] - self.beta[idx_beta]), eps).all():
#                     if abs(b - self.intercept) < eps:
#                         converged = True
#                         print('parameter converged ' + str(step))
#
#                 if abs(l - l_old) < eps:
#                     converged = True
#                     print('lagrangian converged step = ' + str(step) + ' num alpha = ' + str(len(idx_alpha))
#                           + ' num beta = ' + str(len(idx_beta)))
#
#                 if l <= l_old:
#                     idx_beta_new = idx_beta.reshape(-1, self.dim)
#                     self.support_index_alpha = idx_alpha.copy()
#                     self.support_index_beta = []
#                     for ii in range(self.dim):
#                         self.support_index_beta.append(support_index_beta[idx_beta_new[:, ii]])
#                     self.alpha = alpha.copy()
#                     self.beta = beta.copy()
#                     self.intercept = b
#                     # converged = False
#                     # print('lagrangian converged step = ' + str(step)+ ' num alpha = '+ str(len(idx_alpha))+' num beta = '
#                     #       + str(len(idx_beta)))
#                 if step > 2:
#                     if abs(l_dif) < 10**-6:
#                         converged = True
#                         print('lagrangian diff converged step = '+ str(step) + ' num alpha = ' + str(len(idx_alpha))
#                               + ' num beta = ' + str(len(idx_beta))+ ' lagrangian diff = ' + str(l_dif))
#                     if step < 10:
#                         converged = False
#             # converged = False
#             l_old = lagrangian(alpha, beta, b)
#             l_keep.append(l_old)
#             alpha_keep.append(alpha)
#             beta_keep.append(beta)
#             b_keep.append(b)
#             f_error_keep.append(np.sum(f_error[idx_alpha]))
#             g_error_keep.append(np.sum(g_error[idx_beta]))
#
#             if step > max_iter:
#                 print('iteration is not converged ' + str(step))
#                 # print(abs(alpha - self.alpha))
#                 # print(abs(beta - self.beta))
#                 # print(abs(b - self.intercept))
#                 break
#             step += 1
#
#             idx_beta = idx_beta.reshape(-1, self.dim)
#             self.support_index_alpha = idx_alpha.copy()
#             self.support_index_beta = []
#             for ii in range(self.dim):
#                 self.support_index_beta.append(support_index_beta[idx_beta[:, ii]])
#             self.alpha = alpha.copy()
#             self.beta = beta.copy()
#             self.intercept = b
#
#             # if not converged:
#             #     idx_beta = idx_beta.reshape(-1, self.dim)
#             #     self.support_index_alpha = idx_alpha.copy()
#             #     self.support_index_beta = []
#             #     for ii in range(self.dim):
#             #         self.support_index_beta.append(support_index_beta[idx_beta[:, ii]])
#             #     self.alpha = alpha.copy()
#             #     self.beta = beta.copy()
#             #     self.intercept = b
#                 # print((f_error))
#             #
#             #     self._is_fitted = True
#             #     self.beta = self.beta.reshape(-1, self.dim)
#             #     print(f_error[:self.n_samples][25:30])
#             #     print(f_error[self.n_samples:][25:30])
#             #     print(g_error[:self.n_samples][25:30])
#             #     print(g_error[self.n_samples:][25:30])
#             #     print('num alpha = ' +str(len(idx_alpha)))
#             #     print(l_old)
#             #     plt.figure()
#             #     plt.plot(x_predict, self.predict(x_predict), 'g')
#             #     plt.plot(self.x, self.y, color='k', ls='None', marker='o')
#             #     # self.support_index_alpha = support_index_alpha
#             #     # plt.plot(x_predict, self.predict(x_predict), 'b')
#             #     plt.plot(x_predict, func(x_predict) - self.epsilon, color='k', ls='--', alpha=0.4)
#             #     plt.plot(x_predict, func(x_predict) + self.epsilon, color='k', ls='--', alpha=0.4)
#             #     plt.plot(self.x[25:30], self.y[25:30], color='r', ls='None', marker='o')
#             #     plt.show()
#             #     self.beta = beta.copy()
#         plt.figure()
#         plt.subplot(121)
#         plt.plot(l_keep)
#         plt.subplot(122)
#         plt.plot(l_diff)
#         # print(l_diff)
#         # plt.subplot(153)
#         # plt.plot(g_error_keep)
#         # plt.subplot(154)
#         # plt.plot(alpha_keep)
#         # plt.subplot(155)
#         # plt.plot(beta_keep)
#         # plt.show()
#         # plt.plot(f_error_keep[:]+g_error_keep[:])
#         # print('max function error = ' + str(np.max(f_error)))
#         if self.n_samples_prime != 0:
#             print('max gradient error = ' + str(np.max(g_error)))
#         self.beta = self.beta.reshape(-1, self.dim)
#
#     # def _fit_rls(self, C1=1.0, C2=1.0):
#     #
#     #     k, g, k_, j = self._create_mat(C1=C1, C2=C2)
#     #
#     #     if self.n_samples == 0:
#     #         # mat = np.zeros([self.n_samples_prime * self.dim, self.n_samples_prime * self.dim])
#     #         mat = j
#     #     elif self.n_samples_prime == 0:
#     #         mat = np.zeros([self.n_samples + 1, self.n_samples + 1])
#     #         mat[:-1, :-1] = k
#     #         mat[-1, :self.n_samples] = 1.
#     #         mat[:self.n_samples, -1] = 1.
#     #     else:
#     #         mat = np.zeros([self.n_samples + self.n_samples_prime * self.dim + 1,
#     #                         self.n_samples + self.n_samples_prime * self.dim + 1])
#     #         mat[:self.n_samples, self.n_samples:-1] = g
#     #         mat[self.n_samples:-1, self.n_samples:-1] = j
#     #         mat[:self.n_samples, :self.n_samples] = k
#     #         mat[self.n_samples:-1, :self.n_samples] = k_
#     #         mat[-1, :self.n_samples] = 1.
#     #         mat[:self.n_samples, -1] = 1.
#     #
#     #     # ToDo: Implement partition scheme for inverting the matrix
#     #
#     #     matinv = np.linalg.inv(mat)
#     #
#     #     if self.n_samples != 0:
#     #         y_vec = np.concatenate([self.y, self.y_prime.flatten(), np.zeros(1)])
#     #     else:
#     #         y_vec = np.concatenate([self.y, self.y_prime.flatten()])
#     #
#     #     a_b = matinv.dot(y_vec)
#     #     self.alpha = a_b[:self.n_samples]
#     #
#     #     if self.n_samples != 0:
#     #         self.beta = a_b[self.n_samples:-1].reshape((self.dim, -1)).T
#     #         self.intercept = a_b[-1]
#     #     else:
#     #         self.beta = a_b[self.n_samples:].reshape((self.dim, -1)).T
#     #         self.intercept = 0.0
#     #
#     #     self.support_index_alpha = np.arange(0, self.n_samples, 1)
#     #     self.support_index_beta = []
#     #     for ii in range(self.dim):
#     #         self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1))
#
#     # def _fit_rls_b(self, C1=1.0, C2=1.0):
#     #
#     #     # mat  [M1+M2, M1+M2]
#     #     # M1 [N_samples]
#     #     # M2 [N_samples_prime*dim]
#     #
#     #     # [[B  C]
#     #     #  [E  D]]
#     #     # B [M1,M1] (k_xy)
#     #     # C [M1, M2] (k_dy)
#     #     # D [M2, M2] (k_dxdy)
#     #     # E [M2, M1] (k_dx)
#     #
#     #     # N ... atoms
#     #     y_ = np.concatenate([self.y, self.y_prime.flatten()])
#     #     mat = np.zeros([self.n_samples + self.n_samples_prime * self.dim, self.n_samples + self.n_samples_prime * self.dim])
#     #
#     #     b, c, e, d = self._create_mat(C1=C1, C2=C2)
#     #     b = np.linalg.inv(b + 1)
#     #     # [[t, u]
#     #     #  [w, v]]
#     #
#     #     v = np.linalg.inv(d-e.dot(b.dot(c)))
#     #     u = -b.dot(c.dot(v))
#     #     w = -v.dot(e.dot(b))
#     #     t = b-b.dot(c.dot(w))
#     #
#     #     mat[:self.n_samples, :self.n_samples] = t
#     #     mat[self.n_samples:, :self.n_samples] = w
#     #     mat[:self.n_samples, self.n_samples:] = u
#     #     mat[self.n_samples:, self.n_samples:] = v
#     #
#     #     a = mat.dot(y_)
#     #
#     #     self.alpha = a[0:self.n_samples].reshape(-1).T
#     #     self.beta = a[self.n_samples:].reshape((self.dim, -1)).T
#     #     self.intercept = sum(self.alpha)
#     #
#     #     self.support_index_alpha = np.arange(0, self.n_samples, 1)
#     #     self.support_index_beta = []
#     #     for ii in range(self.dim):
#     #         self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1))
#
#     def _fit_simple(self, C1=1.0, C2=1.0):
#         if self.n_samples_prime == 0:
#             # inequalities have to be func >= 0
#             constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:self.n_samples]-x_[self.n_samples:]))},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(-x_ + C1)})
#
#             def dual_func(x__):
#                 # x = [alpha, alpha_star]
#                 alpha = x__[:self.n_samples]
#                 alpha_star = x__[self.n_samples:]
#                 return -(-0.5 *((alpha-alpha_star).T.dot(self.kernel.kernel(self.x,self.x).dot(alpha-alpha_star))) \
#                         + self.epsilon * np.sum(alpha + alpha_star) + np.dot(self.y.T, (alpha - alpha_star)))
#
#             res = spmin.minimize(dual_func, np.zeros(self.n_samples*2), method='SLSQP', constraints=constrains)
#
#             self.alpha = res.x[:self.n_samples]-res.x[self.n_samples:]
#             self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]
#
#         elif self.n_samples == 0:
#             def dual_func(x__):
#                 beta = x__[:self.n_samples_prime*self.dim]
#                 beta_star = x__[self.n_samples_prime*self.dim:]
#                 return -(sum(((beta[:, ii]-beta_star[:, ii]).T.dot(
#                         self.kernel.kernel(self.x_prime, self.x_prime, nx=ii, ny=jj).dot(beta[:, jj]-beta_star[:,jj]))
#                         for ii, jj in zip(range(self.dim), range(self.dim))))) - self.epsilon_beta*np.sum(beta-beta_star)\
#                         + sum(np.dot(self.y_prime[:, ii], (beta[:,ii]-beta_star[:,ii])) for ii in range(self.dim))
#
#             constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:+self.n_samples_prime*self.dim]
#                               - x_[self.n_samples_prime*self.dim:]))},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(-x_ + C2)})
#             res = spmin.minimize(dual_func, np.zeros(self.n_samples * 2 + 2 * self.n_samples_prime * self.dim)
#                                  , method='SLSQP', constraints=constrains)
#             self.alpha = np.zeros(len(self.x))
#             self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10 ** -8]
#             self.beta = (res.x[:self.n_samples_prime * self.dim] - res.x[self.n_samples_prime * self.dim:])
#
#             self.beta = self.beta.reshape(-1, self.dim)
#             self.support_index_beta = []
#             for ii in range(self.dim):
#                 self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1, dtype=int)
#                                                [np.abs(self.beta[:, ii]) > 10 ** -8])
#
#         else:
#             def dual_func(x__):
#                 alpha = x__[:self.n_samples]
#                 alpha_star = x__[self.n_samples:2*self.n_samples]
#
#                 beta = x__[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
#                 beta_star = x__[2*self.n_samples+self.n_samples_prime*self.dim:]
#                 beta = beta.reshape(-1, self.dim)
#                 beta_star = beta_star.reshape(-1, self.dim)
#
#                 func = -0.5*((alpha-alpha_star).T.dot(self.kernel.kernel(self.x, self.x).dot(alpha-alpha_star))
#                        + sum((alpha-alpha_star).T.dot(self.kernel.kernel(
#                         self.x_prime, self.x, nx=ii) .dot(beta[:, ii] - beta_star[:, ii]))for ii in range(self.dim))
#                        + sum((beta[:,ii] - beta_star[:, ii]).T.dot(self.kernel.kernel(
#                         self.x, self.x_prime, ny=ii).dot(alpha-alpha_star)) for ii in range(self.dim))
#                        + sum(((beta[:,ii]-beta_star[:,ii]).T.dot(
#                         self.kernel.kernel(self.x_prime, self.x_prime, nx=ii, ny=jj).dot(beta[:, jj]-beta_star[:,jj]))
#                          for ii, jj in zip(range(self.dim), range(self.dim))))) \
#                        - self.epsilon*np.sum(alpha+alpha_star) + np.dot(self.y.T, (alpha-alpha_star)) \
#                        - self.epsilon_beta*np.sum(beta-beta_star) + sum(np.dot(self.y_prime[:, ii],
#                          (beta[:,ii]-beta_star[:,ii])) for ii in range(self.dim))
#                 return -func
#
#             constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:self.n_samples]
#                                                                            -x_[self.n_samples:2*self.n_samples]))},
#                           {'type': 'eq', 'fun': lambda x_: np.array(np.sum(
#                               x_[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
#                               - x_[2*self.n_samples+self.n_samples_prime*self.dim:]))},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(-x_[:2*self.n_samples] + C1)},
#                           {'type': 'ineq', 'fun': lambda x_: np.array(-x_[2*self. n_samples:] + C2)})
#
#
#             res = spmin.minimize(dual_func, np.zeros(self.n_samples*2+2*self.n_samples_prime*self.dim)
#                                  , method='SLSQP', constraints=constrains)
#             self.alpha = res.x[:self.n_samples] - res.x[self.n_samples:2*self.n_samples]
#             self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]
#             self.beta = (res.x[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
#                          - res.x[2*self.n_samples+self.n_samples_prime*self.dim:])
#
#             self.beta = self.beta.reshape(-1, self.dim)
#             self.support_index_beta = []
#             for ii in range(self.dim):
#                 self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1, dtype=int)
#                                                [np.abs(self.beta[:, ii]) > 10 ** -8])
#
#         # Todo calculation of intercept in higher dimensions --> input(x,y) output(f(x,y))
#         self.intercept = 0.0
#         # if alpha and alpha_star are in the region [0,C] then b = y-epsilon-<w,x>
#         # w in the simple case = (alpha-alpha_star)*x_i
#         # w in the advanced case = (alpha-alpha_star)*x_i+(beta-beta_star)*x_i'
#         self._is_fitted = True
#         b = self.y[self.support_index_alpha]-self.epsilon-self.predict(self.x[self.support_index_alpha])
#         self.intercept = np.sum(b, axis=0)/len(b)
#
#     def _create_mat(self, C1=None, C2=None):
#         # [[k, g]
#         #  [k_ j]]
#
#         k = np.zeros([self.n_samples, self.n_samples])
#         g = np.zeros([self.n_samples, self.n_samples_prime*self.dim])
#         k_ = np.zeros([self.n_samples_prime*self.dim, self.n_samples])
#         j = np.zeros([self.n_samples_prime*self.dim, self.n_samples_prime*self.dim])
#
#         if self.n_samples_prime != 0:
#             j = np.zeros([self.n_samples_prime * self.dim, self.n_samples_prime * self.dim])
#             for nx in range(0, self.dim):
#                 ind = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]
#                 mat = self.kernel.kernel(self.x_prime, self.x_prime, nx=nx, ny=nx)
#                 if C2 is not None:
#                     mat += np.eye(self.n_samples_prime) / C2
#                 j[ind[0]:ind[1], ind[0]:ind[1]] = mat
#
#         if self.n_samples != 0:
#             mat = self.kernel.kernel(self.x, self.x)
#             if C1 is not None:
#                 mat += np.eye(self.n_samples)/C1
#             k[:,:] = mat
#
#         if self.n_samples != 0 and self.n_samples_prime != 0:
#             for nx in range(-1, self.dim):
#                 for ny in range(nx + 1, self.dim):
#                     if nx == -1:
#                         ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
#                         ind2 = [0, self.n_samples]
#                     else:
#                         ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
#                         ind2 = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]
#
#                     k_[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny)
#                     g[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny).T
#
#         return k, g, k_, j

class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def kernel(self, x, y, nx=-1, ny=-1):
        exp_mat = np.exp(-self.gamma * (np.tile(np.sum(x**2, axis=1), (len(y), 1)).T +
            np.tile(np.sum(y**2, axis=1), (len(x), 1)) - 2*x.dot(y.T)))
        if nx == ny:
            if nx == -1:
                return exp_mat
            else:
                return -2.0 * self.gamma * exp_mat * (2.0 * self.gamma * np.subtract.outer(x[:,ny].T, y[:,ny])**2 - 1)
        elif nx == -1:
            return -2.0 * self.gamma * exp_mat * np.subtract.outer(x[:,ny].T, y[:,ny])
        elif ny == -1:
            return 2.0 * self.gamma * exp_mat * np.subtract.outer(x[:,nx].T, y[:,nx])
        else:
            return -4.0 * self.gamma**2 * exp_mat * np.subtract.outer(x[:,nx].T, y[:,nx]) * np.subtract.outer(x[:,ny].T, y[:,ny])
