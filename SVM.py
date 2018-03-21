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
        if y_prime_train is None:
            self.n_samples_prime = 0
            self.n_dim = len(x_train[0])
            self.y_prime_train = np.array([])
        else:
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
                return self._alpha[self._support_index_alpha].dot(
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
        g = np.zeros([self.n_samples, self.n_samples_prime*self.n_dim])
        k_prime = np.zeros([self.n_samples_prime*self.n_dim, self.n_samples])
        j = np.zeros([self.n_samples_prime*self.n_dim, self.n_samples_prime*self.n_dim])

        if self.n_samples_prime != 0:
            j = np.zeros([self.n_samples_prime * self.n_dim, self.n_samples_prime * self.n_dim])
            for nx in range(0, self.n_dim):
                ind = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]
                mat = self.kernel.kernel(self.x_prime_train, self.x_prime_train, nx=nx, ny=nx)
                if C2 is not None:
                    mat += np.eye(self.n_samples_prime) / C2
                j[ind[0]:ind[1], ind[0]:ind[1]] = mat

        if self.n_samples != 0:
            mat = self.kernel.kernel(self.x_train, self.x_train)
            if C1 is not None:
                mat += np.eye(self.n_samples)/C1
            k[:,:] = mat

        if self.n_samples != 0 and self.n_samples_prime != 0:
            for nx in range(-1, self.n_dim):
                for ny in range(nx + 1, self.n_dim):
                    if nx == -1:
                        ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
                        ind2 = [0, self.n_samples]
                    else:
                        ind1 = [ny * self.n_samples_prime, (ny + 1) * self.n_samples_prime]
                        ind2 = [nx * self.n_samples_prime, (nx + 1) * self.n_samples_prime]

                    k_prime[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel.kernel(self.x_prime_train, self.x_train, nx=nx, ny=ny)
                    g[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel.kernel(self.x_prime_train, self.x_train, nx=nx, ny=ny).T

        return k, g, k_prime, j


class IRWLS(SVM):
    # Todo lagrangian function

    def error_weight(self, error, constant):
        weight = np.minimum(np.maximum(0., constant/error), constant/self.error_cap)
        return weight

    def error_function(self, alpha, beta, b, k, g, k_prime, j):
        func_error = np.zeros(self.n_samples * 2)
        grad_error = np.zeros(self.n_samples_prime * self.n_dim * 2)
        if self.n_samples != 0:
            func_prediction = alpha[self._support_index_alpha].dot(k[self._support_index_alpha, :]) +\
                              beta[self._support_index_beta].dot(k_prime[self._support_index_beta, :]) + b
            func_error[:self.n_samples] = func_prediction - self.y_train - self.epsilon
            func_error[self.n_samples:] = -func_prediction + self.y_train - self.epsilon
        if self.n_samples_prime != 0:
            grad_prediction = alpha[self._support_index_alpha].dot(g[self._support_index_alpha, :]) +\
                              beta[self._support_index_beta].dot(j[self._support_index_beta, :])
            grad_error[:self.n_samples_prime*self.n_dim] = grad_prediction - self.y_prime_train-self.epsilon
            grad_error[self.n_samples_prime * self.n_dim:] = -grad_prediction + self.y_prime_train - self.epsilon
        return func_error, grad_error

    def lagrangian(self, alpha_, beta_, b_):
        pass

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, C1=1., C2=1., error_cap=10**-8, epsilon=0.1,
            max_iter=10**5, eps=10**-6):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self.y_prime_train = self.y_prime_train.flatten()
        self.epsilon = epsilon
        k, g, k_prime, j = self._create_mat()
        self.error_cap = error_cap

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

            target_vec = np.concatenate([self.y_train[self._support_index_alpha] + ((a_ - a_star_) / (a_ + a_star_)) * epsilon,
                                 self.y_prime_train.flatten()[self._support_index_beta] + ((s_ - s_star_) / (s_ + s_star_)) * epsilon,
                                 np.array([0])])

            weight = np.linalg.inv(calc_mat).dot(target_vec)

            alpha_s = np.zeros(self.n_samples)
            alpha_s[self._support_index_alpha] = weight[:n_support_alpha]
            beta_s = np.zeros(self.n_samples_prime * self.n_dim)
            beta_s[self._support_index_beta] = weight[n_support_alpha:-1]
            b_s = weight[-1]

            # dir_alpha = alpha - alpha_s
            # dir_beta = beta - beta_s
            # dir_b = b - b_s
            #
            # #todo eta calculation
            # f_error, g_error = self.error_function(alpha, beta, b, k, g, k_prime, j)
            # f_error_s, g_error_s = self.error_function(alpha_s, beta_s, b_s, k, g, k_prime, j)
            #
            # eta = 1.
            #
            # alpha += dir_alpha * eta
            # beta += dir_beta * eta
            # b += dir_b * eta
            alpha = alpha_s.copy()
            beta = beta_s.copy()
            b = b_s

            f_error, g_error = self.error_function(alpha, beta, b, k, g, k_prime, j)
            a = self.error_weight(f_error, C1)
            s = self.error_weight(g_error, C2)

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


class RLS(SVM):
    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, minimze_b=False, C1=1., C2=1.):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)

        k, g, k_prime, j = self._create_mat(C1=C1, C2=C2)

        if not minimze_b:
            mat_size = self.n_samples + self.n_samples_prime * self.n_dim + 1
            mat = np.zeros([mat_size, mat_size])
            mat[:self.n_samples, :self.n_samples] = k
            mat[:self.n_samples, self.n_samples:-1] = g
            mat[self.n_samples:-1, :self.n_samples] = k_prime
            mat[self.n_samples:-1, self.n_samples:-1] = j

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



# # Todo irwls fit d_a and d_s inversion
# # Todo own error Function, maybe own class for the different methods?

# def _fit_irwls(self, C1=1.0, C2=1.0, max_iter=10**2, error_cap=10**-4, eps=10**-8):
#
        # def calc_weight(error, constant):
        #     weight = np.zeros(error.shape)
        #     weight[(error < error_cap) & (error >= 0.)] = constant/error_cap
        #     weight[error >= error_cap] = constant / error[error >= error_cap]
        #     return weight
        #
        # def error_function(alpha_, beta_, b_):
        #     func_error = np.zeros(self.n_samples*2)
        #     grad_error = np.zeros(self.n_samples_prime*self.dim*2)
        #     if self.n_samples != 0 and self.n_samples_prime != 0:
        #         func_error[:self.n_samples] = alpha_[idx_alpha].dot(k[idx_alpha, :]) +\
        #                                       beta_[idx_beta].dot(k_prime[idx_beta, :]) + b_ - self.y - self.epsilon
        #         func_error[self.n_samples:] = -alpha_[idx_alpha].dot(k[idx_alpha, :]) - \
        #                                       beta_[idx_beta].dot(k_prime[idx_beta, :]) - b_ + self.y - self.epsilon
        #
        #         grad_error[self.n_samples_prime*self.dim:] = alpha_[idx_alpha].dot(g[idx_alpha, :]) \
        #                                                      + beta_[idx_beta].dot(j[idx_beta, :])\
        #                                                      - self.y_prime.flatten() - self.epsilon_beta
        #         grad_error[:self.n_samples_prime*self.dim] = -alpha_[idx_alpha].dot(g[idx_alpha, :]) \
        #                                                      - beta_[idx_beta].dot(j[idx_beta, :]) \
        #                                                      + self.y_prime.flatten() - self.epsilon_beta
        #     elif self.n_samples_prime == 0:
        #         func_error[:self.n_samples] = alpha_[idx_alpha].dot(k[idx_alpha, :]) + b_ - self.y - self.epsilon
        #         func_error[self.n_samples:] = - alpha_[idx_alpha].dot(k[idx_alpha, :]) - b_ + self.y - self.epsilon
        #
        #     return func_error, grad_error
        #
        # def lagrangian(alpha_, beta_, b_):
        #     func_error, grad_error = error_function(alpha_, beta_, b_)
        #     _k = k[:, idx_alpha]
        #     _k_prime = k_prime[:, idx_alpha]
        #     _g = g[:, idx_beta]
        #     _j = j[:, idx_beta]
        #
        #     return 1 / 2. * (alpha_[idx_alpha].T.dot(_k[idx_alpha, :].dot(alpha_[idx_alpha]))
        #                      + alpha_[idx_alpha].T.dot(_g[idx_alpha, :].dot(beta_[idx_beta]))
        #                      + beta_[idx_beta].T.dot(_k_prime[idx_beta, :].dot(alpha_[idx_alpha]))
        #                      + (beta_[idx_beta].T.dot(_j[idx_beta, :].dot(beta_[idx_beta])))) \
        #                      + C1 * np.sum(func_error[func_error > 0.]) + C2 * np.sum(grad_error[grad_error > 0.])
        #     # return  C1 * np.sum(func_error[func_error > 0.]) + C2 * np.sum(grad_error[grad_error > 0.])
        #
        #
        # k, g, k_prime, j = self._create_mat()
        #
        # a = np.zeros(self.n_samples * 2)
        # s = np.zeros(self.n_samples_prime*self.dim*2)
        #
        # a[1:self.n_samples:2] = C1
        # a[self.n_samples::2] = C1
        # s[1:self.n_samples_prime*self.dim:2] = C2
        # s[self.n_samples_prime*self.dim::2] = C2
        #
        # support_index_alpha = np.arange(0, self.n_samples, 1)
        # support_index_beta = np.arange(0, self.n_samples_prime * self.dim, 1)
        #
        # size_mat = self.n_samples + self.n_samples_prime*self.dim + 1
        # mat = np.zeros([size_mat, size_mat])
        #
        # mat[:self.n_samples, :self.n_samples] = k
        # mat[:self.n_samples, self.n_samples:-1] = g
        # mat[self.n_samples:-1, :self.n_samples] = k_prime
        # mat[self.n_samples:-1, self.n_samples:-1] = j
        # mat[:self.n_samples, -1] = 1.
        # mat[-1, :self.n_samples] = 1.
        # # mat[-1, :-1] = 1.
        # # mat[:-1, -1] = 1.
        #
        # step = 1
        # converged = False
        # alpha = np.zeros(self.n_samples)
        # beta = np.zeros(self.n_samples_prime * self.dim)
        # b = 0.
        # l_keep = []
        # alpha_keep = []
        # beta_keep = []
        # b_keep = []
        # f_error_keep = []
        # g_error_keep = []
        #
        # x_predict = np.linspace(-11 * np.pi, 11 * np.pi, 1000).reshape(-1, 1)
        # rho = 0.8
        # l_diff = []
        # while not converged:
        #
        #     index_a = np.logical_or(a[:self.n_samples] > 0., a[self.n_samples:] > 0.)
        #     index_s = np.logical_or(s[:self.n_samples_prime*self.dim] > 0., s[self.n_samples_prime*self.dim:] > 0.)
        #
        #     a_ = a[:self.n_samples][index_a]
        #     a_star_ = a[self.n_samples:][index_a]
        #     s_ = s[:self.n_samples_prime*self.dim][index_s]
        #     s_star_ = s[self.n_samples_prime*self.dim:][index_s]
        #
        #     idx_alpha = support_index_alpha[index_a]
        #     idx_beta = support_index_beta[index_s]
        #
        #     d_a = np.linalg.inv(np.eye(len(idx_alpha)) * (a_ + a_star_))  # *(e + e_star)[self.support_index_alpha]/C1
        #     d_s = np.linalg.inv(np.eye(len(idx_beta)) * (s_ + s_star_))  # (d + d_star)[self.support_index_beta]/C2
        #
        #     index = np.logical_and(np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
        #                            .reshape(size_mat, size_mat),
        #                            np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
        #                            .reshape(size_mat, size_mat).T)
        #
        #     calc_mat = mat[index].reshape(len(idx_alpha)+len(idx_beta) + 1, len(idx_alpha)+len(idx_beta) + 1)
        #
        #     if len(idx_alpha) == 0:
        #         # to avoid the singularity if just derivatives occour as support vectors
        #         calc_mat[-1, -1] = 1
        #         print('avoid singularity ' + str(step))
        #
        #     calc_mat[:len(idx_alpha), :len(idx_alpha)] += d_a
        #     calc_mat[len(idx_alpha):-1, len(idx_alpha):-1] += d_s
        #
        #     y_ = np.concatenate([self.y[idx_alpha] + ((a_ - a_star_) / (a_ + a_star_)) * self.epsilon,
        #                          self.y_prime.flatten()[idx_beta] + ((s_ - s_star_) / (s_ + s_star_)) * self.epsilon_beta,
        #                          np.array([0])])
        #     vec_ = np.linalg.inv(calc_mat).dot(y_)
        #
        #     alpha_s = np.zeros(self.n_samples)
        #     alpha_s[idx_alpha] = vec_[:len(idx_alpha)]
        #     beta_s = np.zeros(self.n_samples_prime*self.dim)
        #     beta_s[idx_beta] = vec_[len(idx_alpha):-1]
        #     b_s = vec_[-1]
        #
        #     dir_alpha = -alpha + alpha_s
        #     dir_beta = -beta + beta_s
        #     dir_b = b_s - b
        #     eta = 1.
        #     l_new = lagrangian(alpha + eta * dir_alpha, beta + eta * dir_beta, b + eta *dir_b)
        #     l_dif = lagrangian(alpha_s, beta_s, b_s) - lagrangian(alpha, beta, b)
        #     if step > 1:
        #         ii = 0
        #         while l_old <= l_new:
        #             # print(ii)
        #             eta = rho * eta
        #             l_new = lagrangian(alpha + eta * dir_alpha, beta + eta * dir_beta, b + eta *dir_b)
        #             ii += 1
        #             if ii > 10**3:
        #                 # converged = True
        #                 break
        #     print('-------------')
        #     print(eta)
        #     alpha += dir_alpha*eta
        #     beta += dir_beta*eta
        #     b += dir_b*eta
        #
        #     l_diff.append(l_dif)
        #
        #     f_error, g_error = error_function(alpha, beta, b)
        #     a = calc_weight(f_error, C1)
        #     s = calc_weight(g_error, C2)
        #
        #     # convergence criteria
        #
        #     if step > 1:
        #         l = lagrangian(alpha, beta, b)
        #
        #         if np.less(abs(alpha[idx_alpha] - self.alpha[idx_alpha]), eps).all() \
        #                 and np.less(abs(beta[idx_beta] - self.beta[idx_beta]), eps).all():
        #             if abs(b - self.intercept) < eps:
        #                 converged = True
        #                 print('parameter converged ' + str(step))
        #
        #         if abs(l - l_old) < eps:
        #             converged = True
        #             print('lagrangian converged step = ' + str(step) + ' num alpha = ' + str(len(idx_alpha))
        #                   + ' num beta = ' + str(len(idx_beta)))
        #
        #         if l <= l_old:
        #             idx_beta_new = idx_beta.reshape(-1, self.dim)
        #             self.support_index_alpha = idx_alpha.copy()
        #             self.support_index_beta = []
        #             for ii in range(self.dim):
        #                 self.support_index_beta.append(support_index_beta[idx_beta_new[:, ii]])
        #             self.alpha = alpha.copy()
        #             self.beta = beta.copy()
        #             self.intercept = b
        #             # converged = False
        #             # print('lagrangian converged step = ' + str(step)+ ' num alpha = '+ str(len(idx_alpha))+' num beta = '
        #             #       + str(len(idx_beta)))
        #         if step > 2:
        #             if abs(l_dif) < 10**-6:
        #                 converged = True
        #                 print('lagrangian diff converged step = '+ str(step) + ' num alpha = ' + str(len(idx_alpha))
        #                       + ' num beta = ' + str(len(idx_beta))+ ' lagrangian diff = ' + str(l_dif))
        #             if step < 10:
        #                 converged = False
        #     # converged = False
        #     l_old = lagrangian(alpha, beta, b)
        #     l_keep.append(l_old)
        #     alpha_keep.append(alpha)
        #     beta_keep.append(beta)
        #     b_keep.append(b)
        #     f_error_keep.append(np.sum(f_error[idx_alpha]))
        #     g_error_keep.append(np.sum(g_error[idx_beta]))
        #
        #     if step > max_iter:
        #         print('iteration is not converged ' + str(step))
        #         # print(abs(alpha - self.alpha))
        #         # print(abs(beta - self.beta))
        #         # print(abs(b - self.intercept))
        #         break
        #     step += 1
        #
        #     idx_beta = idx_beta.reshape(-1, self.dim)
        #     self.support_index_alpha = idx_alpha.copy()
        #     self.support_index_beta = []
        #     for ii in range(self.dim):
        #         self.support_index_beta.append(support_index_beta[idx_beta[:, ii]])
        #     self.alpha = alpha.copy()
        #     self.beta = beta.copy()
        #     self.intercept = b
        #
        # self.beta = self.beta.reshape(-1, self.dim)

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
