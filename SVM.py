import numpy as np
import scipy.optimize._minimize as spmin
import matplotlib.pyplot as plt


# Todo matrix elements computation by runtime to save storage

class SVM:
    def __init__(self, epsilon=0.1, epsilon_beta=0.1, kernel='rbf', gamma=0.1, method='rls'):
        self.x = None
        self.x_prime = None
        self.y = None
        self.y_prime = None

        self.epsilon = epsilon
        self.epsilon_beta = epsilon_beta

        self.intercept = None
        self.alpha = None
        self.support_index_alpha = None

        self.dim = None
        self.n_samples = None
        self.n_samples_prime = None

        self.beta = None
        self.support_index_beta = None

        if kernel == 'rbf':
            self.kernel = RBF(gamma=gamma)

        self.method = method
        self._is_fitted = False

        self.debug_mat = None

    def predict_derivative(self, x):
        if not self._is_fitted:
            raise ValueError('not fitted')
        if self.method == 'simple':
            raise ValueError('simple method has can not predict derivatives at the moment')
        ret_mat = np.zeros((len(x), self.dim))
        for ii in range(self.dim):
            ret_mat[:, ii] = self.alpha[self.support_index_alpha].dot(self.kernel.kernel(
                                self.x[self.support_index_alpha], x, nx=ii))\
                            + sum([self.beta[self.support_index_beta[jj], jj].dot(self.kernel.kernel(
                                self.x_prime[self.support_index_beta[jj]], x, nx=ii, ny=jj)) for jj in
                                 range(self.dim)])

    def predict(self, x):
        if not self._is_fitted:
            raise ValueError('not fitted')

        if self.n_samples_prime != 0:
            return self.alpha[self.support_index_alpha].dot(self.kernel.kernel(self.x[self.support_index_alpha], x)) \
                   + sum(self.beta[self.support_index_beta[ii], ii].dot(self.kernel.kernel(
                       self.x_prime[self.support_index_beta[ii]], x, ny=ii)) for ii in range(self.dim)) + self.intercept
        else:
            return self.alpha[self.support_index_alpha].dot(
                    self.kernel.kernel(self.x[self.support_index_alpha], x)) + self.intercept

    def fit(self, x, y, x_prime=None, y_prime=None, C1=1.0, C2=1.0, max_iter=10**4):
        self.x = x
        self.y = y
        self.dim = x.shape[1]

        if x_prime is None:
            self.x_prime = np.empty([0, self.dim])
        else:
            self.x_prime = x_prime

        if y is None:
            self.n_samples = 0
        else:
            self.n_samples = len(y)

        if y_prime is None:
            self.n_samples_prime = 0
            self.y_prime = np.empty(0)
        else:
            self.y_prime = y_prime
            self.n_samples_prime = len(y_prime)

        if self.method == 'irwls':
            y_debug = self._fit_irwls(C1=C1, C2=C2)
            self._is_fitted = True
            return y_debug
        elif self.method == 'rls':
            self._fit_rls(C1=C1, C2=C2)
            self._is_fitted = True
        elif self.method == 'rls_b':
            self._fit_rls_b(C1=C1, C2=C2)
            self._is_fitted = True
        elif self.method == 'simple':
            self._fit_simple(C1=C1, C2=C2)
            self._is_fitted = True
        else:
            raise NotImplementedError('Method is not implemented use irwls, rls or simple')

# Todo irwls fit convergence criteria, and test if epsilon = 0 and epsilon_beta = 0
# Todo irwls fit d_a and d_s inversion
# Todo own error Function, maybe own class for the different methods?

    def _fit_irwls(self, C1=1.0, C2=1.0, max_iter=10**4):

        def calc_weight(error, constant):
            weight = np.zeros(error.shape)
            # weight[error < 0.] = 0.
            weight[error == 0.] = constant
            weight[error > 0.] = constant / error[error > 0.]
            return weight

        def error_function(alpha_, beta_, b_):
            func_error = np.zeros(self.n_samples*2)
            grad_error = np.zeros(self.n_samples_prime*self.dim*2)
            func_error[::2] = k.T.dot(alpha_) + k_.T.dot(beta_) + b_ - self.y - self.epsilon
            func_error[1::2] = -k.T.dot(alpha_) - k_.T.dot(beta_) - b_ + self.y - self.epsilon
            grad_error[::2] = g.T.dot(alpha_) + j.T.dot(beta_) - self.y_prime.flatten() - self.epsilon_beta
            grad_error[1::2] = -g.T.dot(alpha_) + j.T.dot(beta_) + self.y_prime.flatten() - self.epsilon_beta
            return func_error, grad_error

        def lagrangian(alpha_, beta_, func_error, grad_error):
            return 1 / 2 * (alpha_.T.dot(k.dot(alpha_)) + alpha_.T.dot(g.dot(beta_)) + beta_.T.dot(k_.dot(alpha_))
                            + (beta_.T.dot(j.dot(beta_))) + C1 * np.sum(func_error[func_error > 0])
                            + C2 * np.sum(grad_error[grad_error > 0]))

        k, g, k_, j = self._create_mat()

        a = np.zeros(self.n_samples * 2)
        s = np.zeros(self.n_samples_prime*self.dim*2)

        a[1:self.n_samples:2] = C1
        a[self.n_samples::2] = C1
        s[1:self.n_samples_prime*self.dim:2] = C2
        s[self.n_samples_prime*self.dim::2] = C2

        support_index_alpha = np.arange(0, self.n_samples, 1)
        support_index_beta = np.arange(0, self.n_samples_prime * self.dim, 1)

        size_mat = self.n_samples + self.n_samples_prime*self.dim + 1
        mat = np.zeros([size_mat, size_mat])

        mat[:self.n_samples, :self.n_samples] = k
        mat[:self.n_samples, self.n_samples:-1] = g
        mat[self.n_samples:-1, :self.n_samples] = k_
        mat[self.n_samples:-1, self.n_samples:-1] = j
        mat[:self.n_samples, -1] = 1.
        mat[-1, :self.n_samples] = 1.

        step = 1
        converged = False
        alpha = np.zeros(self.n_samples)
        beta = np.zeros(self.n_samples_prime * self.dim)
        b = 0

        # x_predict = np.linspace(-8 * np.pi, 8 * np.pi, 300).reshape(-1, 1)
        debug_y = []

        eta = 1.

        while not converged:

            index_a = np.logical_or(a[:self.n_samples] > 0., a[self.n_samples:] > 0.)
            index_s = np.logical_or(s[:self.n_samples_prime*self.dim] > 0., s[self.n_samples_prime*self.dim:] > 0.)

            a_ = a[:self.n_samples][index_a]
            a_star_ = a[self.n_samples:][index_a]
            s_ = s[:self.n_samples_prime*self.dim][index_s]
            s_star_ = s[self.n_samples_prime*self.dim:][index_s]

            idx_alpha = support_index_alpha[index_a]
            idx_beta = support_index_beta[index_s]

            d_a = np.linalg.inv(np.eye(len(idx_alpha)) * (a_ + a_star_))  # *(e + e_star)[self.support_index_alpha]/C1
            d_s = np.linalg.inv(np.eye(len(idx_beta)) * (s_ + s_star_))  # (d + d_star)[self.support_index_beta]/C2

            index = np.logical_and(np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
                                   .reshape(size_mat, size_mat),
                                   np.tile(np.concatenate([index_a, index_s, np.array([1])]), size_mat)
                                   .reshape(size_mat, size_mat).T)

            calc_mat = mat[index].reshape(len(idx_alpha)+len(idx_beta) + 1, len(idx_alpha)+len(idx_beta) + 1)

            if len(idx_alpha) == 0:
                # to avoid the singularity if just derivatives occour as support vectors
                calc_mat[-1, -1] = 1
                print('avoid singularity ' + str(step))

            calc_mat[:len(idx_alpha), :len(idx_alpha)] += d_a
            calc_mat[len(idx_alpha):-1, len(idx_alpha):-1] += d_s

            y_ = np.concatenate([self.y[idx_alpha] + (a_ - a_star_) / (a_ + a_star_) * self.epsilon,
                                 self.y_prime.flatten()[idx_beta] + (s_ - s_star_) / (s_ + s_star_) * self.epsilon_beta,
                                 np.array([0])])
            # y_ = np.concatenate([self.y[idx_alpha] + a_add_y[idx_alpha], self.y_prime.flatten()[idx_beta] + s_add_y[idx_beta], np.array([0])])

            vec_ = np.linalg.inv(calc_mat).dot(y_)

            alpha_s = np.zeros(self.n_samples)
            alpha_s[idx_alpha] = vec_[:len(idx_alpha)]
            beta_s = np.zeros(self.n_samples_prime*self.dim)
            beta_s[idx_beta] = vec_[len(idx_alpha):-1]
            b_s = vec_[-1]

            alpha += (-alpha+alpha_s)*eta
            beta += (-beta+beta_s)*eta
            b += (-b+b_s)*eta

            f_error, g_error = error_function(alpha, beta, b)
            f_error_s, g_error_s = error_function(alpha_s, beta_s, b_s)
            u_s_f = calc_weight(np.zeros(self.n_samples), f_error_s)
            u_s_g = calc_weight()

            if lagrangian(alpha, beta, f_error, g_error) > lagrangian(alpha_s, beta_s, f_error_s, g_error_s):
                alpha = alpha_s
                beta = beta_s
                b = b_s

            # alpha += (-alpha + alpha_s) * eta
            # beta += (-beta + beta_s) * eta
            # b += (-b + b_s) * eta

            a[:self.n_samples] = calc_weight(f_error[::2], C1)
            a[self.n_samples:] = calc_weight(f_error[1::2], C1)
            s[:self.n_samples_prime*self.dim] = calc_weight(g_error[::2], C2)
            s[self.n_samples_prime*self.dim:] = calc_weight(g_error[1::2], C2)

            if step > 1:

                if np.less(abs(alpha - self.alpha), 10**-4).all() and np.less(abs(beta - self.beta), 10**-4).all() and abs(b - self.intercept) < 10**-4:
                    converged = True
                    print('converged ' + str(step))
            idx_beta = idx_beta.reshape(-1, self.dim)

            self.support_index_alpha = support_index_alpha[idx_alpha]
            self.support_index_beta = []
            for ii in range(self.dim):
                self.support_index_beta.append(support_index_beta[idx_beta[:, ii]])

            # if step < 20:
            #     # if (step % 100) == 0:
            #     debug_y.append(self.predict(x_predict))

            if step >= max_iter:
                print('iteration is not converged ' + str(step))
                print(abs(alpha - self.alpha))
                print(abs(beta - self.beta))
                print(abs(b - self.intercept))
                break
            step += 1

            self.alpha = alpha
            self.beta = beta
            self.intercept = b

        self.beta = self.beta.reshape(-1, self.dim)
        return debug_y

    def _fit_rls(self, C1=1.0, C2=1.0):

        k, g, k_, j = self._create_mat(C1=C1, C2=C2)

        if self.n_samples == 0:
            # mat = np.zeros([self.n_samples_prime * self.dim, self.n_samples_prime * self.dim])
            mat = j
        elif self.n_samples_prime == 0:
            mat = np.zeros([self.n_samples + 1, self.n_samples + 1])
            mat[:-1, :-1] = k
            mat[-1, :self.n_samples] = 1.
            mat[:self.n_samples, -1] = 1.
        else:
            mat = np.zeros([self.n_samples + self.n_samples_prime * self.dim + 1,
                            self.n_samples + self.n_samples_prime * self.dim + 1])
            mat[:self.n_samples, self.n_samples:-1] = g
            mat[self.n_samples:-1, self.n_samples:-1] = j
            mat[:self.n_samples, :self.n_samples] = k
            mat[self.n_samples:-1, :self.n_samples] = k_
            mat[-1, :self.n_samples] = 1.
            mat[:self.n_samples, -1] = 1.

        # ToDo: Implement partition scheme for inverting the matrix

        matinv = np.linalg.inv(mat)

        if self.n_samples != 0:
            y_vec = np.concatenate([self.y, self.y_prime.flatten(), np.zeros(1)])
        else:
            y_vec = np.concatenate([self.y, self.y_prime.flatten()])

        a_b = matinv.dot(y_vec)
        self.alpha = a_b[:self.n_samples]

        if self.n_samples != 0:
            self.beta = a_b[self.n_samples:-1].reshape((self.dim, -1)).T
            self.intercept = a_b[-1]
        else:
            self.beta = a_b[self.n_samples:].reshape((self.dim, -1)).T
            self.intercept = 0.0

        self.support_index_alpha = np.arange(0, self.n_samples, 1)
        self.support_index_beta = []
        for ii in range(self.dim):
            self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

    def _fit_rls_b(self, C1=1.0, C2=1.0):

        # mat  [M1+M2, M1+M2]
        # M1 [N_samples]
        # M2 [N_samples_prime*dim]

        # [[B  C]
        #  [E  D]]
        # B [M1,M1] (k_xy)
        # C [M1, M2] (k_dy)
        # D [M2, M2] (k_dxdy)
        # E [M2, M1] (k_dx)

        # N ... atoms
        y_ = np.concatenate([self.y, self.y_prime.flatten()])
        mat = np.zeros([self.n_samples + self.n_samples_prime * self.dim, self.n_samples + self.n_samples_prime * self.dim])

        b, c, e, d = self._create_mat(C1=C1, C2=C2)
        b = np.linalg.inv(b + 1)
        # [[t, u]
        #  [w, v]]

        v = np.linalg.inv(d-e.dot(b.dot(c)))
        u = -b.dot(c.dot(v))
        w = -v.dot(e.dot(b))
        t = b-b.dot(c.dot(w))

        mat[:self.n_samples, :self.n_samples] = t
        mat[self.n_samples:, :self.n_samples] = w
        mat[:self.n_samples, self.n_samples:] = u
        mat[self.n_samples:, self.n_samples:] = v

        a = mat.dot(y_)

        self.alpha = a[0:self.n_samples].reshape(-1).T
        self.beta = a[self.n_samples:].reshape((self.dim, -1)).T
        self.intercept = sum(self.alpha)

        self.support_index_alpha = np.arange(0, self.n_samples, 1)
        self.support_index_beta = []
        for ii in range(self.dim):
            self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1))

    def _fit_simple(self, C1=1.0, C2=1.0):
        if self.n_samples_prime == 0:
            # inequalities have to be func >= 0
            constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:self.n_samples]-x_[self.n_samples:]))},
                          {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
                          {'type': 'ineq', 'fun': lambda x_: np.array(-x_ + C1)})

            def dual_func(x__):
                # x = [alpha, alpha_star]
                alpha = x__[:self.n_samples]
                alpha_star = x__[self.n_samples:]
                return -(-0.5 *((alpha-alpha_star).T.dot(self.kernel.kernel(self.x,self.x).dot(alpha-alpha_star))) \
                        + self.epsilon * np.sum(alpha + alpha_star) + np.dot(self.y.T, (alpha - alpha_star)))

            res = spmin.minimize(dual_func, np.zeros(self.n_samples*2), method='SLSQP', constraints=constrains)

            self.alpha = res.x[:self.n_samples]-res.x[self.n_samples:]
            self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]

        elif self.n_samples == 0:
            def dual_func(x__):
                beta = x__[:self.n_samples_prime*self.dim]
                beta_star = x__[self.n_samples_prime*self.dim:]
                return -(sum(((beta[:, ii]-beta_star[:, ii]).T.dot(
                        self.kernel.kernel(self.x_prime, self.x_prime, nx=ii, ny=jj).dot(beta[:, jj]-beta_star[:,jj]))
                        for ii, jj in zip(range(self.dim), range(self.dim))))) - self.epsilon_beta*np.sum(beta-beta_star)\
                        + sum(np.dot(self.y_prime[:, ii], (beta[:,ii]-beta_star[:,ii])) for ii in range(self.dim))

            constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:+self.n_samples_prime*self.dim]
                              - x_[self.n_samples_prime*self.dim:]))},
                          {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
                          {'type': 'ineq', 'fun': lambda x_: np.array(-x_ + C2)})
            res = spmin.minimize(dual_func, np.zeros(self.n_samples * 2 + 2 * self.n_samples_prime * self.dim)
                                 , method='SLSQP', constraints=constrains)
            self.alpha = np.zeros(len(self.x))
            self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10 ** -8]
            self.beta = (res.x[:self.n_samples_prime * self.dim] - res.x[self.n_samples_prime * self.dim:])

            self.beta = self.beta.reshape(-1, self.dim)
            self.support_index_beta = []
            for ii in range(self.dim):
                self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1, dtype=int)
                                               [np.abs(self.beta[:, ii]) > 10 ** -8])

        else:
            def dual_func(x__):
                alpha = x__[:self.n_samples]
                alpha_star = x__[self.n_samples:2*self.n_samples]

                beta = x__[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
                beta_star = x__[2*self.n_samples+self.n_samples_prime*self.dim:]
                beta = beta.reshape(-1, self.dim)
                beta_star = beta_star.reshape(-1, self.dim)

                func = -0.5*((alpha-alpha_star).T.dot(self.kernel.kernel(self.x, self.x).dot(alpha-alpha_star))
                       + sum((alpha-alpha_star).T.dot(self.kernel.kernel(
                        self.x_prime, self.x, nx=ii) .dot(beta[:, ii] - beta_star[:, ii]))for ii in range(self.dim))
                       + sum((beta[:,ii] - beta_star[:, ii]).T.dot(self.kernel.kernel(
                        self.x, self.x_prime, ny=ii).dot(alpha-alpha_star)) for ii in range(self.dim))
                       + sum(((beta[:,ii]-beta_star[:,ii]).T.dot(
                        self.kernel.kernel(self.x_prime, self.x_prime, nx=ii, ny=jj).dot(beta[:, jj]-beta_star[:,jj]))
                         for ii, jj in zip(range(self.dim), range(self.dim))))) \
                       - self.epsilon*np.sum(alpha+alpha_star) + np.dot(self.y.T, (alpha-alpha_star)) \
                       - self.epsilon_beta*np.sum(beta-beta_star) + sum(np.dot(self.y_prime[:, ii],
                         (beta[:,ii]-beta_star[:,ii])) for ii in range(self.dim))
                return -func

            constrains = ({'type': 'eq', 'fun': lambda x_: np.array(np.sum(x_[:self.n_samples]
                                                                           -x_[self.n_samples:2*self.n_samples]))},
                          {'type': 'eq', 'fun': lambda x_: np.array(np.sum(
                              x_[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
                              - x_[2*self.n_samples+self.n_samples_prime*self.dim:]))},
                          {'type': 'ineq', 'fun': lambda x_: np.array(x_)},
                          {'type': 'ineq', 'fun': lambda x_: np.array(-x_[:2*self.n_samples] + C1)},
                          {'type': 'ineq', 'fun': lambda x_: np.array(-x_[2*self. n_samples:] + C2)})


            res = spmin.minimize(dual_func, np.zeros(self.n_samples*2+2*self.n_samples_prime*self.dim)
                                 , method='SLSQP', constraints=constrains)
            self.alpha = res.x[:self.n_samples] - res.x[self.n_samples:2*self.n_samples]
            self.support_index_alpha = np.arange(0, self.n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]
            self.beta = (res.x[2*self.n_samples:2*self.n_samples+self.n_samples_prime*self.dim]
                         - res.x[2*self.n_samples+self.n_samples_prime*self.dim:])

            self.beta = self.beta.reshape(-1, self.dim)
            self.support_index_beta = []
            for ii in range(self.dim):
                self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1, dtype=int)
                                               [np.abs(self.beta[:, ii]) > 10 ** -8])

        # Todo calculation of intercept in higher dimensions --> input(x,y) output(f(x,y))
        self.intercept = 0.0
        # if alpha and alpha_star are in the region [0,C] then b = y-epsilon-<w,x>
        # w in the simple case = (alpha-alpha_star)*x_i
        # w in the advanced case = (alpha-alpha_star)*x_i+(beta-beta_star)*x_i'
        self._is_fitted = True
        b = self.y[self.support_index_alpha]-self.epsilon-self.predict(self.x[self.support_index_alpha])
        self.intercept = np.sum(b, axis=0)/len(b)

    def _create_mat(self, C1=None, C2=None):
        # [[k, g]
        #  [k_ j]]

        k = np.zeros([self.n_samples, self.n_samples])
        g = np.zeros([self.n_samples, self.n_samples_prime*self.dim])
        k_ = np.zeros([self.n_samples_prime*self.dim, self.n_samples])
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

                    k_[ind1[0]:ind1[1], ind2[0]:ind2[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny)
                    g[ind2[0]:ind2[1], ind1[0]:ind1[1]] = self.kernel.kernel(self.x_prime, self.x, nx=nx, ny=ny).T

        return k, g, k_, j


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
