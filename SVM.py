import numpy as np
import scipy.optimize._minimize as spmin


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
        self._is_fitted= False

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
            self._fit_irwls(C1=C1, C2=C2) #x, y, x_prime=x_prime, y_prime=y_prime,
            self._is_fitted = True
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

    def _fit_irwls(self, C1=1.0, C2=1.0, max_iter=10**4):
        def calc_weight(weight, error, constant):
            weight[error <= 0] = 0
            weight[error > 0] = 2 * constant / error[error > 0]
            return weight

        k, g, k_, j = self._create_mat()

        step = 0
        converged = False

        a = np.zeros(self.n_samples)
        a_star = np.zeros(self.n_samples)
        s = np.zeros(self.n_samples_prime*self.dim)
        s_star = np.zeros(self.n_samples_prime*self.dim)

        a[1::2] = C1
        s[1::2] = C2
        a_star[::2] = C1
        s_star[::2] = C2

        support_index_alpha = np.arange(0, self.n_samples, 1)
        support_index_beta = np.arange(0, self.n_samples_prime * self.dim, 1)

        if self.n_samples == 0:
            length_mat = self.n_samples + self.n_samples_prime * self.dim
            mat = j

            while not converged:
                index_s = np.logical_or(s > 0., s_star > 0.)
                self.support_index_beta = support_index_beta[index_s]

                idx = np.logical_and(np.tile(index_s, length_mat).reshape(length_mat, length_mat),
                                     np.tile(index_s, length_mat).reshape(length_mat, length_mat).T)

                mat_calc = mat[idx].reshape(len(self.support_index_beta), len(self.support_index_beta))

                s_ = s[self.support_index_beta]
                s_star_ = s_star[self.support_index_beta]
                d_s = np.linalg.inv(np.eye(len(self.support_index_beta)) * (s_ + s_star_))

                mat_calc += d_s
                y_ = self.y_prime.flatten()[self.support_index_beta] + (s_ - s_star_) / (s_ + s_star_) * self.epsilon_beta

                vec = np.linalg.inv(mat_calc).dot(y_)

                beta = np.zeros(self.n_samples_prime * self.dim)

                beta[self.support_index_beta] = vec

                d = mat[self.n_samples:, self.n_samples:].T.dot(beta) - self.y_prime.flatten() - self.epsilon_beta
                d_star = -mat[self.n_samples:, self.n_samples:].T.dot(beta) + self.y_prime.flatten() - self.epsilon_beta

                if step > 1:
                    if error_d < d.dot(d) and error_d_star < d_star.dot(d_star):
                        converged = True
                        print('converged')
                error_d = d.dot(d)
                error_d_star = d_star.dot(d_star)

                s = calc_weight(s, d, C2)
                s_star = calc_weight(s_star, d_star, C2)

                step += 1
                if step > max_iter:
                    break
            b = 0.
            alpha = np.zeros(self.n_samples)
            index_a = np.logical_or(a > 0., a_star > 0.)

        else:
            length_mat = self.n_samples + self.n_samples_prime * self.dim + 1
            mat = np.zeros([length_mat, length_mat])
            mat[:self.n_samples, -1] = 1.
            mat[-1, :self.n_samples] = 1.

            if self.n_samples_prime == 0:
                mat[:-1, :-1] = k
            else:
                mat[:self.n_samples, :self.n_samples] = k
                mat[:self.n_samples, self.n_samples:-1] = g
                mat[self.n_samples:-1, :self.n_samples] = k_
                mat[self.n_samples:-1, self.n_samples:-1] = j

            while not converged:
                index_a = np.logical_or(a > 0., a_star > 0.)
                self.support_index_alpha = support_index_alpha[index_a]

                index_s = np.logical_or(s > 0., s_star > 0.)
                self.support_index_beta = support_index_beta[index_s]

                index = np.concatenate([index_a, index_s, np.array([1])])
                idx = np.logical_and(np.tile(index, length_mat).reshape(length_mat,length_mat),
                                     np.tile(index, length_mat).reshape(length_mat, length_mat).T)

                mat_calc = mat[idx].reshape(len(self.support_index_alpha) + len(self.support_index_beta) + 1,
                                            len(self.support_index_alpha) + len(self.support_index_beta) + 1)

                a_ = a[self.support_index_alpha]
                a_star_ = a_star[self.support_index_alpha]
                s_ = s[self.support_index_beta]
                s_star_ = s_star[self.support_index_beta]

                d_a = np.linalg.inv(np.eye(len(self.support_index_alpha))*(a_ + a_star_))
                d_s = np.linalg.inv(np.eye(len(self.support_index_beta))*(s_ + s_star_))

                mat_calc[:len(self.support_index_alpha), :len(self.support_index_alpha)] += d_a
                mat_calc[len(self.support_index_alpha):-1, len(self.support_index_alpha):-1] += d_s

                y_ = np.concatenate([self.y[self.support_index_alpha] + (a_ - a_star_) / (a_ + a_star_) * self.epsilon,
                                     self.y_prime.flatten()[self.support_index_beta] + (s_ - s_star_) / (
                                         s_ + s_star_) * self.epsilon_beta, np.array([0])])

                vec = np.linalg.inv(mat_calc).dot(y_)

                alpha = np.zeros(self.n_samples)
                alpha[self.support_index_alpha] = vec[:len(self.support_index_alpha)]
                beta = np.zeros(self.n_samples_prime*self.dim)

                if self.n_samples != 0:
                    beta[self.support_index_beta] = vec[len(self.support_index_alpha):-1]
                    b = vec[-1]

                    e = mat[:self.n_samples, :self.n_samples].T.dot(alpha) + mat[self.n_samples:-1, :self.n_samples].T.dot(beta) \
                        + b - self.y - self.epsilon
                    e_star = self.y - mat[:self.n_samples, :self.n_samples].T.dot(alpha) \
                             - mat[self.n_samples:-1, :self.n_samples].T.dot(beta) - b - self.epsilon
                    d = mat[:self.n_samples, self.n_samples:-1].T.dot(alpha) \
                        + mat[self.n_samples:-1, self.n_samples:-1].T.dot(beta) - self.y_prime.flatten() - self.epsilon_beta
                    d_star = -mat[:self.n_samples, self.n_samples:-1].T.dot(alpha) \
                             - mat[self.n_samples:-1, self.n_samples:-1].T.dot(beta) + self.y_prime.flatten() - self.epsilon_beta
                if step > 1:

                    if self.n_samples != 0 and self.n_samples_prime != 0:
                        if error_e < e.dot(e) and error_e_star < e_star.dot(e_star) \
                                and error_d < d.dot(d) and error_d_star < d_star.dot(d_star):
                            converged = True
                            print('converged')
                    elif self.n_samples != 0:
                        if error_e < e.dot(e) and error_e_star < e_star.dot(e_star):
                            converged = True
                            print('converged')

                error_e = e.dot(e)
                error_e_star = e_star.dot(e_star)
                error_d = d.dot(d)
                error_d_star = d_star.dot(d_star)

                a = calc_weight(a, e, C1)
                a_star = calc_weight(a_star, e_star, C1)
                s = calc_weight(s, d, C2)
                s_star = calc_weight(s_star, d_star, C2)

                step += 1
                if step > max_iter:
                    break

        self.alpha = alpha
        self.beta = beta.reshape(-1, self.dim)
        self.intercept = b

        index_s = index_s.reshape(-1, self.dim)

        self.support_index_alpha = np.arange(0, self.n_samples, 1)[index_a]
        self.support_index_beta = []
        for ii in range(self.dim):
            self.support_index_beta.append(np.arange(0, self.n_samples_prime, 1)[index_s[:,ii]])

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
            mat[:self.n_samples, self.n_samples:-1] = k_
            mat[self.n_samples:-1, self.n_samples:-1] = j
            mat[:self.n_samples, :self.n_samples] = k
            mat[self.n_samples:-1, :self.n_samples] = g
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
        # Todo only derivative fit
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
            dual_func(np.zeros(self.n_samples*2+2*self.n_samples_prime*self.dim))
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
