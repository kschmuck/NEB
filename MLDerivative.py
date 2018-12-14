import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, cholesky
import copy as cp
import warnings
import scipy.optimize as sp_opt

np.set_printoptions(linewidth=320)
debug_flag = False
# Todo check if all methods work with only function values, only derivatives, or both


class ML:
    """ Parent class for machine learning algorithms
    :param kernel is the mapping of the points to the feature space, kernel returns the pair distances of given points
                    and derivatives
    """
    def __init__(self, kernel, reg_value, reg_derivative):
        self.x_train = None
        self.x_prime_train = None
        self.y_train = None
        self.y_prime_train = None
        self._y_target = None

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
            self._y_target = self.y_train
        else:
            self.n_samples_prime, self.n_dim = y_prime_train.shape
            self.y_prime_train = self.y_prime_train.flatten('F')
            self._y_target = np.concatenate([self.y_train, self.y_prime_train])

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


class SMO(ML):
    # TODO http://jonchar.net/notebooks/SVM/#Dual-form
    def __init__(self, kernel, C1=1., C2=1., epsilon=1e-4, epsilon_prime=1e-4):
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        self._mat = None
        self._mat_size = None

        self._weight = None
        self._x_vec = None
        self._error = 0

        self._obj = []
        self.tol = None
        self.alpha_tol = None
        ML.__init__(self, kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, alpha_tol=1e-3, tol=1e-3):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)
        self._mat_size = self.n_samples + self.n_samples_prime*self.n_dim + 1
        self.alpha_tol = alpha_tol
        self.tol = tol
        self._x_vec = self.x_train
        # self._x_vec = np.concatenate([self.x_train, self.x_prime_train])
        self._weight = np.zeros(self._mat_size)
        self._mat = np.zeros([self._mat_size, self._mat_size])
        self._mat[-1, :self.n_samples] = 1
        self._mat[:self.n_samples, -1] = 1
        self._mat[:-1, :-1] = create_mat(self.kernel, self.x_train, self.x_train, x1_prime=self.x_prime_train,
                                         x2_prime=self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim)
        self._error = self._y_target - self._mat[:-1, :].dot(self._weight)

        self.train()
        self._is_fitted = True

    def take_step(self, i1, i2, eps=1e-5, C=1e-5):

        # Skip if chosen alphas are the same
        if i1 == i2:
            return 0

        alph1 = self._weight[i1]
        alph2 = self._weight[i2]
        y1 = self._y_target[i1]
        y2 = self._y_target[i2]

        E1 = self._error[i1]
        E2 = self._error[i2]
        s = y1 * y2

        # Compute L & H, the bounds on new possible alpha values
        if (y1 != y2):
            L = max(0, alph2 - alph1)
            H = min(self._reg_value, self._reg_value + alph2 - alph1)
        elif (y1 == y2):
            L = max(0, alph1 + alph2 - self._reg_value)
            H = min(self._reg_value, alph1 + alph2)
        if (L == H):
            return 0

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self._x_vec[i1], self._x_vec[i1])
        k12 = self.kernel(self._x_vec[i1], self._x_vec[i2])
        k22 = self.kernel(self._x_vec[i2], self._x_vec[i2])
        eta = 2 * k12 - k11 - k22

        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            a2 = alph2 - y2 * (E1 - E2) / eta
            # Clip a2 based on bounds L & H
            if L < a2 < H:
                a2 = a2
            elif (a2 <= L):
                a2 = L
            elif (a2 >= H):
                a2 = H

        # If eta is non-negative, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self._weight.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = self.objective_function(alphas_adj, self._y_vec, self.kernel, self._x_vec)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = self.objective_function(alphas_adj, self._y_vec, self.kernel, self._x_vec)
            if Lobj > (Hobj + eps):
                a2 = L
            elif Lobj < (Hobj - eps):
                a2 = H
            else:
                a2 = alph2

        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self._reg_value - 1e-8):
            a2 = self._reg_value

        # If examples can't be optimized within epsilon (eps), skip this pair
        if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
            return 0

        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph2 - a2)

        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

        # Set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 and a1 < C:
            b_new = b1
        elif 0 < a2 and a2 < C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas & threshold
        self._weight[i1] = a1
        self._weight[i2] = a2

        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self._reg_value:
                self._error[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(self.n_samples) if (n != i1 and n != i2)]
        self._error[non_opt] = self._error[non_opt] + \
                                y1 * (a1 - alph1) * self.kernel(self._x_vec[i1], self._x_vec[non_opt]) + \
                                y2 * (a2 - alph2) * self.kernel(self._x_vec[i2], self._x_vec[non_opt]) + self._intercept - b_new

        # Update model threshold
        self._intercept = b_new

        return 1

    def examine_example(self, i2):

        y2 = self._y_target[i2]
        alph2 = self._weight[i2]
        E2 = self._error[i2]
        r2 = E2 * y2

        # Proceed if error is within specified tolerance (tol)
        if ((r2 < -self.tol and alph2 < self._reg_value) or (r2 > self.tol and alph2 > 0)):

            if len(self._weight[(self._reg_value!= 0) & (self._weight != self._reg_value)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                if self._error[i2] > 0:
                    i1 = np.argmin(self._error)
                elif self._error[i2] <= 0:
                    i1 = np.argmax(self._error)
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1

            # Loop through non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((self._weight!= 0) & (self._weight!= self._reg_value))[0],
                              np.random.choice(np.arange(self.n_samples))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1

            # loop through all alphas, starting at a random point
            for i1 in np.roll(np.arange(self.n_samples), np.random.choice(np.arange(self.n_samples))):
                step_result, model = self.take_step(i1, i2, model)
                if step_result:
                    return 1

        return 0

    def train(self):

        numChanged = 0
        examineAll = 1

        while (numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(self._weight.shape[0]):
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self._weight)
                        self._obj.append(obj_result)
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self._weight != 0) & (self._weight != self._reg_value))[0]:
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self._weight)
                        self._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

    def objective_function(self, weight):
        return weight.dot(self._mat.dot(weight)) + self._reg_value*sum(self._error)


class TSVR(ML):
    # TODO speed up
    """ implementation of the twin support vector machine
        See [Chen et al., 20014
        (https://link.springer.com/article/10.1007/s00500-014-1342-5)
        ([pdf](https://link.springer.com/content/pdf/10.1007%2Fs00500-014-1342-5.pdf))
    """
    def __init__(self, kernel, C1=1., C2=1., epsilon=1e-4, epsilon_prime=1e-4):
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        self._mat = None
        self._mat_size = None
        self._y_vec = None

        ML.__init__(self, kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None):
        self._fit(x_train, y_train, x_prime_train=x_prime_train, y_prime_train=y_prime_train)

        self._mat_size = self.n_dim * self.n_samples_prime + self.n_samples + 1
        self._mat = np.zeros([self._mat_size, self._mat_size-1])

        self._mat[:-1, :] = create_mat(self.kernel, self.x_train, self.x_train, x1_prime=self.x_prime_train,
                                       x2_prime=self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim,
                                       eval_gradient=False)
        self._mat[-1, :self.n_samples] = 1.

        weight_upper = self._solve_dual_BFGS(boundary_flag='upper')
        weight_lower = self._solve_dual_BFGS(boundary_flag='lower')

        # mat = create_mat(self.kernel, self.x_train, self.x_train, x1_prime=self.x_prime_train,
        #                                x2_prime=self.x_prime_train, dx_max=self.n_dim, dy_max=self.n_dim,
        #                                eval_gradient=False)
        # e = np.ones([self._mat_size - 1, 1])
        #
        # g_mat = np.concatenate([mat, e], axis=1)
        # i_mat = np.eye(self._mat_size)
        # i_mat[:self.n_samples, :self.n_samples] *= self._reg_value
        # i_mat[self.n_samples:-1, self.n_samples:-1] *= self._reg_derivative
        # inv_mat = np.linalg.inv(g_mat.T.dot(g_mat) + i_mat).dot(g_mat.T)
        # vec = np.zeros(self._mat_size-1)
        # vec[:self.n_samples] = self.epsilon
        # vec[self.n_samples:] = self.epsilon_prime
        # import time
        # print('start_fitting')
        # t = time.clock()
        # weight_upper = self.solve_dual(inv_mat, g_mat, vec)
        # print('end fitting %f') %(time.clock()-t)
        # print('start_fitting')
        # t = time.clock()
        # weight_lower = self.solve_dual(inv_mat, g_mat, -vec)
        # print('end fitting %f') % (time.clock() - t)

        # # F1 = self._y_target-vec
        # # F2 = self._y_target + vec
        # # H = -g_mat.dot(inv_mat)
        # # K1 = F1.T.dot(H)-F1.T
        # # K2 = -F2.T.dot(H)+F2.T
        # # H = (H+H.T)/2.
        # # G = np.ones([self._mat_size-1, self._mat_size-1])
        # # reg_vec = np.empty(self._mat_size-1)
        # # reg_vec[:self.n_samples] = self._reg_value
        # # reg_vec[self.n_samples:] = self._reg_derivative
        # # A = np.zeros([self._mat_size-1, self._mat_size-1])
        # # b =np.zeros(self._mat_size-1)
        # #
        # # weight_upper = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(K1), cvxopt.matrix(G), cvxopt.matrix(reg_vec))
        # #
        # # weight_lower = cvxopt.solvers.qp(H, K2)
        #
        # #
        # # # for prediction  weight = (weight_upper+weight_lower)/2
        weight = (weight_upper + weight_lower)/2.
        if debug_flag:
            self.weight_upper = weight_upper
            self.weight_lower = weight_lower
            self.weight = weight

        self._alpha = weight[:self.n_samples]
        idx_alpha = self._alpha != 0

        self._support_index_alpha = np.arange(0, self.n_samples)[idx_alpha]

        self._beta = weight[self.n_samples:-1]
        self._support_index_beta = 0
        idx_beta = self._beta != 0
        beta_idx = idx_beta.reshape(self.n_dim, -1).T

        if debug_flag:
            print(sum(idx_alpha))
            print(sum(idx_beta))
        self._support_index_beta = []
        self._beta = self._beta.reshape(self.n_dim, -1).T
        for ii in range(self.n_dim):
            self._support_index_beta.append(np.arange(0, self.n_samples_prime)[beta_idx[:, ii]])

        self._intercept = weight[-1]
        self._is_fitted = True

    def solve_dual(self, inv_mat, g_mat, vec):
        # TODO does not work properly
        sign = np.sign(vec[0])
        target_vec = self._y_target + vec

        def func(dummy_weight):
            term_a = 0.5*dummy_weight.T.dot(g_mat.dot(inv_mat).dot(dummy_weight))
            term_b = sign*(target_vec.dot(g_mat.dot(inv_mat))-target_vec).T.dot(dummy_weight)
            return -(term_a + term_b)

        cons = ({'type': 'ineq', 'fun': lambda x: -x[:self.n_samples] + self._reg_value},
                {'type': 'ineq', 'fun': lambda x: x},
                {'type': 'ineq', 'fun': lambda x: -x[self.n_samples:] + self._reg_derivative})
        d_weight = np.zeros(self._mat_size-1)
        res = sp_opt.minimize(func, d_weight, constraints=cons)
        d_weight = res.x
        weight = inv_mat.dot(target_vec + sign*d_weight)
        return weight

    def _solve_dual_BFGS(self, boundary_flag=None, imax=500, mu=1e0):
        reg_val = np.zeros(self._mat_size - 1)
        reg_val[:self.n_samples] = self._reg_value
        reg_val[self.n_samples:] = self._reg_derivative

        def max_func(x):
            idx = x <= 0
            x[idx] = 0.
            return x

        def step_func(x):
            idx = x >= 0
            x = np.zeros_like(x)
            x[idx] = 1.
            return x

        def function2min(_beta):
            return -mu*E1.T.dot(_beta) + 0.5 * (np.linalg.norm(max_func(I_mat.dot(_beta) - reg_val)) ** 2 +
                                                   np.linalg.norm(max_func(-I_mat.dot(_beta) - reg_val)) ** 2 +
                                                   np.linalg.norm(max_func(L_T.dot(_beta) - S2)) ** 2 +
                                                   np.linalg.norm(max_func(L_T.dot(_beta) - S1)) ** 2)

            # return -mu * E1.T.dot(_beta) + 0.5 * (np.linalg.norm(max_func(I_mat.dot(_beta) - reg_val)) ** 2 +
            #                                   np.linalg.norm(max_func(-I_mat.dot(_beta) - reg_val)) ** 2 +
            #                                   np.linalg.norm(max_func(L_T[:-1, :].dot(_beta) - S2[:-1])) ** 2 +
            #                                   np.linalg.norm(max_func(L_T[:-1, :].dot(_beta) - S1[:-1])) ** 2)

        def derivative(_beta):
            return -mu*E1 + I_mat.T.dot(max_func(I_mat.dot(_beta) - reg_val))\
                   - I_mat.T.dot(max_func(-I_mat.dot(_beta) - reg_val)) + L_T.T.dot(max_func(L_T.dot(_beta) - S2))\
                   - L_T.T.dot(max_func(-L_T.dot(_beta) - S1))
            # return -mu*E1 + I_mat.T.dot(max_func(I_mat.dot(_beta) - reg_val))\
            #        - I_mat.T.dot(max_func(-I_mat.dot(_beta) - reg_val)) + L_T[:-1, :].T.dot(max_func(L_T[:-1, :].dot(_beta) - S2[:-1]))\
            #        - L_T[:-1, :].T.dot(max_func(-L_T[:-1, :].dot(_beta) - S1[:-1]))

        def hessian(_beta):
            I_beta = I_mat.dot(_beta)
            L_beta = L_T.dot(_beta)
            e = np.ones(self._mat_size-1)
            return I_mat.T.dot(np.diag(step_func(I_beta-e)+step_func(-I_beta-e))).dot(I_mat) \
                    + L_T.T.dot(np.diag(step_func(L_beta-S2)+step_func(-L_beta-S1))).dot(L_T)
            # I_beta = I_mat.dot(_beta)
            # L_beta = L_T[:-1, :].dot(_beta)
            # e = np.ones(self._mat_size - 1)
            # return I_mat.T.dot(np.diag(step_func(I_beta - e) + step_func(-I_beta - e))).dot(I_mat) \
            #    + L_T[:-1, :].T.dot(np.diag(step_func(L_beta - S2[:-1]) + step_func(-L_beta - S1[:-1]))).dot(L_T[:-1, :])

        if self.x_prime_train is None:
            if boundary_flag == 'upper':
                y = self.y_train + self.epsilon
            elif boundary_flag == 'lower':
                y = self.y_train - self.epsilon
            else:
                raise ValueError('no boundary given ')

            print('none')
            E1 = np.zeros([2 * self.n_samples])
            E1[self.n_samples:] = y
            I_1 = np.eye(self.n_samples)
            I_mat = np.concatenate([I_1, I_1]).T
            L_T = np.zeros([self.n_samples * 2 + 1, self.n_samples * 2])
            L_T[:self.n_samples, :self.n_samples] = I_1
            L_T[self.n_samples:, self.n_samples:] = self._mat
            S1 = np.ones(self.n_samples * 2 + 1)
            S1[:self.n_samples] = 0.
            S2 = np.ones(self.n_samples * 2 + 1)
            S2[:self.n_samples] = 0

            beta_init = np.ones(2*self.n_samples)
            beta = sp_opt.fmin_bfgs(function2min, beta_init, derivative, maxiter=imax)

            p = 1/mu*max_func(L_T.dot(beta) - S2)[self.n_samples:]
            q = 1/mu*max_func(-L_T.dot(beta) - S1)[self.n_samples:]
        else:

            y = self._y_target
            if boundary_flag == 'upper':
                y[:self.n_samples] += self.epsilon
                y[self.n_samples:] += self.epsilon_prime
            elif boundary_flag == 'lower':
                y[:self.n_samples] -= self.epsilon
                y[self.n_samples:] -= self.epsilon_prime
            else:
                raise ValueError('no boundary given ')

            E1 = np.zeros((self._mat_size-1) * 2)
            E1[self._mat_size-1:] = y
            I_1 = np.eye(self._mat_size-1)
            I_mat = np.concatenate([I_1, I_1]).T
            L_T = np.zeros([(self._mat_size - 1) * 2 + 1, (self._mat_size - 1) * 2])
            L_T[:self._mat_size-1, :self._mat_size-1] = I_1
            L_T[self._mat_size-1:, self._mat_size-1:] = self._mat
            S1 = np.ones((self._mat_size - 1)*2 + 1)
            S1[:self._mat_size-1] = 0.
            S2 = np.ones((self._mat_size - 1)*2 + 1)
            S2[:self._mat_size-1] = 0.

            beta_init = np.ones(2*(self._mat_size-1))
            # beta = []
            # beta.append(np.zeros(2*(self._mat_size-1)))
            # step = 0
            # converged = False
            # func_val = []
            # lambda_k = 1./np.arange(2, 20, 2)
            # while not converged:
            #     print(step)
            #     func_val.append(function2min(beta[-1]))
            #     dk = -derivative(beta[-1]).dot(np.linalg.inv(hessian(beta[-1])))
            #     if func_val[-1] - function2min(beta[-1] + dk) >= -0.25*derivative(beta[-1]).T.dot(dk):
            #         step_size = 1
            #     else:
            #         for element in lambda_k:
            #             if func_val[-1] - function2min(beta[-1] + element*dk) >= -element*0.25 * derivative(beta[-1]).T.dot(dk):
            #                 step_size = element
            #                 break
            #     beta.append(beta[-1]+step_size*dk)
            #     if np.linalg.norm(beta[-1]-beta[-2]) <= tol:
            #         converged = True
            #
            #     step += 1
            #     if step >= imax:
            #         converged = True
            res = sp_opt.minimize(function2min, beta_init, jac=derivative, hess=hessian, method='Newton-CG')#, maxiter=imax)
            beta = res.x
            # beta = sp_opt.fmin_bfgs(function2min, beta_init, derivative, maxiter=imax)
            p = 1/mu*max_func(L_T.dot(beta) - S2)[self._mat_size-1:]
            q = 1/mu*max_func(-L_T.dot(beta) - S1)[self._mat_size-1:]

        weight = p - q
        return weight

    def predict_debug(self, x, weight_flag=0):
        if debug_flag:
            if weight_flag == -1:
                #lower weight
                self._alpha = self.weight_lower[:self.n_samples]
                self._intercept = self.weight_lower[-1]
            elif weight_flag == 1:
                self._alpha = self.weight_upper[:self.n_samples]
                self._intercept = self.weight_upper[-1]
            else:
                self._alpha = self.weight[:self.n_samples]
                self._intercept = self.weight[-1]

        if debug_flag:
            print(self._alpha[self._support_index_alpha])
            print(self._intercept)
        return self.predict(x)


class IRWLS(ML):
    """
    Implementation of the iterative reweighed least square support vector machine for the simultaneous learning of the
    function value and derivatives

    See [Lazaro et al., 2005]
    (https://www.sciencedirect.com/science/article/pii/S0925231205001657)
    ([pdf](https://ac.els-cdn.com/S0925231205001657/1-s2.0-S0925231205001657-main.pdf?_tid=fb34049c-b6a5-4e46-b305-3bebaf5b62aa&acdnat=1530192553_6eacd5a6333a7be4066aeac96b9ca70d))
    """

    def __init__(self, kernel, C1=1., C2=1., epsilon=1e-3, epsilon_prime=1e-3, max_iter=1e4):
        """
        :param kernel:
        :param C1: regularization of function values
        :param C2: regularization of derivative functions
        :param epsilon:  error insensitive region around the function value
        :param epsilon_prime: error insensitive region around the derivative values
        :param max_iter: maximum iteration of the algorithm
        """
        self._mat = None
        self.mat_size = None
        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        self.max_iter = max_iter

        ML.__init__(self, kernel, C1, C2)

    def fit(self, x_train, y_train, x_prime_train=None, y_prime_train=None, eps=1e-6):
        """
        Fit the model to given training data
        :param x_train: function training points shape (n_samples, n_features)
        :param y_train: function values corresponding to x_train shape (n_samples, 1)
        :param x_prime_train: derivative training points shape (n_samples, n_features)
        :param y_prime_train: derivative values corresponding to the x_prime_train (n_samples, n_features)
        :param eps: tolerance for the change in the end of the  lagrangian abs(l[-1]-l[-2])/l[-2] <= eps
        :return: None
        """
        if debug_flag:
            self.debug_plotting = (list(), list(), list())
            debug_idx_a = []
            debug_idx_s = []
            counter = 0

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

        #initialization of the weights with zero --> otherwise the lagrangian starts with a low value
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

            # error and lagrangian are calculated with all weights not only the supporting one like in Lazaros algorithm
            val_error, val_error_star, grad_error, grad_error_star = self.get_error(self._mat, weight)
            lagrangian.append(self.get_lagrangian(self._mat, weight, val_error + val_error_star, grad_error + grad_error_star))

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

                    if (cost <= lagrangian[-2]*(1 + eps)):
                        conv = True
                    else:
                        pond *= 0.5
                    iter_step += 1
                    if iter_step >= 1e2:
                        if debug_flag:
                            counter += 1
                        conv = True

                if debug_flag:
                    self.get_lagrangian(self._mat, dummy_weight, val_error + val_error_star, grad_error + grad_error_star,
                                    flag=debug_flag)

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
            print('iterative procedure failed %d times') %(counter)
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
        weight[error >= epsilon] = 2 * constant * (error[error >= epsilon] - epsilon) / error[error >= epsilon]
        weight[np.logical_and(error < epsilon, error > 0.)] = constant / epsilon
        weight[weight > 1 / epsilon] = 1 / epsilon
        return weight

    def _get_index(self, a, a_star, s, s_star):
        """
        creates index for the function and derivative error weight and for the actual weight as boolean vector
        :param a: upper function error weight
        :param a_star: lower function error weight
        :param s: upper derivative error weight
        :param s_star: lower derivative error weight
        :return: index of positive function error weight. index of positive derivative error weight, concatention  of both
        """
        idx_a = np.logical_or(a > 0., a_star > 0.)
        idx_s = np.logical_or(s > 0., s_star > 0.)
        idx_weight = np.concatenate([idx_a, idx_s, np.ones(1)])
        idx_weight = np.ndarray.astype(idx_weight, dtype=bool)
        return idx_a, idx_s, idx_weight

    def _get_target(self, idx_a, idx_s, a, a_star, s, s_star):
        """
        creates the target vector
        :param idx_a: positive function error weight index
        :param idx_s: positive derivative error weight index
        :param a: upper function error weight
        :param a_star: lower function error weight
        :param s: upper derivative error weight
        :param s_star: lower derivative error weight
        :return: target vector
        """
        return np.concatenate([self.y_train[idx_a] + (a[idx_a] - a_star[idx_a]) / (a[idx_a] + a_star[idx_a]) * self.epsilon,
                               self.y_prime_train[idx_s] + (s[idx_s] - s_star[idx_s]) / (s[idx_s] + s_star[idx_s])
                               * self.epsilon_prime, np.zeros([1])])

    def _get_mat(self, idx_weight):
        """
        selects a sub matrix of the original kernel matrix --> supporting matrix
        :param idx_weight: positive error weights index
        :return: sub matrix
        """
        idx_mat = np.logical_and(np.tile(idx_weight, self.mat_size).reshape(self.mat_size, self.mat_size),
                                 np.tile(idx_weight, self.mat_size).reshape(self.mat_size, self.mat_size).T)
        mat = cp.copy(self._mat)
        return mat[idx_mat].reshape(np.sum(idx_weight, dtype=int), np.sum(idx_weight, dtype=int))

    def get_error(self, mat, weight):
        """
        calculated the error of the prediction, negative errors are set to zero!
        :param mat: kernel matrix (total like in lazaros algorithm)
        :param weight: weights (total weight vector like in lazaros algorithm)
        :return: upper function error, lower function error, upper derivative error, lower derivative error
        """
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
        """
        calculated the actual lagrangian
        :param mat: kernel matrix (total like in lazaros algorithm)
        :param weight_vector: weights (total weight vecotor like in lazaros algorihm)
        :param val_error: function error
        :param grad_error: gradient error
        :param flag: debugging flag
        :return: lagrangian
        """
        weight_regularization = weight_vector.T.dot(mat.dot(weight_vector))/2.
        val_error = self._reg_value * sum(self._error_func(val_error, self.epsilon))
        grad_error = self._reg_derivative * sum(self._error_func(grad_error, self.epsilon_prime))

        if flag:
            self.debug_plotting[0].append(weight_regularization)
            self.debug_plotting[1].append(val_error)
            self.debug_plotting[2].append(grad_error)
        return weight_regularization + val_error + grad_error

    def _error_func(self, error, epsilon, approximate_function='L2'):
        """
        error approximation for the lagrangian
        :param error: actual error
        :param epsilon: insensitive region around the values
        :param approximate_function: error function approximation method default L2
        :return: lagrangian error value
        """
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
    See [Jayadeva et al., 2008]
    (https://www.sciencedirect.com/science/article/pii/S0020025508001291)
    ([pdf](https://ac.els-cdn.com/S0020025508001291/1-s2.0-S0020025508001291-main.pdf?_tid=3b4eb40c-d151-4b41-bf67-41783b08a262&acdnat=1530192958_9b64cae250c74899636a2713b72f3ccb))
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
    """ Implementation of a Gaussian process Regressor
    Hyper parameter optimization is done via negative log marginal likelihood
    See [Koistinen et al., 2017]
    (https://aip.scitation.org/doi/abs/10.1063/1.4986787)
    ([pdf](https://aip.scitation.org/doi/pdf/10.1063/1.4986787))
    """
    def __init__(self, kernel, noise_value=1e-10, noise_derivative=1e-10, opt_method='LBFGS_B', opt_parameter=True,
                 opt_restarts=0, normalize_y=False):
        """

        :param kernel: only RBF kernels are supported.
        :param noise_value: uncertainty in the function values
        :param noise_derivative: uncertainty  in the derivatives
        :param opt_method: method for the optimization of the kernels hyper parameters only LBFGS_B is supported at the moment. default LBFGS_B,
        :param opt_parameter: determines if the kernel hyper parameters should be optimized. default True
        :param opt_restarts: number of random initialized optimizations steps in addition to the first
        :param normalize_y: if true the mean of the function values is subtracted from the input function values. default True
        """
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
        """
        Fit the model to given training data
        Parameters:
            x_train: training points with shape (n_samples, n_features)
            y_train: function values corresponding to x_train shape (n_samples, 1)
            x_prime_train: training points for the derivatives (n_samples_derivative, n_features)
            y_prime_train: derivative values corresponding to x_prime_train (n_samples_derivative, n_features)
        """
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

    def optimize(self, hyper_parameter):
        """
        Function to optimize kernels hyper parameters
        :param hyper_parameter: new kernel hyper parameters
        :return: negative log marignal likelihood, derivative of the negative log marignal likelihood
        """
        self.set_hyper_parameter(hyper_parameter)
        log_marginal_likelihood, d_log_marginal_likelihood = self.log_marginal_likelihood(derivative=self._opt_flag)

        return -log_marginal_likelihood, -d_log_marginal_likelihood

    def log_marginal_likelihood(self, derivative=False):
        """
        calculate the log marignal likelihood
        :param derivative: determines if the derivative to the log marignal likelihood should be evaluated. default False
        :return: log marinal likelihood, derivative of the log marignal likelihood
        """
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
        # summation inspired form scikit-learn Gaussian process regression
        temp = (np.multiply.outer(alpha, alpha) - cho_solve((L, True), np.eye(L.shape[0])))[:, :, np.newaxis]
        d_log_mag_likelihood = 0.5 * np.einsum("ijl,ijk->kl", temp, k_grad)
        d_log_mag_likelihood = d_log_mag_likelihood.sum(-1)

        return log_mag_likelihood, d_log_mag_likelihood

    def _cholesky(self, kernel):
        """
        save routine to evaluate the cholesky factorization and weights
        :param kernel: kernel matrix
        :return: lower cholesky matrix, weights.
        """
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
        # Todo error estimation
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
    """
    if the cholesky decomposition fails this function is called to show what is possibly wrong
    :param mat: kernel matrix
    :return: None
    """
    print('symmertric = ' + str(np.allclose(mat, mat.T, 10**-32)))
    if not np.allclose(mat, mat.T, 10**-32):
        raise ValueError('not symmetric')
    print('derterminante = ' + str(np.linalg.det(mat)))
    eig_val, eig_vec = np.linalg.eigh(mat)
    print('eigenvalues = ' + str(eig_val))
    print('dimension = ' + str(mat.shape))
    print('----------------')


def create_mat(kernel, x1, x2, x1_prime=None, x2_prime=None, dx_max=0, dy_max=0, eval_gradient=False):
    """
    creates the kernel matrix with respect to the derivatives.
    :param kernel: given kernel like RBF
    :param x1: training points shape (n_samples, n_featuresprediction
    :param x2: training or prediction points (n_samples, n_features)
    :param x1_prime: derivative training points (n_samples, n_features)
    :param x2_prime: derivative training or prediction points (n_samples, n_features)
    :param dx_max: maximum derivative in x1_prime
    :param dy_max: maximum derivative in x2_prime
    :param eval_gradient: flag if kernels derivative have to be evaluated. default False
    :return: kernel matrix, derivative of the kernel matrix
    """
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
