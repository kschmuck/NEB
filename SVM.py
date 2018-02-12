import numpy as np
# import tensorflow as tf
# from tensorflow.contrib.opt import ScipyOptimizerInterface
import scipy.optimize._minimize as spmin


def euclidean_distance(x, y):
    # input values are in the shape of [n_samples, n_features]!
    #  if n_samples = 1 --> reshape to have a shape of [1, n_features]

    xx = np.sum(x**2, axis=1).reshape(-1, 1)
    yy = np.sum(y**2, axis=1).reshape(1, -1)
    xy = 2.0*np.dot(x, y.T)
    squared_distance = xx + yy - xy
    return squared_distance


class SVM:
    def __init__(self, epsilon=0.1, epsilon_beta=0.1, kernel='rbf', gamma=0.1):
        self.epsilon = epsilon
        self.epsilon_beta = epsilon_beta
        self.b = None
        self.alpha = None
        self.support_index = None

        self.beta = None
        self.support_index_beta = None
        self.input_derivative = None

        self.target = None
        self.input_value = None

        if kernel == 'rbf':
            self.kernel = RBF(gamma=gamma)

        def loss(u):
            l = np.zeros(u)
            l[u >= self.epsilon] = u[u >= self.epsilon] ** 2 - 2 * u[
            u >= self.epsilon] * self.epsilon + self.epsilon ** 2
            return l

    # def fit_new(self, value, target, derivative=None, C = 1.0):
    #     kernel = self.kernel.derivative_kernel(value, derivative, y=value, dy=derivative)
    #     n_samples = len(target)
    #
    #     def func2min(x):
    #         alpha = x[:n_samples]
    #         alpha_star = x[n_samples:2*n_samples]
    #         beta = x[2*n_samples:3*n_samples]
    #         beta_star = x[3*n_samples:]
    #         func = 0.5
    #
    #     beta_k = np.zeros(n_samples)
    #     u = target
    #     rho = 0.9
    #     a = np.zeros(n_samples)
    #
    #     check_u = u >= self.epsilon
    #     a[check_u.flatten()] = 2*C * (u[check_u]-self.epsilon)/u[check_u]
    #     kernel = self.kernel.kernel(value, y=value)
    #     alpha_matrix = np.eye(n_samples)
    #     alpha_matrix *= a
    #     beta_s = np.dot(np.invert(kernel+alpha_matrix), target)
    #     eta = 1.
    #     beta_k += eta*(beta_s-beta_k)
    #     eta *= eta*rho



    def fit(self, input_value, target_value, derivative_target=None, C=1.0, D=1.0):
        self.target = target_value
        self.input_value = input_value
        kernel = self.kernel.kernel(input_value)
        n_samples = len(target_value)
        if derivative_target is None:
            # inequalities have to be func >= 0
            constrains = ({'type': 'eq', 'fun': lambda x: np.array(np.sum(x[:n_samples]-x[n_samples:]))},
                          {'type': 'ineq', 'fun': lambda x: np.array(x)},
                          {'type': 'ineq', 'fun': lambda x: np.array(-x + C)})

            def dual_func(x):
                # x = [alpha, alpha_star]
                alpha = x[:n_samples]
                alpha_star = x[n_samples:]
                term_a = -0.5 * np.dot(np.dot(np.transpose(alpha - alpha_star), kernel),alpha - alpha_star)
                term_b = self.epsilon * np.sum(alpha + alpha_star) + np.dot(self.target.T, (alpha - alpha_star))
                return -(term_a + term_b)

            res = spmin.minimize(dual_func, np.zeros(n_samples*2), method='SLSQP', constraints=constrains)

            self.alpha = res.x[:n_samples]-res.x[n_samples:]
            self.support_index = np.arange(0, n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]

        else:
            # first_derivative, second_derivative = self.kernel.derivative_kernel(input_value, derivative_target)
            self.input_derivative = derivative_target
            k_xy, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(input_value, derivative_target, y=input_value, dy=derivative_target)

            # def dual_func_test(x):
            #
            #     function = 0.5*((alpha-alpha_star))

            def dual_func(x):
                # x contains alpha (first), alpha_star(second), beta(third), beta_star (fourth)
                # x = [alpha, alpha_star, beta, beta_star]
                index_parameter = n_samples #int(len(x)/4)
                alpha = x[:index_parameter]
                alpha_star = x[index_parameter:2*index_parameter]
                beta = x[2*index_parameter:3*index_parameter]
                beta_star = x[3*index_parameter:]

                # # term_a  =  (alpha_j-alpha_star_j).T*<xj, xi>*(alpha_i-alpha_star_i)
                # term_a = -0.5*np.dot(np.dot(np.transpose(alpha-alpha_star), k_xy), (alpha-alpha_star))
                #
                # # term_b = (beta_j-beta_star_j).T*<dx_j, dx_i>*(beta_i-beta_star_i)
                # term_b = -0.5*np.dot(np.dot(np.transpose(beta-beta_star), k_dxdy), (beta-beta_star))
                #
                # # term_d = (beta_j-beta_star_j)*<dx_j, x_i>*(alpha_i, alpha_star_i)
                # term_d = -0.5*np.dot(np.dot(np.transpose(beta-beta_star), k_dy), (alpha-alpha_star))
                #
                # # term_c = (alpha_j-alpha_star_j)*<x_j, dx_i>*(beta_i-beta_star_i)
                # term_c = -0.5*np.dot(np.dot(np.transpose(alpha-alpha_star), k_dx), (beta-beta_star))
                #
                # # term_e = -epsilon*(alpha_i+alpha_star_i) + y_i*(alpha_i-alpha_star_i)
                # term_e = -self.epsilon*np.sum(alpha+alpha_star)-np.dot(target_value.T,(alpha-alpha_star))
                #
                # # term_f = -epsilon*(beta_i+beta_star_i)+ y_i'*(beta_i-beta_star_i)
                # term_f = -self.epsilon_beta*np.sum(beta+beta_star)-np.sum(np.dot(derivative_target.T, (beta-beta_star)))
                #
                # return -(term_a + term_b + term_c + term_d + term_e + term_f)
                # func = -.5*(np.dot(np.transpose(alpha_star-alpha), np.dot((alpha_star-alpha),k_xy))
                #             + np.dot(np.transpose(beta_star-beta), np.dot((beta_star-beta), k_dxdy))
                #             + np.dot(np.transpose(alpha_star-alpha), np.dot((beta_star-beta), k_dy))
                #             + np.dot(np.transpose(beta_star-beta), np.dot((alpha_star-alpha), k_dx)))\
                #             - np.sum((alpha+alpha_star)*self.epsilon + (beta+beta_star)*self.epsilon_beta) \
                #             + np.dot(np.transpose(alpha_star-alpha), target_value)+np.dot(np.transpose(beta_star-beta), derivative_target)

                func = -0.5 * (np.dot(np.dot(np.transpose(alpha - alpha_star), k_xy), (alpha-alpha_star))
                            + np.dot(np.dot(np.transpose(alpha - alpha_star), k_dx), (beta-beta_star))
                            + np.dot(np.dot(np.transpose(beta-beta_star), k_dy), (alpha-alpha_star))
                            + np.dot(np.dot(np.transpose(beta-beta_star), k_dxdy), (beta-beta_star)))\
                            - self.epsilon*np.sum(alpha+alpha_star) + np.dot(self.target.T, (alpha-alpha_star))\
                            - self.epsilon_beta*np.sum(beta-beta_star) + np.dot(self.input_derivative.T, (beta-beta_star))
                return -func

            constrains = ({'type': 'eq', 'fun': lambda x: np.array(np.sum(x[:n_samples]-x[n_samples:2*n_samples]))},
                          {'type': 'eq', 'fun': lambda x: np.array(np.sum(x[n_samples*2:n_samples*3] - x[n_samples*3:]))},
                          {'type': 'ineq', 'fun': lambda x: np.array(x)},
                          {'type': 'ineq', 'fun': lambda x: np.array(-x[:2*n_samples] + C)},
                          {'type': 'ineq', 'fun': lambda x: np.array(-x[2 * n_samples:] + D)})

            res = spmin.minimize(dual_func, np.zeros(n_samples*4), method='SLSQP', constraints=constrains)
            self.alpha = res.x[:n_samples] - res.x[n_samples:2*n_samples]
            self.support_index = np.arange(0, n_samples, 1, dtype=int)[np.abs(self.alpha) > 10**-8]
            self.beta = res.x[2*n_samples:3*n_samples] - res.x[3*n_samples:]
            self.support_index_beta = np.arange(0, n_samples, 1, dtype=int)[np.abs(self.beta) > 10 ** -8]

        self.b = 0.0
        # if alpha and alpha_star are in the region [0,C] then b = y-epsilon-<w,x>
        # w in the simple case = (alpha-alpha_star)*x_i
        # w in the advanced case = (alpha-alpha_star)*x_i+(beta-beta_star)*x_i'
        b = target_value[self.support_index].reshape(-1)-self.epsilon-self.predict(input_value[self.support_index])
        self.b = np.sum(b, axis=0)/len(b)

    def predict(self, predict_value):
        if self.beta is not None:
            # x_i --> support vectors, dx_i --> support gradient vector
            # predict = <w,x> + b
            # w = (alpha_i-alpha_star_i)*<x_i| + (beta_i-beta_star_i)*<dx_i|
            # k_xy, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(self.input_value[self.support_index], self.input_derivative[self.support_index], y=predict_value)
            k_xy = self.kernel.kernel(self.input_value[self.support_index], y=predict_value)
            k_, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(self.input_value[self.support_index_beta], self.input_derivative[self.support_index_beta], y=predict_value)

            alpha = self.alpha[self.support_index]
            beta = self.beta[self.support_index_beta]

            prediction = np.dot(alpha.T, k_xy) + np.dot(beta.T, k_dx) + self.b
            # k_xy, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(self.input_value, self.input_derivative,
            #                                                          y=predict_value)
            # prediction = np.dot(self.alpha, k_xy) + np.dot(self.beta, k_dx) + self.b
        else:
            kernel = self.kernel.kernel(self.input_value[self.support_index], y=predict_value)
            prediction = np.dot(self.alpha[self.support_index], kernel) + self.b

        return prediction

    def predict_derivative(self, predict_value):
        k_xy, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(self.input_value[self.support_index_beta], self.input_derivative[self.support_index_beta], y=predict_value)
        # k_xy, k_dx, k_dy, k_dxdy = self.kernel.derivative_kernel(self.input_value, self.input_derivative, y=predict_value)
        beta = self.beta[self.support_index_beta]
        return np.dot(beta, k_dx)


class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = -gamma

    def kernel(self, x, y=None):

        if y is None:
            y = x
        squared_distance = euclidean_distance(x, y)
        kernel = np.exp(self.gamma*squared_distance)
        return kernel

    def derivative_kernel(self, x, dx, y=None, dy=None):
        # derivative of kernel with respect to the different components,
        # K(x, y), d/dx*K(x, y), d/dy*K(x, y), d/dx*d/dy*K(x, y)
        if y is None:
            y = x

        squared_distance = euclidean_distance(x, y)

        k_xy = np.exp(self.gamma*squared_distance)

        # x[0] --> features of the first sample --> all samples have the same feature number
        # len(dx) = num samples
        k_dx = np.zeros([len(x), len(y)])
        k_dy = np.zeros([len(x), len(y)])
        k_dxdy = np.zeros([len(x), len(y)])

        for ii in range(0, len(x)):
            for jj in range(0, len(y)):
                # k_dx = -2*gamma*<x_i|x_j>*(x_i-x_j)*dx_i
                k_dx[ii, jj] = -2 * self.gamma * k_xy[ii, jj] * np.dot((x[ii, :] - y[jj, :]), dx[ii, :])
                if dy is not None:
                    # k_dy = 2*gamma*<x_i|x_j>*(x_i-x_j)*dx_j
                    k_dy[ii, jj] = 2*self.gamma * k_xy[ii, jj]*np.dot((x[ii, :]-y[jj, :]), dy[jj, :])
                    # k_dxdy = -4*gamma^2*<x_i|x_j>*(x_i-x_j)^2*dx_i*dx_j+2*gamma <x_i|x_j>
                    k_dxdy[ii, jj] = np.sum((-4 * self.gamma ** 2 * np.square(x[ii, :] - y[jj, :]) + 2 * self.gamma) * dx[ii, :]* dy[jj, :] * k_xy[ii, jj])

        # for ii in range(0, len(x)):
        #     for jj in range(0, len(y)):
        #         # k_dx = -2*gamma*<x_i|x_j>*(x_i-x_j)*dx_i
        #         k_dx[ii, jj] = -2 * self.gamma * k_xy[ii, jj] * (x[ii,:]-y[jj, :])
        #         if dy is not None:
        #             # k_dy = 2*gamma*<x_i|x_j>*(x_i-x_j)*dx_j
        #             k_dy[ii, jj] = 2*self.gamma * k_xy[ii, jj]*(x[ii, :]-y[jj, :])
        #             # k_dxdy = -4*gamma^2*<x_i|x_j>*(x_i-x_j)^2*dx_i*dx_j+2*gamma <x_i|x_j>
        #             k_dxdy[ii, jj] = (-4 * self.gamma**2 *(x[ii]-y[jj])*(x[ii]-y[jj]) + 2*self.gamma) * k_xy[ii, jj]


        return k_xy, k_dx, k_dy, k_dxdy
