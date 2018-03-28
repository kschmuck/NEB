import numpy as np


def scale_step(step, trust_radius):
    if np.linalg.norm(step) > trust_radius:
        return step / np.linalg.norm(step) * trust_radius
    return step


class SteepestDecent:
    def __init__(self, trust_radius, alpha=0.1): # , gamma=0.7, n_back=100, alpha=1.0, epsilon=0.1):
        self.alpha = alpha
        self.trust_radius = trust_radius
        self.gradient_before = None

        # self.alpha_0 = self.alpha
        # self.gamma = gamma
        # self.n_back = n_back
        # self.n_0 = n_back
        # self.epsilon = epsilon

    def step(self, gradient_function, func_values, *args):
        func, gradient = gradient_function(func_values, *args)
        # if self.gradient_before is None:
        #     self.gradient_before = gradient
        # else:
        #     self.alpha, self.n_back, self.skip = backtracking(gradient, self.gradient_before, self.epsilon, self.alpha,
        #                                                       self.alpha_0, self.n_0, self.n_back, self.gamma)
        # step = scale_step(gradient, self.trust_radius)
        # func_values += step*self.alpha
        # return func_values
        return func_values + scale_step(self.alpha * gradient, self.trust_radius)


class Verlete:
    def __init__(self, delta_t, trust_radius):
        self.velocity = None
        self.delta_t = delta_t
        self.trust_radius = trust_radius

    def step(self, gradient_fucntion, func_values, *args):
        func, gradient = gradient_fucntion(func_values, *args)
        if self.velocity is None:
            self.velocity = gradient * self.delta_t
        self.velocity = self.velocity + 0.5 * self.delta_t * gradient
        uf = np.dot(self.velocity, gradient)
        if uf <= 0.0:
            self.velocity[:] = 0.0
        else:
            self.velocity[:] = uf / np.dot(gradient, gradient) * gradient
        self.velocity = self.velocity + 0.5 * self.delta_t * gradient
        return func_values + scale_step(self.delta_t * self.velocity, self.trust_radius)


class Fire:

    def __init__(self, delta_t, delta_t_max, trust_radius, alpha=0.1, m=1, n_min=5,
                 f_inc=1.1, f_dec=0.5, alpha_dec=0.99): # m=4
        # Parameter from http://nanosurf.fzu.cz/wiki/doku.php?id=fire_minimization
        self.alpha = alpha
        self.alpha_start = alpha
        self.mass = m
        self.n_min = n_min
        self.n_min_step = 0
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.delta_t = delta_t
        self.delta_t_max = delta_t_max
        self.alpha_dec = alpha_dec
        self.velocity = None

        self.trust_radius = trust_radius

    def step(self, gradient_function, func_values, *args):
        # algorithm from http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
        func, gradient = gradient_function(func_values, *args)
        if self.velocity is None:
            self.velocity = gradient * self.delta_t
        vf = np.dot(gradient, self.velocity)
        if np.linalg.norm(gradient) != 0:
            self.velocity = (1.0 - self.alpha) * self.velocity + self.alpha * gradient / np.linalg.norm(
                gradient) * np.linalg.norm(self.velocity)
        else:
            self.velocity = (1.0 - self.alpha) * self.velocity + self.alpha * gradient * np.linalg.norm(self.velocity)
        if (vf >= 0.0):
            if (self.n_min_step > self.n_min):
                self.delta_t = min(self.delta_t * self.f_inc, self.delta_t_max)
                self.alpha = self.alpha * self.alpha_dec
            self.n_min_step = self.n_min_step + 1
        else:
            self.delta_t = self.delta_t * self.f_dec
            self.velocity = 0.0
            self.alpha = self.alpha_start
            self.n_min_step = 0
        self.velocity = self.velocity + self.delta_t * gradient / self.mass
        return func_values + scale_step(self.velocity*self.delta_t, self.trust_radius)


def backtracking(force_i, force_j, epsilon, alpha, alpha_0, n_0, n_back, gamma):
    # algorithm: cite: Computational Implementation of Nudged Elastic Band, Rigid Rotation and Corresponding Force Optimization
    # Herbol, Stevenson, Clancy
    # Journal of Chemical Theory and Computation 2017, 13, 3250-3259

    # force_i .... current force
    # force_j .... force before
    # RMS (root mean square is necessary) of forces
    force_i = np.sqrt(np.dot(force_i, force_i)/len(force_j))
    force_j = np.sqrt(np.dot(force_j, force_j)/len(force_j))
    chk = force_i - force_j
    chk /= abs(force_i+force_j)
    skip = False
    if chk > epsilon:
        alpha = alpha * gamma
        skip = True
        n_back = n_0
    else:
        n_back = n_back - 1
        if n_back < 0:
            n_back = n_0
            if alpha < alpha_0:
                alpha = alpha_0
                skip = True
            else:
                alpha = alpha / gamma
    return alpha, n_back, skip

# Backtracking is not working correctly, depends very strong on the parameters
class ConjuageGradient:
    # special case of CG with backtracking and no line search, thus evaluation of the force is very expensive
    # algorithm: cite: Computational Implementation of Nudged Elastic Band, Rigid Rotation and Corresponding Force Optimization
    # Herbol, Stevenson, Clancy
    # Journal of Chemical Theory and Computation 2017, 13, 3250-3259
    def __init__(self, trust_radius, gamma=0.5, n_back=10, alpha=1.0, epsilon=0.2):
        self.beta = 1.0
        self.alpha = alpha
        self.s = None
        self.force = None
        self.force_before = None
        self.trust_radius = trust_radius
        self.epsilon = epsilon

        # parameter for backtracking
        self.alpha_0 = self.alpha
        self.gamma = gamma
        self.n_back = n_back
        self.n_0 = n_back
        self.skip = False

    def step(self, gradient_func, func_values, *args):
        func, gradient = gradient_func(func_values, *args)
        if self.s is None:
            self.s = gradient
            self.force = gradient
        else:
            self.force = gradient
            self.alpha, self.n_back, self.skip = backtracking(self.force, self.force_before, self.epsilon, self.alpha, self.alpha_0, self.n_0, self.n_back, self.gamma)
            beta = np.dot(self.force.T, self.force) / np.dot(self.force_before.T, self.force_before)
            if np.isnan(beta) | np.isinf(beta):
                beta = 1.0
            self.s = self.force - beta * self.s

        self.s *= self.alpha
        self.s = scale_step(self.s, self.trust_radius)
        func_values = func_values + self.s
        self.force_before = self.force
        return func_values

class BFGS:
    # special case of BFGS with out any line search, but backtracking
    # algorithm: cite: Computational Implementation of Nudged Elastic Band, Rigid Rotation and Corresponding Force Optimization
    # Herbol, Stevenson, Clancy
    # Journal of Chemical Theory and Computation 2017, 13, 3250-3259

    def __init__(self, trust_radius, gamma=0.9, n_back=5, alpha=1.0, epsilon=0.1):
        # initialize the hessian  -- > wikipedia initial guess with identity matrix

        self.d = None
        self.hessian = None
        self.trust_radius = trust_radius
        self.s = None
        self.gradient_hold = None
        self.func_values_hold = None
        self.is_identity = True

        # parameter for backtracking
        self.alpha = alpha
        self.first_iter = True
        self.alpha_0 = self.alpha
        self.gamma = gamma
        self.n_back = n_back
        self.n_0 = n_back
        self.skip = False
        self.epsilon = epsilon

    def step(self, gradient_func, func_values, *args):
        # self.func_values = func_values
        func, gradient = gradient_func(func_values, *args)
        self.func_values_hold = func_values
        if self.gradient_hold is None:
            self.gradient_hold = gradient
        else:
            self.alpha, self.n_back, skip = backtracking(gradient, self.gradient_hold, self.epsilon, self.alpha,
                                                         self.alpha_0, self.n_0, self.n_back, self.gamma)
            if skip:
                self.hessian = np.eye(len(func_values), len(func_values))
                self.is_identity = True
                return func_values
            sigma = self.d #func_values - self.func_values_hold
            y = -gradient + self.gradient_hold
            roh = 1.0/np.dot(y.T, sigma)
            if self.is_identity:
                self.hessian = np.dot(y.T, sigma) / np.dot(y.T,y) * self.hessian
                self.is_identity = False
            A = np.eye(len(func_values), len(func_values)) - np.outer(sigma, y.T) * roh
            B = np.eye(len(func_values), len(func_values)) - np.outer(y, sigma.T) * roh
            self.hessian = np.dot(A, np.dot(self.hessian, B)) + np.outer(sigma, sigma.T)*roh

        if self.hessian is None:
            self.hessian = np.eye(len(func_values), len(func_values))
            self.is_identity = True

        self.s = np.dot(self.hessian, gradient)
        self.d = scale_step(self.s, self.trust_radius)
        func_values += self.alpha * self.d

        return func_values

