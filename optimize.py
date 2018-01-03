import numpy as np
import xyz_file_writer as xyz_writer
import scipy as sp
import scipy.optimize as sp_opt

def scale_step(step, trust_radius):
    if np.linalg.norm(step) > trust_radius:
        return step / np.linalg.norm(step) * trust_radius
    return step


class SteepestDecent:
    def __init__(self, alpha, trust_radius):
        self.alpha = alpha
        self.trust_radius = trust_radius

    def new_step(self, gradient_function, func_values, *args):
        func, gradient = gradient_function(func_values, *args)
        return func_values + scale_step(self.alpha * gradient, self.trust_radius)


class Verlete:
    def __init__(self, delta_t, trust_radius):
        self.velocity = None
        self.delta_t = delta_t
        self.trust_radius = trust_radius

    def new_step(self, gradient_fucntion, func_values, *args):
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

    def new_step(self, gradient_function, func_values, *args):
        # algorithm from http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
        func, gradient = gradient_function(func_values, *args)
        if self.velocity is None:
            self.velocity = gradient * self.delta_t
        vf = np.dot(gradient, self.velocity)
        self.velocity = (1.0 - self.alpha) * self.velocity + self.alpha * gradient / np.linalg.norm(
            gradient) * np.linalg.norm(self.velocity)
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


class ConjuageGradient:
    def __init__(self, alpha, trust_radius):
        self.beta = 1.0
        self.alpha = alpha
        self.s = None
        self.force = None
        self.force_before = None
        self.trust_radius = trust_radius

    def new_step(self, gradient_func, func_values, *args):
        func, gradient = gradient_func(func_values, *args)
        if self.s is None:
            # func, gradient = gradient_func(func_values, *args)
            self.s = gradient
            self.force = gradient
        else:
            self.force = gradient
            beta = np.dot(self.force.T, self.force) / np.dot(self.force_before.T, self.force_before)

            if np.isnan(beta) | np.isinf(beta):
                beta = 1.0
            self.s = self.force + beta * self.s
        self.s = scale_step(self.s, self.trust_radius)
        func_values = func_values + self.s
        # func, gradient = gradient_func(func_values, *args)
        self.force_before = self.force
        # self.force = gradient

        return func_values


# Todo still fails (no convergence) --> alpha evaluation?
class BFGS:
    def __init__(self, trust_radius, hessian=None):
        # initialize the hessian  -- > wikipedia initial guess with identity matrix
        # cartesian coordinates x, y, z
        self.hessian = hessian
        self.true_hessian = False
        if hessian is not None:
            self.true_hessian = True
        self.gradient = None
        self.d = None
        self.positions = None
        self.trust_radius = trust_radius
        self.func_values = None
        self.gradient_func = None
        self.s = None

    def new_step(self, gradient_func, func_values, *args):
        self.gradient_func = gradient_func
        self.func_values = func_values
        func, gradient = gradient_func(func_values, *args)
        if self.gradient is None:
            self.gradient = gradient
        if self.hessian is None:
            self.hessian = np.eye(len(func_values), len(func_values))
        else:
            if not self.true_hessian:
                self.s = -np.dot(self.hessian, gradient)
                alpha = 1.0
                # alpha should be returned by a function that minimizes func(x+s*alpha) with alpha > 0
                # should get better
                self.d = scale_step(alpha * self.s, self.trust_radius)
                func_values += self.d
                func, gradient_new = gradient_func(self.func_values, *args)

                y = gradient_new - gradient
                a = np.outer(self.d, self.d.T) / np.dot(self.d.T, y)
                b = np.outer(np.dot(self.hessian, y), np.dot(self.hessian, y).T)/ np.dot(y.T, np.dot(self.hessian, y))
                self.hessian += a - b
            else:
                self.s = -np.dot(self.hessian, gradient)
                alpha = 1.0
                # alpha should be returned by a function that minimizes func(x+s*alpha) with alpha > 0
                # should get better
                self.d = scale_step(alpha * self.s, self.trust_radius)
                func_values += self.d

        return func_values

    def minimize(self, alpha, *args):
        func, gradient = self.gradient_func(self.func_values + alpha * self.s, *args)
        return func


# def center_geometry(positions):
#     # cartesian
#     # atoms in row and coordiantes in columns x, y, z
#     mean_value = np.zeros([len(positions), 3])
#     for kk in range(0, len(positions)):
#         pos = positions[kk, :, :]
#         mean_value[kk, :] = np.array([np.mean(pos[:, 0]), np.mean(pos[:, 1]), np.mean(pos[:, 2])])
#     return mean_value
#
#
# def rotation_geometry(positions):
#     # cartesian
#     # atoms in row and coordiantes in columns x, y, z
#     rotation_matrix_a = []
#     center_geometry(positions)
#
#     for ii in range(1, len(positions)):
#         u, w, v = np.linalg.svd(np.dot(positions[ii, :, :].T, positions[ii-1, :, :]))
#         a_j = np.dot(u, v)
#
#         if np.linalg.det(a_j) < 0:
#             u[:, -1] = -u[:, -1]
#             a_j = np.dot(u, v)
#         positions[ii, :, :] = np.dot(positions[ii, :, :], a_j)
#         rotation_matrix_a.append(a_j)
#
#     return positions, rotation_matrix_a


