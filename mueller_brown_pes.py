import numpy as _np
import scipy.optimize as _sp_opt
import scipy as _sp

# mueller brown potential energy surface
# http://demonstrations.wolfram.com/TrajectoriesOnTheMullerBrownPotentialEnergySurface/
#
# # special points on the surface
# minima
# x = -0.558224, y = 1.44173
# x = 0.623499, y = 0.0280378
# x = -0.0500108, y = 0.466694

# saddle points
# x = 0.212487, y = 0.292988
# x = -0.822002, y = 0.624313


# A, a, b, c, x_zero, y_zero
coefficients = _np.array([[-200,    -1,     0,     -10,   1,     0],
                          [-100,    -1,     0,     -10,   0,     0.5],
                          [-170,    -6.5,   11,    -6.5, -0.5,   1.5],
                          [15,      0.7,    0.6,    0.7,  -1,     1]])


def energy(x, y):
    res = sum(coeff[0]*_np.exp(coeff[1]*(x-coeff[4])**2 + coeff[2]*(x-coeff[4])*(y-coeff[5])
                         + coeff[3]*(y-coeff[5])**2) for coeff in coefficients)
    return res


def gradient(x, y):

    dx = sum((2 * coeff[1] * (x - coeff[4]) + coeff[2] * (y - coeff[5])) * coeff[0]*_np.exp(coeff[1]*(x-coeff[4])**2
                    + coeff[2]*(x-coeff[4])*(y-coeff[5]) + coeff[3]*(y-coeff[5])**2) for coeff in coefficients)

    dy = sum((2 * coeff[3] * (y - coeff[5]) + coeff[2] * (x - coeff[4])) * coeff[0]*_np.exp(coeff[1]*(x-coeff[4])**2
                    + coeff[2]*(x-coeff[4])*(y-coeff[5]) + coeff[3]*(y-coeff[5])**2) for coeff in coefficients)

    return dx, dy


def energy_gradient(xy, *args):
    e = energy_xy(xy)
    grad =gradient_xy(xy)
    return e, grad, None


def energy_xy(xy):
    x = xy[0]
    y = xy[1]
    return energy(x, y)


def gradient_xy(xy):
    x = xy[0]
    y = xy[1]
    dx, dy = gradient(x, y)
    return _np.append(dx,dy)


def energy_xy_list(xy, *args):
    e = []
    for element in xy:
        e.append(energy(element[0], element[1]))
    return e


def gradient_xy_list(xy, *args):
    grad = []
    for element in xy:
        grad.append(gradient(element[0], element[1]))
    return grad


def get_minimum(x, y):
    xy = _np.array([x, y])
    point = _sp_opt.fmin_bfgs(energy_xy, xy, disp=False)
    return point


def plot_mep():
    # points on the way down
    n = 3000

    saddle_a = _np.array([0.212487,  0.292988])
    saddle_b = _np.array([-0.822002, 0.624313])
    # point = saddle(0, 0)
    d = 10 ** -4

    minima = []
    minima.append(_np.array([-0.558224, 1.44173]))
    minima.append(_np.array([0.623499, 0.0280378]))
    minima.append(_np.array([-0.0500108, 0.466694]))

    path = [_np.zeros([n + 1, 2])]
    path.append(_np.zeros([n + 1, 2]))
    path.append(_np.zeros([n + 1, 2]))
    path.append(_np.zeros([n + 1, 2]))
    path[0][0][0] = saddle_a[0] - d
    path[0][0][1] = saddle_a[1] + d
    path[1][0][0] = saddle_a[0] + d
    path[1][0][1] = saddle_a[1] - d
    path[2][0][0] = saddle_b[0] + d
    path[2][0][1] = saddle_b[1] - d
    path[3][0][0] = saddle_b[0] - d
    path[3][0][1] = saddle_b[1] + d

    def normalized_force(x, y):
        force = -_np.array(gradient(x, y))
        return force #/ _np.linalg.norm(force)

    epsilon = 10 ** -4

    for ii in range(n):
        for element in path:
            step = normalized_force(element[ii, 0], element[ii, 1])
            element[ii+1, :] = element[ii, :] + step* epsilon
    # path_reactant[0, 0] = point[0]
    # path_reactant[0, 1] = point[1]
    # path_product[-1, 0] = point[0]
    # path_product[-1, 1] = point[1]
    # while ii <= 10**4:
    #     step_reactant = normalized_force(x_reactant, y_reactant)
    #     x_reactant = x_reactant + epsilon * step_reactant[0]
    #     y_reactant = y_reactant + epsilon * step_reactant[1]
    #
    #     step_product = normalized_force(x_product, y_product)
    #     x_product = x_product + epsilon * step_product[0]
    #     y_product = y_product + epsilon * step_product[1]
    #     ii = ii + 1
    #     if (ii % 10 ** 1) == 0:
    #         jj = jj + 1
    #         path_reactant[jj, 0] = x_reactant
    #         path_reactant[jj, 1] = y_reactant
    #         path_product[-jj-1, 0] = x_product
    #         path_product[-jj-1, 1] = y_product
    path_print = _np.zeros([n*4+4,2])
    path_print[:n+1, :] = path[1][::-1]
    path_print[n+1:2*n+2, :] = path[0]
    path_print[2*n + 2:3 * n + 3, :] = path[2][::-1]
    path_print[3*n + 3:4 * n + 4, :] = path[3]

    x_print = path_print[:, 0]
    y_print = path_print[:, 1]
    return x_print, y_print


def get_pred_grid(n_pred, plot_3D=False):
    if plot_3D:
        xx_pred, yy_pred = _np.meshgrid(_np.linspace(-1.5, 0.85, n_pred),
                                  _np.linspace(-.5, 1.68, n_pred))# for 3D

    else:
        xx_pred, yy_pred = _np.meshgrid(_np.linspace(-1.5, 1., n_pred),
                                    _np.linspace(-.5, 1.7, n_pred))

    return xx_pred, yy_pred


def get_transition_state():
    ts = []
    ts.append(_np.array([0.212487, 0.292988]))
    ts.append(_np.array([-0.822002, 0.624313]))
    return ts


def get_levels():
    return _np.linspace(-150,150, 50)


def get_minima():
    minima = []
    minima.append(get_minimum(-0.558224, 1.44173))
    minima.append(get_minimum(0.623499, 0.0280378))
    minima.append(get_minimum(-0.0500108, 0.466694))
    return minima


