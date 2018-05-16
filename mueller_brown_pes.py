import numpy as _np
# mueller brown potential energy surface
# http://demonstrations.wolfram.com/TrajectoriesOnTheMullerBrownPotentialEnergySurface/



# A, a, b, c, x_zero, y_zero
coefficients = _np.array([[-200,    -1,     0,     -10,   1,     0],
                          [-100,    -1,     0,     -10,   0,     0.5],
                          [-170,    -6.5,   11,    -6.5, -0.5,   1.5],
                          [15,      0.7,    0.6,    0.7,  -1,     1]])


def energy(x, y):
    # for element in coefficients:
    #     print('A = %f, a = %f, b = %f, c= %f, x_0 = %f, y_0 = %f')% (element[0], element[1], element[2], element[3], element[4], element[5])
    res = sum(coeff[0]*_np.exp(coeff[1]*(x-coeff[4])**2 + coeff[2]*(x-coeff[4])*(y-coeff[5])
                         + coeff[3]*(y-coeff[5])**2) for coeff in coefficients)
    return res


def energy_list(xy):
    e = []
    for element in xy:
        e.append(energy(element[0], element[1]))
    return e


def gradient(x, y):
    dx = sum((2 * coeff[1] * (x - coeff[4]) + coeff[3] * (y - coeff[5])) * coeff[0] * (_np.exp(
        coeff[1] * (x - coeff[4]) ** 2 + coeff[2] * (x - coeff[4]) * (y - coeff[5]) + coeff[3] * (y - coeff[5]) ** 2))
              for coeff in coefficients)

    dy = sum((2 * coeff[3] * (y - coeff[5]) + coeff[3] * (x - coeff[4])) * coeff[0] * (_np.exp(
        coeff[1] * (x - coeff[4]) ** 2 + coeff[2] * (x - coeff[4]) * (y - coeff[5]) + coeff[3] * (y - coeff[5]) ** 2))
              for coeff in coefficients)

    return dx, dy


def gradient_list(xy):
    grad = []
    for element in xy:
        grad.append(gradient(element[0], element[1]))
    return grad