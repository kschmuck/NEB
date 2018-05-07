import NEB as neb
import numpy as np
from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list, minimum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MLDerivative as sv
import sklearn as sk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Patch3DCollection
from sklearn import svm, gaussian_process, model_selection
import Kernels
import copy
from sklearn import gaussian_process as gpr

# ### 1D Test case
# offset = 3
# amplitude = 2
# modification = 1/2.
#
# def energy_1D(x):
#     # return np.sinc(x)
#     return amplitude*np.sin(modification*x) + offset
#
#
# def gradient_1D(x):
#     # return np.cos(x)/x+np.sinc(x)/x
#     return modification*amplitude*np.cos(x*modification)
#
#
# gamma = 0.1
# epsilon = 0.01
# c1 = 1.
# c2 = 1.
#
# seed = 123
# np.random.seed(seed)
#
# sv_test = []
# kernel_gpr = Kernels.RBF()*Kernels.ConstantKernel()
# kernel_rls = Kernels.RBF()
#
# # sv_test.append(sv.RLS(kernel_rls))
# sv_test.append(sv.GPR(kernel_gpr))
#
# sk_test = svm.SVR(C=c1, kernel='rbf', gamma=gamma, epsilon=epsilon)
# sk_test_gpr = gpr.GaussianProcessRegressor(gpr.kernels.RBF()*gpr.kernels.ConstantKernel(), normalize_y=True)
#
# x_predict = np.linspace(-15*np.pi, 15*np.pi, 2000).reshape(-1,1)
# x = (np.random.rand(30) - 0.5) * np.pi * 18
# x = x.reshape(-1, 1)
#
# sk_test.fit(x, energy_1D(x).reshape(-1))
# sk_test_gpr.fit(x, energy_1D(x).reshape(-1))
# np.random.seed()
# sv_val = []
#
# # sk_val = [(sk_test.__class__.__name__, sk_test.predict(x_predict)),
# #           (sk_test_gpr.__class__.__name__, sk_test_gpr.predict(x_predict))]
# sk_val = [(sk_test_gpr.__class__.__name__, sk_test_gpr.predict(x_predict, return_std=True))]
#
# for element in sv_test:
#     # element.fit(x, energy_1D(x).reshape(-1))
#     element.fit(x, energy_1D(x).reshape(-1), x_prime_train=x, y_prime_train=gradient_1D(x).reshape(-1, 1))
#     # element.fit(np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1))
#     # sv_val.append((element.__class__.__name__, element.predict(x_predict)))
#
#     sv_val.append((element.__class__.__name__, element.predict(x_predict, error_estimate=True)))
#
# fig = plt.figure()
# plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
# plt.plot(x, energy_1D(x), ls='None', color='k', marker='x', label='Training Points')
#
# for element in sv_val:
#     plt.plot(x_predict, element[1][0], label=element[0])
#     plt.fill_between(x_predict.reshape(-1), element[1][0] + 2. * np.sqrt(element[1][1]),
#                      element[1][0] - 2. * np.sqrt(element[1][1]), facecolor=[0.7539, 0.62890625, 0.89453125, .5],
#                      linewidths=0.0, label='gpr')
#
#
# for element in sk_val:
#     string = element[0] + 'scikit'
#     plt.plot(x_predict, element[1][0], label=string, ls='--')
#     plt.fill_between(x_predict.reshape(-1), element[1][0] + 2. * np.sqrt(element[1][1]),
#                      element[1][0] - 2. * np.sqrt(element[1][1]), facecolor=[0.62890625, 0.89453125, 0.7539, .5],
#                      linewidths=0.0, label='scikit')
# plt.legend()
# plt.show()
# #
# # # plt.fill_between(x_predict.reshape(-1), sk_val.reshape(-1) + 2. * np.sqrt(np.diag(sk_std)), sk_val.reshape(-1) - 2. * np.sqrt(np.diag(sk_std)), facecolor=[0.7539, 0.62890625, 0.89453125, .5],
# # #                  linewidths=0.0)

#######################################################################################################################
# #######################################################################################################################
# ########################################################################################################################
n = 3
grid = 3.
# xx, yy = np.meshgrid(np.linspace(-grid, grid, n), np.linspace(-grid, grid, n))
xy = np.random.uniform(size=(10, 2), low=-5., high=5.)
xx = xy[:,0]
yy = xy[:,1]

n_pred = 30
grid_pred = 7.5
xx_pred, yy_pred = np.meshgrid(np.linspace(-grid_pred, grid_pred, n_pred), np.linspace(-grid_pred, grid_pred, n_pred))
xy_pred = np.concatenate([xx_pred.reshape(-1,1), yy_pred.reshape(-1,1)], axis=1)

#
# minima_a = np.array([3, 1])
# minima_a = minimum(minima_a[0], minima_a[1])
# minima_b = np.array([-2, 2])
# minima_b = minimum(minima_b[0], minima_b[1])
#
# images = neb.create_images(minima_a, minima_b, 5)
# images = neb.ImageSet(images)
# start_pos = images.get_image_position_2D_array()
#
# xy = start_pos
# xx = xy[:,0]
# yy = xy[:,1]

c1 = 1.
c2 = 1.
gamma = 1.1
epsilon = 0.01

sv_val_2d = []
sv_test_2d = []
sk_test_2d = []
sk_val_2d = []

kernel_2d_gpr = Kernels.RBF([1., 1.])*Kernels.ConstantKernel()
sv_test_2d.append(sv.GPR(Kernels.ConstantKernel()*kernel_2d_gpr))
sv_test_2d.append(sv.RLS(Kernels.RBF()))

sk_test_2d.append(gpr.GaussianProcessRegressor(gpr.kernels.RBF([1., 1.]) * gpr.kernels.ConstantKernel(), normalize_y=True))
sk_test_2d.append(svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon))
##
for element in sk_test_2d:
    element.fit(xy, energy(xx, yy).reshape(-1))
    sk_val_2d.append((element.__class__.__name__, element.predict(xy_pred)))

grad_x, grad_y = gradient(xx, yy)
grad = np.concatenate([grad_x.reshape(-1,1), grad_y.reshape(-1, 1)], axis=1)

for element in sv_test_2d:
    # element.fit(xy, energy(xx, yy).reshape(-1))  # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    element.fit(xy, energy(xx, yy).reshape(-1), x_prime_train=xy,  y_prime_train=grad)
    sv_val_2d.append((element.__class__.__name__, element.predict(xy_pred)))


def fixZorderFactory_scatter(zorder_in):
    class FixZorderScatter(Patch3DCollection):
        _zorder = zorder_in
        @property
        def zorder(self):
            return self._zorder
        @zorder.setter
        def zorder(self, value):
            pass
    return FixZorderScatter


def fixZorderFactory_surface(zorder_in):
    class FixZorderSurface(Poly3DCollection):
        _zorder = zorder_in
        @property
        def zorder(self):
            return self._zorder
        @zorder.setter
        def zorder(self, value):
            pass
    return FixZorderSurface


def fixZorderFactory_wireframe(zorder_in):
    class FixZorderWireFrame(Line3DCollection):
        _zorder = zorder_in
        @property
        def zorder(self):
            return self._zorder
        @zorder.setter
        def zorder(self, value):
            pass
    return FixZorderWireFrame


alpha = 0.5
rot_angle = 50. #rot_angle = 45.
azim = -65.#azim = -65.
upper_limit = 3.
lower_limit = -1.5


def plot_3D(axis, x_plot, y_plot, z_plot, true_function, x_fit, y_fit, title):
    axis.plot_surface(x_plot, y_plot, true_function(x_plot, y_plot))
    axis.collections[-1].__class__ = fixZorderFactory_surface(0)
    axis.plot_wireframe(x_plot, y_plot, z_plot, color='y', alpha=alpha)
    axis.collections[-1].__class__ = fixZorderFactory_wireframe(1)
    axis.scatter(x_fit, y_fit, energy(x_fit, y_fit), color='k')
    axis.collections[-1].__class__ = fixZorderFactory_scatter(5)
    axis.elev = rot_angle
    axis.azim = azim
    axis.set_title(title)


def plot_2D(axis, x_plot, y_plot, z_plot, x_fit, y_fit, levels, title):
    axis.contourf(x_plot, y_plot, z_plot, levels=levels)
    axis.plot(x_fit, y_fit, ls='None', marker='x', color='k', label='Fitting Points')
    axis.set_title(title)
    axis.legend()


fig = plt.figure()

plot_3D(fig.add_subplot(221, projection='3d'), xx_pred, yy_pred, sk_val_2d[0][1].reshape(-1, n_pred), energy, xx, yy, sk_val_2d[0][0])
plot_3D(fig.add_subplot(222, projection='3d'), xx_pred, yy_pred, sk_val_2d[1][1].reshape(-1, n_pred), energy, xx, yy, sk_val_2d[1][0])
plot_3D(fig.add_subplot(223, projection='3d'), xx_pred, yy_pred, sv_val_2d[0][1].reshape(-1, n_pred), energy, xx, yy, sv_val_2d[0][0])
plot_3D(fig.add_subplot(224, projection='3d'), xx_pred, yy_pred, sv_val_2d[1][1].reshape(-1, n_pred), energy, xx, yy, sv_val_2d[1][0])

lvl = np.linspace(lower_limit, upper_limit, 50)
fig_2D = plt.figure()
plot_2D(fig_2D.add_subplot(231), xx_pred, yy_pred, sk_val_2d[0][1].reshape(-1, n_pred), xx, yy, lvl, sk_val_2d[0][0])
plot_2D(fig_2D.add_subplot(232), xx_pred, yy_pred, sk_val_2d[1][1].reshape(-1, n_pred), xx, yy, lvl, sk_val_2d[1][0])
plot_2D(fig_2D.add_subplot(233), xx_pred, yy_pred, sv_val_2d[0][1].reshape(-1, n_pred), xx, yy, lvl, sv_val_2d[0][0])
plot_2D(fig_2D.add_subplot(234), xx_pred, yy_pred, sv_val_2d[1][1].reshape(-1, n_pred), xx, yy, lvl, sv_val_2d[1][0])
plot_2D(fig_2D.add_subplot(235), xx_pred, yy_pred, energy(xx_pred, yy_pred), xx, yy, lvl, 'True')


plt.show()



# Gamma = np.linspace(0.01,1.,10)
# fig = plt.figure()
# for element in Gamma:
#     # print(sk_test.get_params(deep=True))
#     sk_test.gamma = element
#     scores = model_selection.cross_val_score(sv.RLS(copy.deepcopy(Kernels.RBF(gamma = element))), x, energy_1D(x).reshape(-1), cv=3)
#     sk_test.fit(x, energy_1D(x).reshape(-1))
#     y_predicted = sk_test.predict(x_predict.reshape(-1,1))
#     string = 'gamma = ' + str(element) + '  score sum = ' + str(np.sum(scores))
#     plt.plot(x_predict, y_predicted, label=string)
#     # print(np.max(scores))
#     # print(sk_test.get_params(deep=True))
# plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k')
# plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)
#
# plt.legend()
# plt.show()