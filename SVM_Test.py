import NEB as neb
import numpy as np
# import optimize as opt
from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list, minimum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import SVM as sv
import MLDerivative as sv
import sklearn as sk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Patch3DCollection
from sklearn import svm, gaussian_process, model_selection
import Kernels
import copy

offset = 3
amplitude = 2
modification = 1/2.

def energy_1D(x):
    # return np.sinc(x)
    return amplitude*np.sin(modification*x) + offset


def gradient_1D(x):
    # return np.cos(x)/x+np.sinc(x)/x
    return modification*amplitude*np.cos(x*modification)


gamma = 0.1
epsilon = 0.01
c1 = 1000000.
c2 = 1.

seed = 123
np.random.seed(seed)

sv_test = []
# sv_test.append(sv.IRWLS(Kernels.RBF()))
sv_test.append(sv.RLS(copy.deepcopy(Kernels.RBF())))
# sv_test.append(sv.GPR(Kernels.newRBFGrad()))

sk_test = svm.SVR(C=c1, kernel='rbf', gamma=gamma, epsilon=epsilon)

# sk_test_2 = gaussian_process.GaussianProcessRegressor()
x_predict = np.linspace(-15*np.pi, 15*np.pi, 2000).reshape(-1,1)
x = (np.random.rand(5) - 0.5) * np.pi * 18
x = x.reshape(-1, 1)

Gamma = np.linspace(0.01,1.,10)
fig = plt.figure()
for element in Gamma:
    # print(sk_test.get_params(deep=True))
    sk_test.gamma = element
    scores = model_selection.cross_val_score(sv.RLS(copy.deepcopy(Kernels.RBF(gamma = element))), x, energy_1D(x).reshape(-1), cv=3)
    sk_test.fit(x, energy_1D(x).reshape(-1))
    y_predicted = sk_test.predict(x_predict.reshape(-1,1))
    string = 'gamma = ' + str(element) + '  score sum = ' + str(np.sum(scores))
    plt.plot(x_predict, y_predicted, label=string)
    # print(np.max(scores))
    # print(sk_test.get_params(deep=True))
plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k')
plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)

plt.legend()
plt.show()

#
# sk_test.fit(x, energy_1D(x).reshape(-1))
# np.random.seed()
# sv_val = []
# sk_val = sk_test.predict(x_predict)
# sv_grad_val = []
# sk_test_2.kernel = gaussian_process.kernels.ConstantKernel(1.)*gaussian_process.kernels.RBF(1.) + gaussian_process.kernels.ConstantKernel(np.mean(energy_1D(x)))
# sk_test_2.normalize_y = True
# sk_test_2.fit(x, energy_1D(x).reshape(-1), )
# sk_val, sk_std = sk_test_2.predict(x_predict, return_cov=True)
# mat = []
# for element in sv_test:
#     element.fit(x, energy_1D(x).reshape(-1))#, noise=10**-10) # np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1)
#     # element.fit(x, energy_1D(x).reshape(-1), x_prime_train=x, y_prime_train=gradient_1D(x).reshape(-1, 1))#10**3)#), C1=cc, C2=vv)#, epsilon=ii, C1=cc, C2=vv, eps=10**-6, max_iter=10**4, error_cap=10**-8)
#     # element.fit(np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1), C1=cc, C2=vv)
#     sv_val.append(element.predict(x_predict))
#
# color = ['g', 'b', 'r']
# fig = plt.figure()
# plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
# plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)
# plt.fill_between(x_predict.reshape(-1), sk_val.reshape(-1) + 2. * np.sqrt(np.diag(sk_std)), sk_val.reshape(-1) - 2. * np.sqrt(np.diag(sk_std)), facecolor=[0.7539, 0.62890625, 0.89453125, .5],
#                  linewidths=0.0)
#
# for jj in range(len(sv_test)):
#     plt.plot(x_predict, sv_val[jj], color=color[jj], label=sv_test[jj].__class__.__name__)
# plt.plot(x_predict, sk_val, color='y', label='scikit', ls='--')
# plt.legend()
# plt.show()

#######################################################################################################################
# #######################################################################################################################
# ########################################################################################################################
n = 3
grid = 3.
# xx, yy = np.meshgrid(np.linspace(-grid, grid, n), np.linspace(-grid, grid, n))
xy = np.random.uniform(size=(10, 2), low = -5., high = 5.)
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
# # images.energy_gradient_func = energy_gradient
# images.set_spring_constant(1)
# # images.update_images('simple_improved')
#
# start_pos = images.get_image_position_2D_array()
# xy = start_pos
# xx = xy[:,0]
# yy = xy[:,1]

c1 = 10000.
c2 = 10000.
gamma = 1.1
epsilon = 0.01

sv_val_2d = []
sv_test_2d = []

sv_test_2d.append(sv.GPR(Kernels.newRBFGrad()))
sv_test_2d.append(sv.RLS(Kernels.RBF()))
sk_test_2d = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)
##
sk_test_2d.fit(xy, energy(xx, yy).reshape(-1))
grad_x, grad_y = gradient(xx, yy)
grad = np.concatenate([grad_x.reshape(-1,1), grad_y.reshape(-1, 1)], axis=1)

for element in sv_test_2d:
    # element.fit(xy, energy(xx, yy).reshape(-1))  # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    element.fit(xy, energy(xx, yy).reshape(-1), x_prime_train=xy,  y_prime_train=grad) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    val = element.predict(xy_pred) #  covar, var

    sv_val_2d.append(val.reshape(-1, n_pred))
    # element.fit(xy, energy(xx, yy).reshape(-1), x_prime_train=xy,  y_prime_train=grad, C1=c1, C2=c2) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    # element.fit(xy, energy(xx, yy).reshape(-1), C1=c1,C2=c2)  # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    # sv_val_2d.append(element.predict(xy_pred).reshape(-1,n_pred))

sk_val_2d = sk_test_2d.predict(xy_pred)

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
fig_2d = plt.figure()
ax_2d_a = fig_2d.add_subplot(221, projection='3d')
upper_limit = 3.
lower_limit = -1.5

ax_2d_a.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_a.collections[-1].__class__ = fixZorderFactory_surface(0)
ax_2d_a.plot_wireframe(xx_pred, yy_pred, sk_val_2d.reshape(-1,n_pred), label='scikit', color='y', alpha=alpha)
ax_2d_a.collections[-1].__class__ = fixZorderFactory_wireframe(1)
ax_2d_a.scatter(xx, yy, energy(xx, yy), color='k')
ax_2d_a.collections[-1].__class__ = fixZorderFactory_scatter(5)
ax_2d_a.elev = rot_angle
ax_2d_a.azim = azim
plt.title('scikit')
# ax_2d_a.scatter(xx, yy, energy(xx, yy), color='k')

ax_2d_b = fig_2d.add_subplot(222, projection='3d')
ax_2d_b.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_b.collections[-1].__class__ = fixZorderFactory_surface(0)
ax_2d_b.plot_wireframe(xx_pred, yy_pred, sv_val_2d[0], label='GPR',  color='y', alpha=alpha)
ax_2d_b.collections[-1].__class__ = fixZorderFactory_wireframe(1)
ax_2d_b.scatter(xx, yy, energy(xx, yy), color='k')
ax_2d_b.collections[-1].__class__ = fixZorderFactory_scatter(5)
ax_2d_b.elev = rot_angle
ax_2d_b.azim = azim
plt.title('GPR')
# ax_2d_b.scatter(xx, yy, energy(xx, yy), color='k')

ax_2d_c = fig_2d.add_subplot(223, projection='3d')
ax_2d_c.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_c.collections[-1].__class__ = fixZorderFactory_surface(0)
ax_2d_c.plot_wireframe(xx_pred, yy_pred, sv_val_2d[1], label='RLS',  color='y', alpha=alpha)
ax_2d_c.collections[-1].__class__ = fixZorderFactory_wireframe(1)
ax_2d_c.scatter(xx, yy, energy(xx, yy), color='k')
ax_2d_c.collections[-1].__class__ = fixZorderFactory_scatter(5)
ax_2d_c.elev = rot_angle
ax_2d_c.azim = azim
plt.title('RLS')
#
# ax_2d_d = fig_2d.add_subplot(224, projection='3d')
# ax_2d_d.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# ax_2d_d.collections[-1].__class__ = fixZorderFactory_surface(0)
# ax_2d_d.plot_wireframe(xx_pred, yy_pred, sv_val_2d[2], label=method,  color='y', alpha=alpha)
# ax_2d_d.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# ax_2d_d.scatter(xx, yy, energy(xx, yy), color='k')
# ax_2d_d.collections[-1].__class__ = fixZorderFactory_scatter(5)
# ax_2d_d.elev = rot_angle
# ax_2d_d.azim = azim
# plt.title(method[2])

# plt.show()


fig_2d_contour = plt.figure()
levels = 50

ax_contour_a = fig_2d_contour.add_subplot(221)
a = ax_contour_a.contourf(xx_pred, yy_pred, sk_val_2d.reshape(-1,n_pred), levels = np.linspace(lower_limit, upper_limit, levels))
fig_2d_contour.colorbar(a, ax=ax_contour_a)
# ax_contour_a.scatter(xx, yy, marker= 'o', color='k')
plt.title('scikit')

ax_contour_b = fig_2d_contour.add_subplot(222)
b = ax_contour_b.contourf(xx_pred, yy_pred, sv_val_2d[0], levels=np.linspace(lower_limit, upper_limit, levels))
fig_2d_contour.colorbar(b, ax=ax_contour_b)
# ax_contour_b.scatter(xx, yy, marker='o', color='k')
plt.title('GPR')

ax_contour_c = fig_2d_contour.add_subplot(224)
c= ax_contour_c.contourf(xx_pred, yy_pred, energy(xx_pred, yy_pred), levels=np.linspace(lower_limit, upper_limit, levels))
fig_2d_contour.colorbar(c, ax=ax_contour_c)
# ax_contour_c.scatter(xx, yy, marker='o', color='k')
plt.title('True')
#
ax_contour_d = fig_2d_contour.add_subplot(223)
d = ax_contour_d.contourf(xx_pred, yy_pred, sv_val_2d[1], levels=np.linspace(lower_limit, upper_limit, levels))
fig_2d_contour.colorbar(d, ax=ax_contour_d)
# ax_contour_d.scatter(xx, yy, marker='o', color='k')
plt.title('RLS')

plt.show()