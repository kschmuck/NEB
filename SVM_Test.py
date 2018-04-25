# import tensorflow as tf
# import NEB as neb
import numpy as np
# import optimize as opt
from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import SVM as sv
import MLDerivative as sv
import sklearn as sk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Patch3DCollection
from sklearn import svm, gaussian_process
import Kernels


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
# c1 = 1.  # 0**10.
# c2 = 1.  # 0**10.
# c1 = 10 ** 10.
# c2 = 10**10.
# c1 = [1., 10**10]
# c2 = [1., 10**10]
# epsilon = [0.0, 0.1]
epsilon = [0.01]#, 0.1]
# c1 = [10**5,10**5,10**5,10**5]#[10**5]
# c2 = [10**5,10**5,10**5,10**5]
# c2 = [0,0,0,0]
c1 = [1.]
c2 = [1.]

seed = 123
np.random.seed(seed)

kernel = Kernels.newRBFGrad()
for ii in epsilon:
    # method = ['simple', 'irwls', 'rls']
    method = ['rls_new Kernel', 'rls old Kernel', 'gpr']#, 'gpr']
    # method = ['gpr']
    for cc, vv in zip(c1, c2):
        sv_test = []
        # for element in method:
        #     sv_test.append(sv.IRWLS(kernel='rbf', gamma=gamma))
        # sv_test.append(sv.IRWLS(kernel))
        sv_test.append(sv.RLS(kernel))
        sv_test.append(sv.RLS(Kernels.RBF()))
        # sv_test.append(sv.GPR(kernel))

        # sv_test.append(sv.IRWLS(kernel='rbf', gamma=gamma))
        # sv_test.append(sv.RLS(kernel='rbf', gamma=gamma))
        # #
        sk_test = svm.SVR(C=cc, kernel='rbf', gamma=gamma, epsilon=ii)
        sk_test_2 = gaussian_process.GaussianProcessRegressor()

        x_predict = np.linspace(-10*np.pi, 10*np.pi, 1000).reshape(-1,1)
        # x = np.arange(-5.5*np.pi+np.pi*3.2/2., 5.5*np.pi+np.pi*3.2/2., np.pi)#
        # x = np.linspace(-8 * np.pi, 8 * np.pi, 50).reshape(-1, 1)
        # x = np.array([np.linspace(np.pi, 1.8*np.pi,3), np.linspace(-np.pi, -1.8*np.pi, 3), np.linspace(2.5*np.pi, 3.8*np.pi,3)])
        # x = np.linspace(0*np.pi, 2*np.pi, 50)
        # x = (np.random.rand(20)-0.5)*np.pi*10
        # print(np.random.get_state())
        x = (np.random.rand(150) - 0.5) * np.pi * 18
        x_shift = 0#np.pi*0.5
        # x = np.array([1,3])#,-2])#*0.3+1.5
        # x = np.array([np.linspace(-2, -1, 2),np.linspace(1, 2, 2)])
        # x = x_predict
        x = x.reshape(-1, 1)

        # fig_test = plt.figure()
        # plt.plot(x_predict, energy_1D(x_predict))
        # plt.plot(x, energy_1D(x), marker='o', ls='None')
        # plt.plot(x + x_shift, energy_1D(x + x_shift), marker='o', ls='None')
        sk_test.fit(x, energy_1D(x).reshape(-1))
        np.random.seed()
        sv_val = []
        sk_val = sk_test.predict(x_predict)
        sv_grad_val = []
        sk_test_2.kernel = gaussian_process.kernels.ConstantKernel(1.)*gaussian_process.kernels.RBF(1.) + gaussian_process.kernels.ConstantKernel(np.mean(energy_1D(x)))
        sk_test_2.normalize_y = True
        sk_test_2.fit(x, energy_1D(x).reshape(-1), )
        sk_val, sk_std = sk_test_2.predict(x_predict, return_cov=True)
        # print(np.exp(sk_test_2.kernel_.theta))
        # sk_test_2.kernel
        for element in sv_test:
            # element.fit(x, energy_1D(x).reshape(-1))#, noise=10**-10) # np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1)
            # element.fit(np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1), C1=cc, C2=vv)
            element.fit(x, energy_1D(x).reshape(-1), x_prime_train=x+x_shift, y_prime_train=gradient_1D(x+x_shift).reshape(-1, 1))#10**3)#), C1=cc, C2=vv)#, epsilon=ii, C1=cc, C2=vv, eps=10**-6, max_iter=10**4, error_cap=10**-8)
            # print(element.kernel.hyper_parameter)
            sv_val.append(element.predict(x_predict))

            # sv_grad_val.append(element.predict_derivative(x_predict))
            # print(element.covariance(x))
            print('max alpha = ' + str(np.max(element._alpha)))

            # print('N alpha = ' + str(len(element._support_index_alpha)) + ' N beta = ' + str(np.shape(element._support_index_beta)) + ' b = ' + str(element._intercept))

        color = ['g', 'b', 'r']
        fig = plt.figure()
        plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
        # plt.plot(x_predict, energy_1D(x_predict)+ii, color='k', ls='--', alpha=0.4)
        # plt.plot(x_predict, energy_1D(x_predict)-ii, color='k', ls='--', alpha=0.4)
        plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)
        plt.fill_between(x_predict.reshape(-1), sk_val.reshape(-1) + 2. * np.sqrt(np.diag(sk_std)), sk_val.reshape(-1) - 2. * np.sqrt(np.diag(sk_std)), facecolor=[0.7539, 0.62890625, 0.89453125, .5],
                         linewidths=0.0)

        # var = np.array(np.sqrt(sv_val[0][2])).reshape(-1)
        # plt.plot(x_predict, sv_val[0], color=color[0], label=method[0])
        # plt.plot(x_predict, sv_val[1][0], color=color[1], label=method[1])
        for jj in range(len(sv_test)):
            # plt.plot(x_predict, sv_val[jj][0], color=color[jj], label=method[jj])
            plt.plot(x_predict, sv_val[jj], color=color[jj], label=method[jj])

            # plt.fill_between(x_predict.reshape(-1), sv_val[jj][0].reshape(-1)+ 2. * var, sv_val[jj][0].reshape(-1) - 2. * var, facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0.0)
        #     # plt.plot(x_support, energy_1D(x_support), ls='None', color='r', marker='o')
        #     # plt.plot(der_support, energy_1D(der_support), ls='None', color='b', marker='x')
        # plt.title('epsilon = ' + str(ii) + ' c1 = ' + str(cc) + ' c2 = ' + str(vv))
        plt.plot(x_predict, sk_val, color='y', label='scikit', ls='--')
        plt.legend()

        # color = ['g', 'b', 'r']
        # fig = plt.figure()
        # plt.plot(x_predict, gradient_1D(x_predict), ls='--', color='k', label='True Path')
        # plt.plot(x_predict, gradient_1D(x_predict)+ii, color='k', ls='--', alpha=0.4)
        # plt.plot(x_predict, gradient_1D(x_predict)-ii, color='k', ls='--', alpha=0.4)
        # plt.plot(x, gradient_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)
        # for jj in range(len(sv_test)):
        #     plt.plot(x_predict, sv_val[jj], color=color[jj], label=method[jj])
        #     # plt.plot(x_support, energy_1D(x_support), ls='None', color='r', marker='o')
        #     # plt.plot(der_support, energy_1D(der_support), ls='None', color='b', marker='x')
        # plt.title('epsilon = ' + str(ii) + ' c1 = ' + str(cc) + ' c2 = ' + str(vv))
        # # plt.plot(x_predict, sk_val, color='y', label='scikit', ls='--')
        # plt.legend()
        plt.show()
#         # print('-----------------------------------------------------------')

# ########################################################################################################################
# ########################################################################################################################
# # ########################################################################################################################
# n = 3
# grid = 3.
# xx, yy = np.meshgrid(np.linspace(-grid, grid, n), np.linspace(-grid, grid, n))
# xy = np.random.uniform(size=(100, 2), low = -5., high = 5.)
# xx = xy[:,0]
# yy = xy[:,1]
#
# # xx = np.concatenate([np.linspace(3,-2, 7), np.array([-0.3, -1])])
# # yy = np.concatenate([np.linspace(1,2,7), np.array([-0.25,-1])])
# # xy = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)], axis=1)
#
# n_pred = 30
# grid_pred = 7.5
# xx_pred, yy_pred = np.meshgrid(np.linspace(-grid_pred, grid_pred, n_pred), np.linspace(-grid_pred, grid_pred, n_pred))
# xy_pred = np.concatenate([xx_pred.reshape(-1,1), yy_pred.reshape(-1,1)], axis=1)
#
# c1 = 10000.
# c2 = 10000.
# gamma = 1.1
# epsilon = 0.01
# # method = 'irwls'
# sv_val_2d = []
# sv_test_2d = []
# # for element in method:
# #     sv_test_2d.append(sv.IRWLS(kernel='rbf',  gamma=gamma))
# sv_test_2d.append(sv.GPR(kernel))
# # sv_test_2d.append(sv.IRWLS(kernel='rbf',  gamma=gamma))
# # sv_test_2d.append(sv.RLS(kernel='rbf',  gamma=gamma))
# sk_test_2d = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)
# ##
# sk_test_2d.fit(xy, energy(xx, yy).reshape(-1))
# grad_x, grad_y = gradient(xx, yy)
# grad = np.concatenate([grad_x.reshape(-1,1), grad_y.reshape(-1, 1)], axis=1)
#
# for element in sv_test_2d:
#     # element.fit(xy, energy(xx, yy).reshape(-1), x_prime_train=xy,  y_prime_train=grad) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
#     element.fit(xy, energy(xx, yy).reshape(-1))  # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
#     val, covar, var = element.predict(xy_pred)
#     sv_val_2d.append(val.reshape(-1, n_pred))
#     # element.fit(xy, energy(xx, yy).reshape(-1), x_prime_train=xy,  y_prime_train=grad, C1=c1, C2=c2) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
#     # element.fit(xy, energy(xx, yy).reshape(-1), C1=c1,C2=c2)  # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
#     # sv_val_2d.append(element.predict(xy_pred).reshape(-1,n_pred))
#
# sk_val_2d = sk_test_2d.predict(xy_pred)
#
#
# def fixZorderFactory_scatter(zorder_in):
#     class FixZorderScatter(Patch3DCollection):
#         _zorder = zorder_in
#         @property
#         def zorder(self):
#             return self._zorder
#         @zorder.setter
#         def zorder(self, value):
#             pass
#     return FixZorderScatter
#
#
# def fixZorderFactory_surface(zorder_in):
#     class FixZorderSurface(Poly3DCollection):
#         _zorder = zorder_in
#         @property
#         def zorder(self):
#             return self._zorder
#         @zorder.setter
#         def zorder(self, value):
#             pass
#     return FixZorderSurface
#
#
# def fixZorderFactory_wireframe(zorder_in):
#     class FixZorderWireFrame(Line3DCollection):
#         _zorder = zorder_in
#         @property
#         def zorder(self):
#             return self._zorder
#         @zorder.setter
#         def zorder(self, value):
#             pass
#     return FixZorderWireFrame
#
#
# alpha = 0.5
# rot_angle = 50. #rot_angle = 45.
# azim = -65.#azim = -65.
# fig_2d = plt.figure()
# ax_2d_a = fig_2d.add_subplot(221, projection='3d')
# upper_limit = 3.
# lower_limit = -1.5
#
# ax_2d_a.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# ax_2d_a.collections[-1].__class__ = fixZorderFactory_surface(0)
# ax_2d_a.plot_wireframe(xx_pred, yy_pred, sk_val_2d.reshape(-1,n_pred), label='scikit', color='y', alpha=alpha)
# ax_2d_a.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# ax_2d_a.scatter(xx, yy, energy(xx, yy), color='k')
# ax_2d_a.collections[-1].__class__ = fixZorderFactory_scatter(5)
# ax_2d_a.elev = rot_angle
# ax_2d_a.azim = azim
# plt.title('scikit')
# # ax_2d_a.scatter(xx, yy, energy(xx, yy), color='k')
#
# ax_2d_b = fig_2d.add_subplot(222, projection='3d')
# ax_2d_b.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# ax_2d_b.collections[-1].__class__ = fixZorderFactory_surface(0)
# ax_2d_b.plot_wireframe(xx_pred, yy_pred, sv_val_2d[0], label=method,  color='y', alpha=alpha)
# ax_2d_b.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# ax_2d_b.scatter(xx, yy, energy(xx, yy), color='k')
# ax_2d_b.collections[-1].__class__ = fixZorderFactory_scatter(5)
# ax_2d_b.elev = rot_angle
# ax_2d_b.azim = azim
# plt.title(method[0])
# # ax_2d_b.scatter(xx, yy, energy(xx, yy), color='k')
#
# # ax_2d_c = fig_2d.add_subplot(223, projection='3d')
# # ax_2d_c.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# # ax_2d_c.collections[-1].__class__ = fixZorderFactory_surface(0)
# # ax_2d_c.plot_wireframe(xx_pred, yy_pred, sv_val_2d[1], label=method,  color='y', alpha=alpha)
# # ax_2d_c.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# # ax_2d_c.scatter(xx, yy, energy(xx, yy), color='k')
# # ax_2d_c.collections[-1].__class__ = fixZorderFactory_scatter(5)
# # ax_2d_c.elev = rot_angle
# # ax_2d_c.azim = azim
# # plt.title(method[1])
# #
# # ax_2d_d = fig_2d.add_subplot(224, projection='3d')
# # ax_2d_d.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# # ax_2d_d.collections[-1].__class__ = fixZorderFactory_surface(0)
# # ax_2d_d.plot_wireframe(xx_pred, yy_pred, sv_val_2d[2], label=method,  color='y', alpha=alpha)
# # ax_2d_d.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# # ax_2d_d.scatter(xx, yy, energy(xx, yy), color='k')
# # ax_2d_d.collections[-1].__class__ = fixZorderFactory_scatter(5)
# # ax_2d_d.elev = rot_angle
# # ax_2d_d.azim = azim
# # plt.title(method[2])
#
# # plt.show()
#
#
# fig_2d_contour = plt.figure()
# levels = 50
#
# ax_contour_a = fig_2d_contour.add_subplot(221)
# a = ax_contour_a.contourf(xx_pred, yy_pred, sk_val_2d.reshape(-1,n_pred), levels = np.linspace(lower_limit, upper_limit, levels))
# fig_2d_contour.colorbar(a, ax=ax_contour_a)
# # ax_contour_a.scatter(xx, yy, marker= 'o', color='k')
# plt.title('scikit')
#
# ax_contour_b = fig_2d_contour.add_subplot(222)
# b = ax_contour_b.contourf(xx_pred, yy_pred, sv_val_2d[0], levels=np.linspace(lower_limit, upper_limit, levels))
# fig_2d_contour.colorbar(b, ax=ax_contour_b)
# # ax_contour_b.scatter(xx, yy, marker='o', color='k')
# plt.title(method[0])
#
# ax_contour_c = fig_2d_contour.add_subplot(224)
# c= ax_contour_c.contourf(xx_pred, yy_pred, energy(xx_pred, yy_pred), levels=np.linspace(lower_limit, upper_limit, levels))
# fig_2d_contour.colorbar(c, ax=ax_contour_c)
# # ax_contour_c.scatter(xx, yy, marker='o', color='k')
# plt.title('True')
# #
# # ax_contour_d = fig_2d_contour.add_subplot(223)
# # d = ax_contour_d.contourf(xx_pred, yy_pred, sv_val_2d[1], levels=np.linspace(lower_limit, upper_limit, levels))
# # fig_2d_contour.colorbar(d, ax=ax_contour_d)
# # # ax_contour_d.scatter(xx, yy, marker='o', color='k')
# # plt.title(method[1])
#
# plt.show()