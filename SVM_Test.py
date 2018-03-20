# import tensorflow as tf
# import NEB as neb
import numpy as np
# import optimize as opt
# from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SVM as sv
import sklearn as sk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Patch3DCollection
from sklearn import svm

offset = 3
amplitude = 2
modification = 1/2.

def energy_1D(x):
    # return np.sinc(x)
    return amplitude*np.sin(modification*x) + offset


def gradient_1D(x):
    # return np.cos(x)/x+np.sinc(x)/x
    return modification*amplitude*np.cos(x*modification)

# Todo simple test only derivatives missing
# Todo test NEB pes surface
# Todo test NEB with real molecule
# Todo Clean up
gamma = 0.08
# c1 = 1.  # 0**10.
# c2 = 1.  # 0**10.
# c1 = 10 ** 10.
# c2 = 10**10.
# c1 = [1., 10**10]
# c2 = [1., 10**10]
# epsilon = [0.0, 0.1]
epsilon = [0.0001]#, 0.1]
# c1 = [10**5,10**5,10**5,10**5]#[10**5]
# c2 = [10**5,10**5,10**5,10**5]
# c2 = [0,0,0,0]
c1 = [1.]
c2 = [1.]

# seed = 123
# np.random.seed(seed)
for ii in epsilon:
    # method = ['simple', 'irwls', 'rls']
    method = ['irwls']
    for cc, vv in zip(c1, c2):
        # pint('epsilon = ' + str(ii))
        # print('c1 = ' + str(cc) +' c2 = ' + str(vv))

        sv_test = []
        for element in method:
            sv_test.append(sv.SVM(kernel='rbf', method=element, gamma=gamma, epsilon=ii, epsilon_beta=ii))
        sk_test = svm.SVR(C=cc, kernel='rbf', gamma=gamma, epsilon=ii)


        x_predict = np.linspace(-10*np.pi, 10*np.pi, 1000).reshape(-1,1)
        # x = np.arange(-5.5*np.pi+np.pi*3.2/2., 5.5*np.pi+np.pi*3.2/2., np.pi)#
        # x = np.linspace(-8 * np.pi, 8 * np.pi, 50).reshape(-1, 1)
        # x = np.array([np.linspace(np.pi, 1.8*np.pi,3), np.linspace(-np.pi, -1.8*np.pi, 3), np.linspace(2.5*np.pi, 3.8*np.pi,3)])
        # x = np.linspace(0*np.pi, 2*np.pi, 50)
        # x = (np.random.rand(10)-0.5)*np.pi*2
        x = (np.random.rand(111) - 0.5) * np.pi * 2
        # x = np.array([1,3])#,-2])#*0.3+1.5
        # x = np.array([np.linspace(-2, -1, 2),np.linspace(1, 2, 2)])
        # x = x_predict
        x = x.reshape(-1, 1)
        sk_test.fit(x, energy_1D(x).reshape(-1))
        print(len(sk_test.support_))
        print(np.max(sk_test.dual_coef_))

        sv_val = []
        sk_val = sk_test.predict(x_predict)
        # print(sk_test.support_vectors_)
        for element in sv_test:
            print(element.method)
            # element.fit(x, energy_1D(x).reshape(-1), C1=cc, C2=vv) # np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1)
            # sv_val.append(element.predict(x_predict))
            # print(element.alpha)
            # a = element.alpha
            # b = element.beta
            # intercept = element.intercept
            # element.fit(np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1), C1=cc, C2=vv)
            element.fit(x, energy_1D(x).reshape(-1), x_prime=x, y_prime=gradient_1D(x).reshape(-1, 1), C1=cc, C2=vv)
            # element.alpha = a
            # element.beta = np.zeros([5,1])
            # element.intercept = intercept
            sv_val.append(element.predict(x_predict))
            # print(element.alpha)
            # print('b = ' + str(element.intercept))
            x_support = x[element.support_index_alpha]
            # der_support = x[element.support_index_beta]
            # print('alpha = ' + str(element.alpha[element.support_index_alpha]))
            # print('N alpha = ' + str(len(element.support_index_alpha)))
            # print('beta = ' + str(element.beta[element.support_index_beta]))
            # print('N beta = ' + str(np.shape(element.support_index_beta)))
            # print(element.support_index_alpha)
            # print(np.max(element.alpha))
            print('N alpha = ' + str(len(element.support_index_alpha)) + ' N beta = ' + str(np.shape(element.support_index_beta)) + ' b = ' + str(element.intercept))

        # fig = plt.figure()
        # plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
        # plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points')
        # ii = 0
        # for element in debug_y:
        #     plt.plot(x_predict, element, label=str(ii))
        #     ii += 1
        # plt.legend()
        # plt.show()

        color = ['g', 'b', 'r']
        fig = plt.figure()
        plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
        plt.plot(x_predict, energy_1D(x_predict)+epsilon, color='k', ls='--', alpha=0.4)
        plt.plot(x_predict, energy_1D(x_predict)-epsilon, color='k', ls='--', alpha=0.4)
        plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points', alpha=0.4)
        for jj in range(len(sv_test)):
            plt.plot(x_predict, sv_val[jj], color=color[jj], label=method[jj])
            plt.plot(x_support, energy_1D(x_support), ls='None', color='r', marker='o')
            # plt.plot(der_support, energy_1D(der_support), ls='None', color='b', marker='x')
        plt.title('epsilon = ' + str(ii) + ' c1 = ' + str(cc) + ' c2 = ' + str(vv))
        # for jj in range(len(sv_val)):
        #     plt.plot(x_predict, sv_val[jj], color=color[jj], label=str(jj))
        plt.plot(x_predict, sk_val, color='y', label='scikit', ls='--')
        plt.legend()
        plt.show()
        # print('-----------------------------------------------------------')

########################################################################################################################
########################################################################################################################
# ########################################################################################################################
# n = 3
# grid = 3.
# # xx, yy = np.meshgrid(np.linspace(-grid, grid, n), np.linspace(-grid, grid, n))
# xy = np.random.uniform(size = (20, 2), low = -3.0, high = 3.0)
# xx = xy[:,0]
# yy = xy[:,1]
# xy = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)], axis=1)
#
# n_pred = 30
# grid_pred = 7.5
# xx_pred, yy_pred = np.meshgrid(np.linspace(-grid_pred, grid_pred, n_pred), np.linspace(-grid_pred, grid_pred, n_pred))
# xy_pred = np.concatenate([xx_pred.reshape(-1,1), yy_pred.reshape(-1,1)], axis=1)
#
# c1 = 1.
# c2 = 1.
# gamma = 0.1
# epsilon = 0.1
# # method = 'irwls'
# sv_val_2d = []
# sv_test_2d = []
# for element in method:
#     sv_test_2d.append(sv.SVM(kernel='rbf', method=element, gamma=gamma, epsilon=epsilon, epsilon_beta=epsilon))
# sk_test_2d = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)
# ##
# sk_test_2d.fit(xy, energy(xx, yy).reshape(-1))
# grad_x, grad_y = gradient(xx, yy)
# grad = np.concatenate([grad_x.reshape(-1,1), grad_y.reshape(-1, 1)], axis=1)
#
# for element in sv_test_2d:
#     element.fit(xy, energy(xx, yy).reshape(-1), C1=c1, C2=c2) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
#     sv_val_2d.append(element.predict(xy_pred).reshape(-1,n_pred))
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
# ax_2d_c = fig_2d.add_subplot(223, projection='3d')
# ax_2d_c.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
# ax_2d_c.collections[-1].__class__ = fixZorderFactory_surface(0)
# ax_2d_c.plot_wireframe(xx_pred, yy_pred, sv_val_2d[1], label=method,  color='y', alpha=alpha)
# ax_2d_c.collections[-1].__class__ = fixZorderFactory_wireframe(1)
# ax_2d_c.scatter(xx, yy, energy(xx, yy), color='k')
# ax_2d_c.collections[-1].__class__ = fixZorderFactory_scatter(5)
# ax_2d_c.elev = rot_angle
# ax_2d_c.azim = azim
# plt.title(method[1])
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
#
# plt.show()
#
#
# fig_2d_contour = plt.figure()
#
# ax_contour_a = fig_2d_contour.add_subplot(221)
# a = ax_contour_a.contourf(xx_pred, yy_pred, energy(xx_pred, yy_pred), levels = np.linspace(-1.5, 1.5, 30))
# fig_2d_contour.colorbar(a, ax=ax_contour_a)
# ax_contour_a.scatter(xx, yy, marker= 'o', color='k')
#
# plt.title('scikit')
#
# ax_contour_b = fig_2d_contour.add_subplot(222)
# b = ax_contour_b.contourf(xx_pred, yy_pred, sv_val_2d[0], levels=np.linspace(-1.5, 1.5, 30))
# fig_2d_contour.colorbar(b, ax=ax_contour_b)
# ax_contour_b.scatter(xx, yy, marker='o', color='k')
# plt.title(method[0])
#
# ax_contour_c = fig_2d_contour.add_subplot(223)
# c= ax_contour_c.contourf(xx_pred, yy_pred, sv_val_2d[1], levels=np.linspace(-1.5, 1.5, 30))
# fig_2d_contour.colorbar(c, ax=ax_contour_c)
# ax_contour_c.scatter(xx, yy, marker='o', color='k')
# plt.title(method[1])
#
# ax_contour_d = fig_2d_contour.add_subplot(224)
# d = ax_contour_d.contourf(xx_pred, yy_pred, sv_val_2d[2], levels=np.linspace(-1.5, 1.5, 30))
# fig_2d_contour.colorbar(d, ax=ax_contour_d)
# ax_contour_d.scatter(xx, yy, marker='o', color='k')
# plt.title(method[2])
#
# plt.show()