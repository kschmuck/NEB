import tensorflow as tf
import NEB as neb
import numpy as np
import optimize as opt
# from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SVM as sv
import sklearn as sk
from sklearn import svm


def energy(x):
    return np.sin(x) + 1


def gradient(x):
    return np.cos(x)


# number_of_images = 10
# k = 10**-1# idpp: 10**-10 #10**-6
# delta_t_fire = 3.5 #3.5
# delta_t_verlete = 0.2## #0.2
# force_max = 0.1
# max_steps = 200
# epsilon = 0.01 #00001
# trust_radius = 0.05 #.1
#
# minima_a = np.array([3, 1])
# minima_b = np.array([-2, 2])
# images = neb.create_images(minima_a, minima_b, number_of_images)
# images = neb.ImageSet(images)
# energy_grad_func = energy_gradient
# images.energy_gradient_func = energy_grad_func
#
# images.update_images('improved')
# data_vector = images.get_image_position_2D_array()
# #
# target_vector = np.array(images.get_image_energy_list())
gamma = 0.4#0.1
epsilon = 0.001
test_derivative = sv.SVM(epsilon=epsilon, epsilon_beta=0.001, gamma=gamma)
test_simple = sv.SVM(epsilon=epsilon, gamma=gamma)
sk_test = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)
#
#
# xx, yy = np.meshgrid(np.linspace(-3., 3., 5), np.linspace(-3., 3., 5))
# xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
# xy = images.get_image_position_2D_array()
# test.fit(data_vector, target_vector.reshape(-1, 1))

# x = np.array([np.linspace(np.pi, 1.8*np.pi,3), np.linspace(-np.pi, -1.8*np.pi, 3), np.linspace(2.5*np.pi, 3.8*np.pi,3)])
x = (np.random.rand(4)-1)*np.pi*4
# x = np.array([1,3,-2])#*0.3+1.5

# x = np.array([np.linspace(-2, -1, 2),np.linspace(1, 2, 2)])

test_derivative.fit(x.reshape(-1,1), np.array(energy(x)).reshape(-1, 1), derivative_target=np.array(gradient(x).reshape(-1,1)), D=10., C=10.)#
test_simple.fit(x.reshape(-1,1), np.array(energy(x).reshape(-1,1)), C=10.)
sk_test.fit(x.reshape(-1,1), np.array(energy(x)).reshape(-1))

x_pred = np.linspace(-5*np.pi, 5*np.pi, 200)

values_derivative = test_derivative.predict_derivative(x_pred.reshape(-1,1))
values = test_derivative.predict(x_pred.reshape(-1,1))-values_derivative

values_simple = test_simple.predict(x_pred.reshape(-1,1))
sk_values = sk_test.predict(x_pred.reshape(-1,1))


fig = plt.figure()
plt.plot(x_pred, energy(x_pred), label='true', color='b')
# plt.plot(x_pred, values_derivative, label='derivative method: derivative')
plt.plot(x_pred, values_simple, label='method: simple',ls='--', color='r')
plt.plot(x_pred, sk_values, label='predicted scikit', ls='--', color='k')
# plt.plot(x_pred, values, label='function values method: derivative')
plt.plot(x_pred, values+values_derivative, label='method: derivative', ls='--', color='g')
plt.plot(x, energy(x), marker='o', ls='None', color='k')
plt.legend()


fig_2 = plt.figure()
ax = fig_2.add_subplot(3,1,1)
ax.plot(x_pred, energy(x_pred), label='true', color='r')
ax.plot(x, energy(x), marker='o', ls='None', color='k')
ax.plot(x_pred, values_derivative, label='derivative method: derivative', color='g')
# ax.plot(x_pred, gradient(x_pred), label='cos', ls='--')

# d_x = test_derivative.predict_derivative(x.reshape(-1,1)) + x
# dx_true = x+gradient(x)
# for ii in range(0, len(x)):
#     ax.arrow(x[ii], energy(x[ii]), d_x[ii], energy(d_x[ii]), head_width=0.25, head_length=0.25, color='r', label='predicted gradient')
#     ax.arrow(x[ii], energy(x[ii]), dx_true[ii], energy(dx_true[ii]), head_width=0.25, head_length=0.25, color='k', label='true gradient')
# ax.legend()

ax_2 = fig_2.add_subplot(3,1,2)
ax_2.plot(x_pred, values, label='function values method: derivative', color='g')
ax_2.plot(x_pred, values_simple, label='method: simple', color='r')
ax_2.plot(x, energy(x), marker='o', ls='None', color='k')
ax_2.plot(x_pred, energy(x_pred), label='true', color='b')
ax_2.legend()

ax_3 = fig_2.add_subplot(3,1,3)
ax_3.plot(x_pred, values+values_derivative, label='derivative+function values method: derivative', color='g')
ax_3.plot(x_pred, energy(x_pred), label='true', color='b')
ax_3.plot(x, energy(x), marker='o', ls='None', color='k')

ax_3.legend()

plt.plot(x_pred, sk_values)


# ax = fig.add_subplot(311, projection='3d')
# ax.plot_surface(xx, yy, energy(xx, yy))
# # ax.plot_surface(xx, yy, values.reshape(len(xx),-1))
# ax.plot_wireframe(xx, yy, values.reshape(len(xx),-1), color='r')
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(xx, yy, sk_values.reshape(len(xx),-1))
# # ax.plot_wireframe(xx, yy, sk_values.reshape(len(xx),-1), color='g')
#
# ax3 = fig.add_subplot(3,1,2, projection='3d')
# ax3.plot_wireframe(xx, yy, sk_values.reshape(len(xx),-1), color='g')
# ax3.plot_wireframe(xx, yy, values.reshape(len(xx),-1), color='r')

# n = 50
# const_xx = np.linspace(-5, 5, n)
# # const_yy = np.zeros(n)+minima_a[1]#np.linspace(-5, 5, n)
# # vary = np.linspace(-4, 4, n)
# xy = np.concatenate([const_xx.reshape(-1,1), const_yy.reshape(-1,1)], axis=1)
# ax1 = fig.add_subplot(3, 1, 3)
# ax1.plot(const_xx, energy(const_xx, const_yy), label='True')
# ax1.plot(const_xx, test.predict(xy), label='with Gradient', color= 'r')
#
# # const_xx = np.linspace(minima_a[0], minima_b[0], n)
# # const_yy = np.linspace(minima_a[1], minima_b[1], n)
# # ax2 = fig.add_subplot(3, 1, 3)
# # ax2.plot(const_xx, test.predict(xy), label='with Gradient ypath minima', color= 'r')
# # ax2.plot(const_xx, energy(const_xx, const_yy), label='True')
#
# ax1.plot(const_xx, sk_test.predict(xy), label='Scikitlearn', color='g')
# ax1.legend()

# plt.show()
# #