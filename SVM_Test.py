import tensorflow as tf
import NEB as neb
import numpy as np
import optimize as opt
from pes import gradient, energy, energy_gradient, energy_xy_list, gradient_xy_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SVM as sv
import sklearn as sk
from sklearn import svm


def energy_1D(x):
    # return np.sinc(x)
    return np.sin(x) +0# 10


def gradient_1D(x):
    # return np.cos(x)/x+np.sinc(x)/x
    return np.cos(x)

# Todo simple test
# Todo rls test
# Todo test all for only y, only y_prime and both in 1D and 2D
# Todo test NEB pes surface
# Todo test NEB with real molecule
# Todo Clean up

gamma = 0.1
epsilon = 0.1
method = ['simple', 'irwls', 'rls']
# method = ['simple']
sv_test = []
for element in method:
     sv_test.append(sv.SVM(kernel='rbf', method=element, gamma=gamma, epsilon=epsilon, epsilon_beta=epsilon))
sk_test = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)

x = np.array([np.linspace(np.pi, 1.8*np.pi,3), np.linspace(-np.pi, -1.8*np.pi, 3), np.linspace(2.5*np.pi, 3.8*np.pi,3)])
# x = (np.random.rand(4)-1)*np.pi*6
# x = np.array([1,3,-2])#*0.3+1.5
# x = np.array([np.linspace(-2, -1, 2),np.linspace(1, 2, 2)])
x = x.reshape(-1, 1)
x_predict = np.linspace(-5*np.pi, 8*np.pi, 300).reshape(-1,1)

c1 = 1.
c2 = 1.

sk_test.fit(x, energy_1D(x).reshape(-1))

sv_val = []
sk_val = sk_test.predict(x_predict)

for element in sv_test:
    # element.fit(x, energy_1D(x).reshape(-1), C1=c1, C2=c2) # np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1)
    # element.fit(np.zeros([0,1]), np.zeros(0), x_prime=x, y_prime=gradient_1D(x).reshape(-1), C1=c1, C2=c2)
    element.fit(x, energy_1D(x).reshape(-1), x_prime=x, y_prime=gradient_1D(x).reshape(-1, 1), C1=c1, C2=c2)
    sv_val.append(element.predict(x_predict))

color = ['r', 'b', 'g']
fig = plt.figure()
plt.plot(x_predict, energy_1D(x_predict), ls='--', color='k', label='True Path')
plt.plot(x, energy_1D(x), ls='None', color='k', marker='o', label='Training Points')
for ii in range(len(sv_test)):
    plt.plot(x_predict, sv_val[ii], color=color[ii], label=method[ii])
plt.plot(x_predict, sk_val, color='y', label='scikit')
plt.legend()
plt.show()

########################################################################################################################
########################################################################################################################
# ########################################################################################################################
n = 3
grid = 3.
xx, yy = np.meshgrid(np.linspace(-grid, grid, n), np.linspace(-grid, grid, n))
xy = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)], axis=1)

n_pred = 20
grid_pred = 7.5
xx_pred, yy_pred = np.meshgrid(np.linspace(-grid_pred, grid_pred, n_pred), np.linspace(-grid_pred, grid_pred, n_pred))
xy_pred = np.concatenate([xx_pred.reshape(-1,1), yy_pred.reshape(-1,1)], axis=1)

c1 = 1.
c2 = 1.
gamma = 0.1
epsilon = 0.1
# method = 'irwls'
sv_val_2d = []
sv_test_2d = []
for element in method:
    sv_test_2d.append(sv.SVM(kernel='rbf', method=element, gamma=gamma, epsilon=epsilon, epsilon_beta=epsilon))
sk_test_2d = svm.SVR(kernel='rbf', gamma=gamma, epsilon=epsilon)
##
sk_test_2d.fit(xy, energy(xx, yy).reshape(-1))
grad_x, grad_y = gradient(xx, yy)
grad = np.concatenate([grad_x.reshape(-1,1), grad_y.reshape(-1, 1)], axis=1)

for element in sv_test_2d:
    element.fit(xy, energy(xx, yy).reshape(-1), C1=c1, C2=c2) # np.zeros([0,2]), np.zeros(0), x_prime=xy, y_prime=grad
    sv_val_2d.append(element.predict(xy_pred).reshape(-1,n_pred))

sk_val_2d = sk_test_2d.predict(xy_pred)


alpha = 0.8
fig_2d = plt.figure()
ax_2d_a = fig_2d.add_subplot(221, projection='3d')
ax_2d_a.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_a.plot_wireframe(xx_pred, yy_pred, sk_val_2d.reshape(-1,n_pred), label='scikit', color='g', alpha=alpha)
plt.title('scikit')
# ax_2d_a.scatter(xx, yy, energy(xx, yy), color='k')

ax_2d_b = fig_2d.add_subplot(222, projection='3d')
ax_2d_b.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_b.plot_wireframe(xx_pred, yy_pred, sv_val_2d[0], label=method,  color='r', alpha=alpha)
plt.title(method[0])
# ax_2d_b.scatter(xx, yy, energy(xx, yy), color='k')

ax_2d_c = fig_2d.add_subplot(223, projection='3d')
ax_2d_c.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_c.plot_wireframe(xx_pred, yy_pred, sv_val_2d[1], label=method,  color='y', alpha=alpha)
plt.title(method[1])

ax_2d_d = fig_2d.add_subplot(224, projection='3d')
ax_2d_d.plot_surface(xx_pred, yy_pred, energy(xx_pred, yy_pred), label='True')
ax_2d_d.plot_wireframe(xx_pred, yy_pred, sv_val_2d[2], label=method,  color='k', alpha=alpha)
plt.title(method[2])

plt.show()