import numpy as np
import tensorflow as tf
from tensorflow.contrib import kernel_methods


def euclidean_distance(x, y):
    # x, y are a second order tensor [n_samples, n_features], n_features 3*N atoms
    # normally placeholder are feed into this function
    # x = tf.cast(x, dtype=tf.float32)
    # y = tf.cast(y, dtype=tf.float32)
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)
    xy = tf.multiply(2.0, tf.matmul(x, y, transpose_b=True))
    squared_distance = tf.transpose(tf.add(xx, tf.transpose(tf.subtract(yy, xy))))

    # xx = tf.reduce_sum(tf.square(x), 2)
    # yy = tf.reduce_sum(tf.square(y), 2)
    # xy = tf.reduce_sum(tf.multiply(2.0, tf.matmul(tf.transpose(x, perm=[1, 0, 2]), tf.transpose(y, perm=[1, 2, 0]))), 0)
    # squared_distance = tf.subtract(tf.add(xx, tf.transpose(yy)), xy)

    return squared_distance


class SVM:
    def __init__(self, epsilon=0.1, kernel='rbf', model='linear'): # cost_funct='standart'
        self.epsilon = tf.constant(epsilon)
        self.sess = tf.Session()
        self.fit_input_place = None
        self.fit_target_place = None

        self.b = None
        self.alpha = None

        self.alpha_star = None

        self.target = None
        self.input_value = None

        self.alpha_star = None

        if kernel == 'rbf':
            self.kernel = RBF()

        if model == 'linear':
            self.model_func = self.linear_model

    def fit(self, input_value, target_value, derivative_target=None):
        self.input_value = input_value
        self.target = target_value

        self.fit_input_place = tf.placeholder(shape=input_value.shape, dtype=tf.float32)
        self.fit_target_place = tf.placeholder(shape=target_value.shape, dtype=tf.float32)

        self.alpha = tf.Variable(tf.random_normal(tf.shape(target_value)), dtype=tf.float32)
        self.alpha_star = tf.Variable(tf.random_normal(tf.shape(target_value)), dtype=tf.float32)

        # self.alpha = self.alpha.assign(tf.maximum(0.0, self.alpha))
        # self.alpha_star = tf.transpose(self.alpha)

        # self.b = tf.Variable(tf.random_normal([1,1]), dtype=tf.float32) #
        self.b = tf.constant(0.0)
        kernel = self.kernel.kernel_fit(self.fit_input_place)

        model_fit = self.linear_model(self.fit_target_place, kernel, self.b)
        # cost and model are still wrong
        cost = tf.reduce_mean(tf.maximum(0.0, tf.add(tf.abs(tf.subtract(model_fit, tf.reduce_sum(self.fit_target_place, 1))/ np.size(input_value,0)), self.epsilon)))
        # cost = tf.reduce_sum(tf.pow(tf.subtract(model_fit, tf.reduce_sum(self.fit_target_place,1)), 2)/-(2*np.size(input_value, 0)))
        # model = self.test_model(self.fit_target_place, kernel)
        # cost = tf.reduce_sum(model)

        svm_opt = tf.train.GradientDescentOptimizer(.01)
        train_step = svm_opt.minimize(cost)
        self.sess.run(tf.global_variables_initializer())
        for ii in range(0,10):
            self.sess.run(train_step, feed_dict={self.fit_input_place: input_value, self.fit_target_place: target_value})
            print(self.sess.run(self.alpha))
            print('-------------------')

        return self.sess.run(self.alpha), self.sess.run(self.b)

    def predict(self, predict_value):
        # placeholder_in = tf.placeholder(shape=input_value.shape, dtype=tf.float32)
        # placeholder_out = tf.placeholder(shape=target_value.shape, dtype=tf.float32)
        placeholder__predict = tf.placeholder(shape=predict_value.shape, dtype=tf.float32)
        kernel = self.kernel.kernel_predict(self.fit_input_place, placeholder__predict)
        model_predict = tf.reduce_sum(tf.matmul(self.alpha-self.alpha_star, kernel),1)
        values = self.sess.run(model_predict, feed_dict={self.fit_input_place: self.input_value, self.fit_target_place: self.target, placeholder__predict: predict_value})
        return values

    def predict_dev(self):
        pass

    def linear_model(self, target, kernel, b):
        # TODO model is still wrong --> should be dual problem model
        # model = tf.reduce_sum(tf.add(tf.matmul(tf.matmul(self.alpha, tf.transpose(target)), kernel), b),0)
        # Simple linear Model
        # model = tf.add(tf.matmul(tf.transpose(self.alpha), kernel), b)
        # model = tf.reduce_sum(model, 0)
        # model = tf.reduce_sum(tf.matmul(tf.multiply(tf.transpose(target), b), kernel),0)
        ya = tf.matmul(target, tf.transpose(self.alpha))
        model = tf.subtract(tf.reduce_sum(self.alpha,0),tf.multiply(0.5,tf.reduce_sum(tf.matmul(ya, tf.matmul(tf.transpose(ya), kernel)),0)))
        # print('model shape: '+str(model.shape))
        return model

    # def test_model(self, target, kernel):
    #     term_a = -0.5*tf.reduce_sum(tf.matmul(tf.matmul(self.alpha-self.alpha_star, self.alpha-self.alpha_star), kernel),1)
    #     term_b = tf.reduce_sum(self.epsilon*tf.add(self.alpha, self.alpha_star),1)
    #     term_c = tf.reduce_sum(tf.matmul(self.alpha - self.alpha_star, target),1)
    #     model = term_a + term_b + term_c
    #     return model


class RBF:
    def __init__(self, gamma=-.1):
        self.gamma = tf.constant(gamma)

    def kernel_fit(self, x):
        squared_distance = euclidean_distance(x, x)
        kernel = tf.exp(tf.multiply(-self.gamma, tf.abs(squared_distance)))
        return kernel

    def kernel_predict(self, x, y):
        squared_distance = euclidean_distance(x, y)
        kernel = tf.exp(tf.multiply(-self.gamma, tf.abs(squared_distance)))
        return kernel


