"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model(object):
    def __init__(self, batch_size, kernel_name="linear", C=1):

        self.x_input = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.y_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prediction_grid = tf.placeholder(shape=[None, 784], dtype=tf.float32)

        self.A = tf.Variable(tf.zeros(shape=[784,1]))
        self.b = tf.Variable(tf.zeros(shape=[1,1]))

        self.model_output = tf.subtract(tf.matmul(self.x_input, self.A), self.b)

        l2_norm = tf.reduce_sum(tf.square(self.A))

        constant = tf.constant((1 / (batch_size * C)), dtype=tf.float32)

        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(self.y_input, self.model_output))))
        
        self.loss = tf.add(classification_term, tf.multiply(constant, l2_norm))

        self.prediction = tf.sign(self.model_output)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y_input), dtype=tf.float32))

        """

        Kernelized Version

        # Implementation built from:
        # https://github.com/nfmcclure/tensorflow_cookbook/blob/master/04_Support_Vector_Machines/04_Working_with_Kernels/04_svm_kernels.py

        self.x_input = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.y_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prediction_grid = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(shape=[1,batch_size]), constraint=lambda x: tf.clip_by_value(x, 0, C))
        self.gamma = tf.constant(1, dtype=tf.float32)
        constant = tf.constant(0.5, dtype=tf.float32)

        self.my_kernel = Model._linear_kernel(self.x_input) if kernel_name == "linear" else Model._gaussian_kernel(self.x_input, self.gamma)

        # Dual Problem
        self.model_output = tf.matmul(self.b, self.my_kernel)
        self.first_term = tf.reduce_sum(self.b)
        self.b_vec_cross = tf.matmul(tf.transpose(self.b), self.b)
        self.y_target_cross = tf.matmul(self.y_input, tf.transpose(self.y_input))
        self.second_term = tf.multiply(constant, tf.reduce_sum(tf.multiply(self.my_kernel, tf.multiply(self.b_vec_cross, self.y_target_cross))))
        self.loss = tf.subtract(self.second_term, self.first_term) # This is the negative of the dual problem in order to minimize

        self.pred_kernel = Model._linear_pred_kernel(self.x_input, self.prediction_grid) if kernel_name == "linear" else Model._gaussian_pred_kernel(self.x_input, self.prediction_grid, self.gamma)

        # Output stuff
        prediction_output = tf.matmul(tf.multiply(tf.transpose(self.y_input),self.b),self.pred_kernel)
        self.prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.prediction),tf.squeeze(self.y_input)), tf.float32))
        """

    @staticmethod
    def _linear_kernel(x_data):
        return tf.matmul(x_data, tf.transpose(x_data))

    @staticmethod
    def _gaussian_kernel(x_data, gamma):
        dist = tf.reduce_sum(tf.square(x_data), 1)
        dist = tf.reshape(dist, [-1,1])
        sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data,tf.transpose(x_data)))), tf.transpose(dist))
        return tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    @staticmethod
    def _linear_pred_kernel(x_data, prediction_grid):
        return tf.matmul(x_data, tf.transpose(prediction_grid))

    @staticmethod
    def _gaussian_pred_kernel(x_data, prediction_grid, gamma):
        rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
        rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data,tf.transpose(prediction_grid)))), tf.transpose(rB))
        return tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))