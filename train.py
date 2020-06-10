"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sklearn.datasets as datasets

import matplotlib.pyplot as plt

from model import Model
# from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
np.random.seed(config['random_seed'])

# TODO GET IT WORKING FOR SVM????
# Set up adversary

# attack = LinfPGDAttack(model, 
#                        config['epsilon'],
#                        config['k'],
#                        config['a'],
#                        config['random_start'],
#                        config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
# SVM won't need a checkpoint just a final result?

# model_dir = config['model_dir']
# if not os.path.exists(model_dir):
#   os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

# SVM case what variables do we need?

# Specific to test dataset
# Implementation built from:
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/04_Support_Vector_Machines/04_Working_with_Kernels/04_svm_kernels.py
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=.5,noise=.1)
y_vals = np.array([1 if y==1 else -1 for y in y_vals])

class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

batch_size = 500

svm_model = Model(batch_size)

# x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# b = tf.Variable(tf.random_normal(shape=[1,batch_size]))

# # Linear Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# # # Gaussian kernel
# # gamma = tf.constant(-50.0)
# # dist = tf.reduce_sum(tf.square(x_data), 1)
# # dist = tf.reshape(dist, [-1,1])
# # sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data,tf.transpose(x_data)))), tf.transpose(dist))
# # my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# print(tf.shape(my_kernel))
# print(tf.shape(b))

# # Dual problem
# model_output = tf.matmul(b, my_kernel)
# first_term = tf.reduce_sum(b)
# b_vec_cross = tf.matmul(tf.transpose(b), b)
# y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
# second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross,y_target_cross)))
# loss = tf.negative(tf.subtract(first_term, second_term))

# # Prediction and accuracy function

# # Linear
# pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

# # # Gaussian
# # rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
# # rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
# # pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data,tf.transpose(prediction_grid)))), tf.transpose(rB))
# # pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b),pred_kernel)
# prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction),tf.squeeze(y_target)), tf.float32))

# Optimizer stuff
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(svm_model.loss)
init = tf.initialize_all_variables()

loss_vec = []
batch_accuracy = []

X = None
Y = None
grid_predictions = []

# Start tensorflow session and start training
sess = tf.Session()
sess.run(init)

for i in range(2000):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  X = x_vals[rand_index]
  Y = np.transpose([y_vals[rand_index]])
  sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
  temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
  loss_vec.append(temp_loss)
  acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X,svm_model.y_input: Y,svm_model.prediction_grid:X})
  batch_accuracy.append(acc_temp)

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                    np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_predictions] = sess.run(svm_model.prediction, feed_dict={svm_model.x_input: X,
                                                  svm_model.y_input: Y,
                                                  svm_model.prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# Evaluate on new/unseen data points
# New data points:
new_points = np.array([(-0.75, -0.75),
                       (-0.5, -0.5),
                       (-0.25, -0.25),
                       (0.25, 0.25),
                       (0.5, 0.5),
                       (0.75, 0.75)])

[evaluations] = sess.run(svm_model.prediction, feed_dict={svm_model.x_input: x_vals,
                                                svm_model.y_input: np.transpose([y_vals]),
                                                svm_model.prediction_grid: new_points})

for ix, p in enumerate(new_points):
    print('{} : class={}'.format(p, evaluations[ix]))

sess.close()
