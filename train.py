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
from scipy import stats

import matplotlib.pyplot as plt

from model import Model
# from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Return mnist data with only 1's and 7's as tensor
def prepare_mnist():

  mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")

  mnist_x_train = mnist[0][0]
  mnist_y_train = mnist[0][1]
  mnist_x_test = mnist[1][0]
  mnist_y_test= mnist[1][1]

  # Extract only ones and sevens

  indices_1_train = mnist_y_train == 1
  indices_1_test = mnist_y_test == 1
  indices_7_train = mnist_y_train == 7
  indices_7_test = mnist_y_test == 7

  mnist_x_train_1 = mnist_x_train[indices_1_train]
  mnist_y_train_1 = mnist_y_train[indices_1_train]
  mnist_x_test_1 = mnist_x_test[indices_1_test]
  mnist_y_test_1 = mnist_y_test[indices_1_test]

  mnist_x_train_7 = mnist_x_train[indices_7_train]
  mnist_y_train_7 = mnist_y_train[indices_7_train]
  mnist_x_test_7 = mnist_x_test[indices_7_test]
  mnist_y_test_7 = mnist_y_test[indices_7_test]

  # Combine numpy arrays into one dataset

  mnist_x_train = np.concatenate((mnist_x_train_1, mnist_x_train_7))
  mnist_y_train = np.concatenate((mnist_y_train_1, mnist_y_train_7))
  mnist_x_test = np.concatenate((mnist_x_test_1, mnist_x_test_7))
  mnist_y_test = np.concatenate((mnist_y_test_1, mnist_y_test_7))

  # Flatten x arrays

  mnist_x_train_flatten = np.empty((mnist_x_train.shape[0], 784))
  for idx, x in enumerate(mnist_x_train):
    mnist_x_train_flatten[idx] = x.flatten()/255

  mnist_x_test_flatten = np.empty((mnist_x_test.shape[0], 784))
  for idx, x in enumerate(mnist_x_test):
    mnist_x_test_flatten[idx] = x.flatten()/255

  # Encode y arrays (1 -> 0 and 7 -> 1)

  mnist_y_train = np.array([1 if y == 7 else -1 for y in mnist_y_train])
  mnist_y_test = np.array([1 if y == 7 else -1 for y in mnist_y_test])

  return (mnist_x_train_flatten, mnist_y_train, mnist_x_test_flatten, mnist_y_test)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
np.random.seed(config['random_seed'])

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

mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")

(mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y) = prepare_mnist()

x_vals = mnist_train_x
y_vals = mnist_train_y

batch_size = 2163

svm_model = Model(batch_size)

# Optimizer stuff
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
# my_opt = tf.train.AdamOptimizer(learning_rate=0.0001) # initialization of zeros (for b) and learning rate of 0.0001 with gradient clipping at -1 and 1 results in ~96% accuracy

# gvs = my_opt.compute_gradients(svm_model.loss)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# train_step = my_opt.apply_gradients(capped_gvs)

train_step = my_opt.minimize(svm_model.loss)
init = tf.initialize_all_variables()

loss_vec = []
batch_accuracy = []

first_vec = []
second_vec = []
Y_7_count = []

X = None
Y = None
grid_predictions = []

# Start tensorflow session and start training
sess = tf.Session()
sess.run(init)

# # Stochastic 
# for i in range(len(x_vals)):

#   # Create data
#   X = np.array(x_vals[i]).reshape(1, 784)
#   Y = np.transpose(np.array(y_vals[i]).reshape(1, 1))

#   # Train model
#   sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})

  
#   # Values to print
#   val = sess.run(svm_model.my_kernel, feed_dict={svm_model.x_input: X})
#   val_b = sess.run(svm_model.b)
#   vec_cross = sess.run(svm_model.b_vec_cross, feed_dict={svm_model.b: val_b})
#   target_cross = sess.run(svm_model.y_target_cross, feed_dict={svm_model.y_input: Y})
#   first_term = sess.run(svm_model.first_term, feed_dict={svm_model.b: val_b})
#   second_term = sess.run(svm_model.second_term, feed_dict={svm_model.b: val_b, svm_model.my_kernel: val, svm_model.y_target_cross: target_cross})


#   # Storing measures
#   temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
#   loss_vec.append(temp_loss)
#   acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X,svm_model.y_input: Y,svm_model.prediction_grid:X})
#   batch_accuracy.append(acc_temp)

#   first_vec.append(first_term)
#   second_vec.append(second_term)

#   # if (i+1)%10==0:
#   # print('Kernel Value = ' + str(val))
#   print('b val = ' + str(val_b))
#   # print('b_vec_cross = ' + str(vec_cross))
#   # print('y_target_cross = ' + str(target_cross))
#   # print('first_term = ' + str(first_term))
#   # print('second_term = ' + str(second_term))
#   print('Loss = ' + str(temp_loss))
#   print('Accuracy = ' + str(acc_temp))
#   print('Step #' + str(i+1))

# Batch 
for i in range(1000):

  # Create batch
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  X = x_vals[rand_index]
  Y = np.transpose([y_vals[rand_index]])
  Y_7_count.append(list(Y).count(1))

  # Train model
  sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})

  
  # Values to print
  val = sess.run(svm_model.my_kernel, feed_dict={svm_model.x_input: X})
  val_b = sess.run(svm_model.b)
  vec_cross = sess.run(svm_model.b_vec_cross, feed_dict={svm_model.b: val_b})
  target_cross = sess.run(svm_model.y_target_cross, feed_dict={svm_model.y_input: Y})
  first_term = sess.run(svm_model.first_term, feed_dict={svm_model.b: val_b})
  second_term = sess.run(svm_model.second_term, feed_dict={svm_model.b: val_b, svm_model.my_kernel: val, svm_model.y_target_cross: target_cross})


  # Storing batch set performance
  temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
  loss_vec.append(temp_loss)
  acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X, svm_model.y_input: Y, svm_model.prediction_grid: X})
  batch_accuracy.append(acc_temp)

  # # Storing test set performance
  # temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: mnist_test_x, svm_model.y_input: np.transpose([mnist_test_y])})
  # loss_vec.append(temp_loss)
  # acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: mnist_test_x,svm_model.y_input: np.transpose([mnist_test_y]),svm_model.prediction_grid:mnist_test_x})
  # batch_accuracy.append(acc_temp)

  first_vec.append(first_term)
  second_vec.append(second_term)

  # if (i+1)%10==0:
  # print('Kernel Value = ' + str(val))
  print('b val = ' + str(val_b))
  # print('b_vec_cross = ' + str(vec_cross))
  # print('y_target_cross = ' + str(target_cross))
  # print('first_term = ' + str(first_term))
  # print('second_term = ' + str(second_term))
  print('Loss = ' + str(temp_loss))
  print('Accuracy = ' + str(acc_temp))
  print('Step #' + str(i+1))

# print(sum(Y_7_count) / len(Y_7_count))

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

# Plot first over time
plt.plot(first_vec, 'k-')
plt.title('First Term per Generation')
plt.xlabel('Generation')
plt.ylabel('First Term')
plt.show()

# Plot second over time
plt.plot(second_vec, 'k-')
plt.title('Second Term per Generation')
plt.xlabel('Generation')
plt.ylabel('Second Term')
plt.show()

sess.close()
