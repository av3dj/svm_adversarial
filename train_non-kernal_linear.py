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

# Setting up the data and the model
mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
model = Model()

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
# Implementation built from: https://medium.com/cs-note/tensorflow-ch4-support-vector-machines-c9ad18878c76
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])

setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 100
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

model_output = tf.subtract(tf.matmul(x_data, A), b)

# Loss stuff
l2_norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0.1])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1.,tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# Prediction and accuracy functions
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Optimizer stuff
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()

loss_vec = []
train_accuracy = []
test_accuracy = []

# Make this train an SVM
with tf.Session() as sess:
  sess.run(init)

  for i in range(500):
    print('Running Epoch %d ...', i)
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    X = x_vals_train[rand_index]
    Y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: X, y_target: Y})
    temp_loss = sess.run(loss, feed_dict={x_data: X, y_target: Y})
    loss_vec.append(temp_loss)
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

  # print('loss_vec: ' + str(loss_vec))
  # print('train_accuracy: ' + str(train_accuracy))
  # print('test_accuracy: ' + str(test_accuracy))
  print(type(train_step))
  print(float(loss))

plt.figure()
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(len(loss_vec)), loss_vec)
plt.show()

plt.figure()
plt.title("Training Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.plot(range(len(train_accuracy)), train_accuracy)
plt.show()

plt.figure()
plt.title("Testing Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Testing Accuracy")
plt.plot(range(len(test_accuracy)), test_accuracy)
plt.show()

  

