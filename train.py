"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
from timeit import default_timer as timer

# Use previous api of TensorFlow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.backend import clear_session

import numpy as np
import sklearn.datasets as datasets
from scipy import stats

import matplotlib.pyplot as plt

from model import Model
from pgd_attack import LinfPGDAttack
from preprocess import prepare_mnist

def train_model(dataset, config, adversarial, mixed):

  clear_session()

  # Set seeds
  tf.set_random_seed(config['random_seed'])
  np.random.seed(config['random_seed'])

  # Set save directory
  model_dir = ""
  if adversarial:
    if mixed:
      model_dir = config['model_dir_adv_mixed']
    else:
      model_dir = config['model_dir_adv']
  else:
    if mixed:
      model_dir = config['model_dir_mixed']
    else:
      model_dir = config['model_dir']
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  # Set dataset  
  x_vals = dataset['X_train']
  y_vals = dataset['Y_train']

  # Get parameters
  batch_size = config['batch_size']
  C = config['C']
  learning_rate = config['learning_rate']

  # Setup tensorflow objects
  svm_model = Model(batch_size, C=C)
  global_step = tf.compat.v1.train.get_or_create_global_step()

  attack = LinfPGDAttack(svm_model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'],
                       config['beta'],
                       config['random_seed'])

  # Set optimizer for model training 
  my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  train_step = my_opt.minimize(svm_model.loss)
  init = tf.initialize_all_variables()

  # Variables used during training
  X = None
  Y = None

  # Start tensorflow session
  with tf.Session() as sess:
    sess.run(init)

    # Training: Batch Gradient Descent
    for i in range(100):

      # Create randomly selected batch
      rand_index = np.random.choice(len(x_vals), size=batch_size)
      X = x_vals[rand_index]
      Y = np.transpose([y_vals[rand_index]])

      # In case of adversarial training we perturb the batch data
      X_adv = None
      if adversarial:
        X_adv = attack.perturb(X, Y, sess)

      # Storing batch set performance
      temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})

      acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X, svm_model.y_input: Y, svm_model.prediction_grid: X})

      if (i+1)%1==0:
        print('\nStep #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('Natural Accuracy = ' + str(acc_temp))

      # Train model
      if adversarial:
        sess.run(train_step, feed_dict={svm_model.x_input: X_adv, svm_model.y_input: Y})
      else:
        sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})

    # Save model
    saver = tf.train.Saver(max_to_keep=3)

    if adversarial:
      if mixed:
        saver.save(sess,
                      os.path.join(config['model_dir_adv_mixed'], 'model_' + 'batch-size-' + str(config['batch_size']) + '_C-' + str(config['C']) + '_learning-rate-' + str(config['learning_rate'])),
                      global_step=global_step)
      else:
        saver.save(sess,
                      os.path.join(config['model_dir_adv'], 'model_' + 'batch-size-' + str(config['batch_size']) + '_C-' + str(config['C']) + '_learning-rate-' + str(config['learning_rate'])),
                      global_step=global_step)
    else:
      if mixed:
        saver.save(sess,
                      os.path.join(config['model_dir_mixed'], 'model_' + 'batch-size-' + str(config['batch_size']) + '_C-' + str(config['C']) + '_learning-rate-' + str(config['learning_rate'])),
                      global_step=global_step)
      else:
        saver.save(sess,
                      os.path.join(config['model_dir'], 'model_' + 'batch-size-' + str(config['batch_size']) + '_C-' + str(config['C']) + '_learning-rate-' + str(config['learning_rate'])),
                      global_step=global_step)


# with open('config.json') as config_file:
#     config = json.load(config_file)

# # Set seeds
# tf.set_random_seed(config['random_seed'])
# np.random.seed(config['random_seed'])

# model_dir = config['model_dir']
# if not os.path.exists(model_dir):
#   os.makedirs(model_dir)

# # Acquire mnist dataset (only 1s and 7s)
# print("getting dataset")
# (mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y) = prepare_mnist(mixed=config['mixed_dataset'])
# print("got dataset")

# x_vals = mnist_train_x
# y_vals = mnist_train_y

# # Set batch size (currently SVM only allows data that matches this batch size (can't feed smaller or larger) ??? potential fix somehow)
# batch_size = 2000#2163 Increasing batch size requires adjustment of C value

# # Setup model and adversary
# svm_model = Model(batch_size, C=0.01)

# global_step = tf.compat.v1.train.get_or_create_global_step()

# attack = LinfPGDAttack(svm_model, 
#                        config['epsilon'],
#                        config['k'],
#                        config['a'],
#                        False,
#                        config['loss_func'])

# # Set optimizer for model training 
# my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

# train_step = my_opt.minimize(svm_model.loss)
# init = tf.initialize_all_variables()

# # Bookeeping stuff
# loss_vec = []
# batch_accuracy = []

# X = None
# Y = None

# # Start tensorflow session and start training
# sess = tf.Session()
# sess.run(init)

# # Batch Gradient Descent
# for i in range(100):

#   # Create randomly selected batch
#   rand_index = np.random.choice(len(x_vals), size=batch_size)
#   X = x_vals[rand_index]
#   Y = np.transpose([y_vals[rand_index]])

#   # In case of adversarial training we perturb the batch data\
#   X_adv = None
#   if config['adv_training']:
#     X_adv = attack.perturb(X, Y, sess)

#   # Storing batch set performance
#   temp_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
#   loss_vec.append(temp_loss)

#   acc_temp = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X, svm_model.y_input: Y, svm_model.prediction_grid: X})
#   batch_accuracy.append(acc_temp)

#   if (i+1)%1==0:
#     print('Loss = ' + str(temp_loss))
#     print('Natural Accuracy = ' + str(acc_temp))
#     print('Step #' + str(i+1))

#   # Train model
#   if config['adv_training']:
#     sess.run(train_step, feed_dict={svm_model.x_input: X_adv, svm_model.y_input: Y})
#   else:
#     sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})

# # Save model to checkpoint
# saver = tf.train.Saver(max_to_keep=3)

# model_dir = config['model_dir']

# if config['adv_training']:
#   saver.save(sess,
#                  os.path.join(config['model_dir_adv'], 'checkpoint-adv'),
#                  global_step=global_step)
# else:
#   saver.save(sess,
#                  os.path.join(config['model_dir'], 'checkpoint'),
#                  global_step=global_step)

# # Plot batch accuracy
# plt.plot(batch_accuracy, 'k-', label='Accuracy')
# plt.title('Normal Batch Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

# # Plot loss over time
# plt.plot(loss_vec, 'k-')
# plt.title('Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()

# sess.close()
