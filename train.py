"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import pickle
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

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from model import Model
from pgd_attack import LinfPGDAttack
from preprocess import prepare_mnist

def train_model(dataset, config, plotter, adversarial, mixed):

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
  weight_decay = config['weight_decay']
  C = 1.0 / ( batch_size * weight_decay )
  # C = config['C']
  learning_rate = config['learning_rate']

  # Setup tensorflow objects
  svm_model = Model(batch_size, C=C)
  global_step = tf.compat.v1.train.get_or_create_global_step()

  attack = LinfPGDAttack(svm_model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['momentum'],
                       config['beta'],
                       config['random_seed'],
                       plotter=plotter)

  # Set optimizer for model training 
  my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  train_step = my_opt.minimize(svm_model.loss)
  init = tf.initialize_all_variables()

  # Variables used during training
  X = None
  Y = None

  clean_loss_history = []
  clean_accuracy_history = []
  robust_loss_history = []
  robust_accuracy_history = []

  train_history = {}

  X_adv = None
  X_adv_save = None
  Y_save = None
  Y = None

  A = None
  b = None

  # Start tensorflow session
  with tf.Session() as sess:
    sess.run(init)

    # Training: Batch Gradient Descent
    for i in range(100): # Orig: 100

      # Create randomly selected batch
      rand_index = np.random.choice(len(x_vals), size=batch_size)
      X = x_vals[rand_index]
      Y = np.transpose([y_vals[rand_index]])

      # In case of adversarial training we perturb the batch data
      X_adv = None
      if adversarial:
        X_adv = attack.perturb(X, Y, sess, debug=i==40)
        X = X_adv

      # Storing batch set performance
      clean_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
      clean_acc = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X, svm_model.y_input: Y, svm_model.prediction_grid: X})

      robust_loss = 0
      robust_acc = 0

      if adversarial:
        robust_loss = sess.run(svm_model.loss, feed_dict={svm_model.x_input: X_adv, svm_model.y_input: Y})
        robust_acc = sess.run(svm_model.accuracy, feed_dict={svm_model.x_input: X_adv, svm_model.y_input: Y, svm_model.prediction_grid: X_adv})

      if (i+1)%1==0:
        print('\nStep #' + str(i+1))
        print('Clean Loss = ' + str(clean_loss))
        print('Clean Accuracy = ' + str(clean_acc))
        if adversarial:
          print('Robust Loss = ' + str(robust_loss))
          print('Robust Accuracy = ' + str(robust_acc))

      clean_loss_history.append(str(clean_loss[0][0]))
      clean_accuracy_history.append(str(clean_acc))
      if adversarial:
        robust_loss_history.append(str(robust_loss[0][0]))
        robust_accuracy_history.append(str(robust_acc))

      # Train model
      if adversarial:
        if not mixed:
          X_adv_save = X_adv
          Y_save = Y
        # print(X_adv - X)
        sess.run(train_step, feed_dict={svm_model.x_input: X_adv, svm_model.y_input: Y})
      else:
        sess.run(train_step, feed_dict={svm_model.x_input: X, svm_model.y_input: Y})
      
      plotter.plot(sess, model=svm_model, X=X, Y=Y, train_iter=i, pgd_attack=False)

    # Save model
    A = sess.run(svm_model.A)
    b = sess.run(svm_model.b)
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
  
  train_history['clean loss'] = clean_loss_history
  train_history['clean accuracy'] = clean_accuracy_history
  train_history['robust loss'] = robust_loss_history
  train_history['robust accuracy'] = robust_accuracy_history
  train_history['A'] = A
  train_history['b'] = b

  # print(train_history)

  data = {
    'X': X_adv_save,
    'Y': Y_save
  }

  # print(data)

  with open('gaussian_perturbed_train_test.npz', 'wb') as f:
    pickle.dump(data, f, protocol=2)

  return train_history
