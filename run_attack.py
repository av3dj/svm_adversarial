"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
#    Accuracy: 52.60%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.backend import clear_session

from preprocess import prepare_mnist

import numpy as np

from model import Model

def run_attack(checkpoint, dataset, x_adv, config, adv_testing=False, mixed_dataset=False):

  clear_session()

  # Get dataset
  mnist_test_x = dataset['X_test']
  mnist_test_y = dataset['Y_test']

  num_eval_examples = config['num_eval_examples']
  eval_batch_size = config['eval_batch_size'] # If changing this make sure to change this elsewhere

  # Create model
  model = Model(batch_size=eval_batch_size)
  saver = tf.train.Saver()

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  l_inf = np.amax(np.abs(mnist_test_x[0:num_eval_examples] - x_adv))
  
  # Check constraints (removed for gaussian?)
  # if l_inf > config['epsilon'] + 0.0001 and not mixed_dataset:
  #   print('maximum perturbation found: {}'.format(l_inf))
  #   print('maximum perturbation allowed: {}'.format(config['epsilon']))
  #   return

  with tf.Session() as sess:

    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      if bstart == eval_batch_size:
        break

      # Create batch
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = np.transpose([mnist_test_y[bstart:bend]])
    
      dict_nat = {model.x_input: mnist_test_x[bstart:bend, :],
                  model.y_input: y_batch,
                  model.prediction_grid: mnist_test_x[bstart:bend, :]}
      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch,
                  model.prediction_grid: x_batch}

      # Calculate accuracy
      if adv_testing:
        accuracy = sess.run(model.accuracy,
                                        feed_dict=dict_adv)
      else:
        accuracy = sess.run(model.accuracy,
                                        feed_dict=dict_nat)

      total_corr += accuracy * eval_batch_size
    

  accuracy = total_corr / num_eval_examples

  return accuracy

def check_and_run_attack(dataset, config, adversarial, mixed):

  clear_session()

  # Set seed
  tf.set_random_seed(config['random_seed'])
  np.random.seed(config['random_seed'])

  # Get model
  model_file = None
  if adversarial:
    if mixed:
      model_file = tf.train.latest_checkpoint(config['model_dir_adv_mixed'])
    else:
      model_file = tf.train.latest_checkpoint(config['model_dir_adv'])
  else:
    if mixed:
      model_file = tf.train.latest_checkpoint(config['model_dir_mixed'])
    else:
      model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()
  
  # Get adversarial dataset
  path = ""
  if adversarial:
    if mixed:
      path = "attack-adv-mixed.npy"
    else:
      path = "attack-adv.npy"
  else:
    if mixed:
      path = "attack-normal-mixed.npy"
    else:
      path = "attack-normal.npy"

  x_adv = np.load(path)

  clean_acc = 0
  robust_acc = 0

  if model_file is None:
    print('No checkpoint found')
  # elif x_adv.shape != (2000, 3):
  #   print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
  # elif np.amax(x_adv) > 1.0001 or \
  #      np.amin(x_adv) < -0.0001 or \
  #      np.isnan(np.amax(x_adv)):
  #   print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
  #                                                             np.amin(x_adv),
  #                                                             np.amax(x_adv)))
  else:
    clean_acc = run_attack(model_file, dataset, x_adv, config, adv_testing=False, mixed_dataset=mixed)
    robust_acc = run_attack(model_file, dataset, x_adv, config, adv_testing=True, mixed_dataset=mixed)

  return (clean_acc, robust_acc)