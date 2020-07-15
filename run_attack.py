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
import sys
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from preprocess import prepare_mnist

import numpy as np

from model import Model

def run_attack(checkpoint, x_adv, epsilon, adv_testing=False, mixed_dataset=False):

  # Get dataset
  (mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y) = prepare_mnist(mixed=mixed_dataset)

  num_eval_examples = 2000
  eval_batch_size = 2000 # If changing this make sure to change this elsewhere

  # Create model
  model = Model(batch_size=eval_batch_size)
  saver = tf.train.Saver()

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  l_inf = np.amax(np.abs(mnist_test_x[0:num_eval_examples] - x_adv))
  
  # Check constraints
  if l_inf > epsilon + 0.0001 and not mixed_dataset:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

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

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))

if __name__ == '__main__':
  import json

  with open('config.json') as config_file:
    config = json.load(config_file)
  
  tf.set_random_seed(config['random_seed'])
  np.random.seed(config['random_seed'])

  checkpoint = None
  if config['adv_training']:
    checkpoint = tf.train.latest_checkpoint(config['model_dir_adv'])
  else:
    checkpoint = tf.train.latest_checkpoint(config['model_dir'])
  if checkpoint is None:
    print('No model found')
    sys.exit()

  x_adv = np.load(config['store_adv_path'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (2000, 784):
    print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 1.0001 or \
       np.amin(x_adv) < -0.0001 or \
       np.isnan(np.amax(x_adv)):
    print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config['epsilon'], config['adv_testing'], config['mixed_dataset'])
