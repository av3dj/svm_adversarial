"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python import pywrap_tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.backend import clear_session
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import json
import sys
import math
from preprocess import prepare_mnist
from model import Model

class LinfPGDAttack(object):
  def __init__(self, model, epsilon, k, a, random_start, momentum, beta, random_seed, plotter=None):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon # Threshold of changes to original image
    self.k = k # Number of iterations for perturb loop
    self.a = a # Step size for each perturb
    self.rand = random_start # Add random noise to each image
    self.momentum = momentum

    self.plotter = plotter

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    self.beta = beta
    self.V = 0 
    self.loss = model.loss

    self.grad = tf.gradients(self.loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, train_debug=False, debug=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm.
       
       train_debug: controls loss printing during steps
       debug: controls plotting and gradient print statements

       """

    if self.rand:
      # Random Start
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) 
    else:
      # Start at exact position
      x = np.copy(x_nat)

    if train_debug:
        loss_prev = sess.run(self.model.loss, feed_dict={self.model.x_input: x,
                                              self.model.y_input: y})
    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})
      gradient = grad

      # Do momentum gradient step
      if self.momentum:
        if i == 0: 
          self.V = (1-self.beta) * grad
        else:
          self.V = self.beta * self.V + (1-self.beta) * grad
        gradient = self.V

      sign = np.sign(gradient)

      if np.count_nonzero(sign) == 0:
        print('No change therefore stopping perturbation early ' + str(i))
        break

      perturbation = self.a * sign
      x += perturbation

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      # x = np.clip(x, 0, 1) # only relevant for MNIST

      if debug and i % 25 == 0:
        # print(grad)
        # print(stats.describe(gradient))
        # print(perturbation)
        # print(stats.describe(sign[0]))
        # test_loss = sess.run(self.loss, feed_dict={self.model.x_input: x, self.model.y_input: y})
        # print(test_loss)
        continue

      if debug and self.plotter is not None:
        self.plotter.plot(sess, self.model, X=x, Y=y, pgd_attack_iter=i, pgd_attack=True)

    if train_debug:
      loss_new = sess.run(self.model.loss, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})
      print('Old Loss:' + str(loss_prev))
      print('New Loss:' + str(loss_new))

    return x

def create_attack(dataset, config, adversarial, mixed):
  clear_session()
  
  # Set seeds
  tf.set_random_seed(config['random_seed'])
  np.random.seed(config['random_seed'])

  # Extract dataset
  mnist_test_x = dataset['X_test']
  mnist_test_y = dataset['Y_test']

  # Load model
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

  # Set Parameters
  num_eval_examples = config['num_eval_examples']
  eval_batch_size = config['eval_batch_size']
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

  # Get tensorflow objects
  model = Model(batch_size=eval_batch_size)
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['momentum'],
                         config['beta'],
                         config['random_seed'])
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch

    x_adv = [] # adv accumulator

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      if bstart == 2000:
        break
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist_test_x[bstart:bend, :]
      y_batch = np.transpose([mnist_test_y[bstart:bend]])

      x_batch_adv = attack.perturb(x_batch, y_batch, sess, train_debug=False)

      x_adv.append(x_batch_adv)

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
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)