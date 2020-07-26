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

import json
import sys
import math
from preprocess import prepare_mnist
from model import Model

# Test
# - attack the normally trained model using the PGD attack
#   - Ideally, a good PGD attack would decrease the accuracy down to 0%
# - obtain the adversarially trained model using the adversarial training.
#   - use the PGD attack to attack the adversarially trained model and see what is the accuracy of the model on the attacked images.
#   - should give maybe > 90%, but not a guarantee
#   - For attack strength, set the epsilon value to be 0.3, for image pixels scaled in [0,1] range

class LinfPGDAttack(object):
  def __init__(self, model, epsilon, k, a, random_start, momentum, beta, random_seed):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon # Threshold of changes to original image
    self.k = k # Number of iterations for perturb loop
    self.a = a # Step size for each perturb
    self.rand = random_start # Add random noise to each image
    self.momentum = momentum

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    self.beta = beta
    self.V = 0 # Velocity 

    loss = model.loss

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range

    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      gradient = grad

      if self.momentum:
        if i == 0: # init velocity # if instable try implementing bias-corrected version
          self.V = (1-self.beta) * grad
        else:
          self.V = self.beta * self.V + (1-self.beta) * grad

        gradient = self.V

      sign = np.sign(gradient)

      # if i % 10 == 0:
      #   print(stats.describe(self.V[0]))
      #   print(stats.describe(sign[0]))

      perturbation = self.a * sign
      
      x += perturbation

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

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

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

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



# if __name__ == '__main__':
#   import json
#   import sys
#   import math
#   from preprocess import prepare_mnist

#   with open('config.json') as config_file:
#     config = json.load(config_file)

#   (mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y) = prepare_mnist(mixed=config['mixed_dataset'])

#   from model import Model
  
#   tf.set_random_seed(config['random_seed'])
#   np.random.seed(config['random_seed'])

#   model_file = None
#   if config['adv_training']:
#     model_file = tf.train.latest_checkpoint(config['model_dir_adv'])
#   else:
#     model_file = tf.train.latest_checkpoint(config['model_dir'])
#   if model_file is None:
#     print('No model found')
#     sys.exit()

#   num_eval_examples = config['num_eval_examples']
#   eval_batch_size = config['eval_batch_size']
#   num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

#   model = Model(batch_size=eval_batch_size)
#   attack = LinfPGDAttack(model,
#                          config['epsilon'],
#                          config['k'],
#                          config['a'],
#                          config['random_start'],
#                          config['loss_func'])
#   saver = tf.train.Saver()

#   with tf.Session() as sess:
#     # Restore the checkpoint
#     saver.restore(sess, model_file)

#     # Iterate over the samples batch-by-batch

#     x_adv = [] # adv accumulator

#     print('Iterating over {} batches'.format(num_batches))

#     for ibatch in range(num_batches):
#       print(ibatch)
#       bstart = ibatch * eval_batch_size
#       if bstart == 2000:
#         break
#       bend = min(bstart + eval_batch_size, num_eval_examples)
#       print('batch size: {}'.format(bend - bstart))

#       x_batch = mnist_test_x[bstart:bend, :]
#       y_batch = np.transpose([mnist_test_y[bstart:bend]])

#       x_batch_adv = attack.perturb(x_batch, y_batch, sess)

#       x_adv.append(x_batch_adv)

#     print('Storing examples')
#     path = config['store_adv_path']
#     x_adv = np.concatenate(x_adv, axis=0)
#     np.save(path, x_adv)
#     print('Examples stored in {}'.format(path))
