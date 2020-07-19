import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Return mnist data with only 1's and 7's as tensor
def prepare_mnist(mixed=False):

  if mixed:
    mnist_orig = np.load('mnist_17_attack_clean-centroid_normc-0.8_epsilon-0.3.npz', allow_pickle=True)

    mnist = {
      'X_train': mnist_orig['X_modified'],
      'Y_train': mnist_orig['Y_modified'],
      'X_test': mnist_orig['X_test'],
      'Y_test': mnist_orig['Y_test']
    }
    
    return mnist

  else: 
    mnist_orig = np.load('mnist_17_train_test.npz', allow_pickle=True)
    
    mnist = {
      'X_train': mnist_orig['X_train'],
      'Y_train': mnist_orig['Y_train'],
      'X_test': mnist_orig['X_test'],
      'Y_test': mnist_orig['Y_test']
    }

    return mnist


