import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Return mnist data with only 1's and 7's as tensor
def prepare_mnist(mixed=False):

  if mixed:
    mnist = np.load('mnist_17_attack_clean-centroid_normc-0.8_epsilon-0.3.npz', allow_pickle=True)
    
    return (mnist['X_modified'], mnist['Y_modified'], mnist['X_test'], mnist['Y_test'])

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
