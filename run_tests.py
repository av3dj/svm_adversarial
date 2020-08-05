from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import time

# Use previous api of TensorFlow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sklearn.datasets as datasets
from scipy import stats

import matplotlib.pyplot as plt

from model import Model
from pgd_attack import create_attack
from preprocess import prepare_mnist
from train import train_model
from run_attack import check_and_run_attack

with open('config.json') as config_file:
    config = json.load(config_file)

mnist_regular = prepare_mnist(mixed=False)
mnist_mixed = prepare_mnist(mixed=True)

"""
Train (normal, normal w/mixed, adv, adv w/mixed)

- train.py
"""

# print("\n=========================================================================================")
# print("=                               Training Normal SVM model                               =")
# print("=========================================================================================\n")
# t0 = time.time()
# history = train_model(mnist_regular, config, False, False)
# t1 = time.time()
# history['title'] = 'Normal SVM with Normal Dataset'
# with open(config['train_history_dir'] + 'normal.json', 'w') as f:
#     f.write(json.dumps(history))
# print("\nRunning Time: " + str(t1-t0) + " seconds\n")

# print("\n=========================================================================================")
# print("=                       Training Normal SVM model w/ Mixed Dataset                      =")
# print("=========================================================================================\n")
# t0 = time.time()
# history = train_model(mnist_mixed, config, False, True)
# t1 = time.time()
# history['title'] = 'Normal SVM with Mixed Dataset'
# with open(config['train_history_dir'] + 'normal_mixed.json', 'w') as f:
#     f.write(json.dumps(history))
# print("\nRunning Time: " + str(t1-t0) + " seconds\n")

# print("\n=========================================================================================")
# print("=                               Training Adversarial SVM model                          =")
# print("=========================================================================================\n")
# t0 = time.time()
# history = train_model(mnist_regular, config, True, False)
# t1 = time.time()
# history['title'] = 'Adversarial SVM with Normal Dataset'
# with open(config['train_history_dir'] + 'adversarial_normal.json', 'w') as f:
#     f.write(json.dumps(history))
# print("\nRunning Time: " + str(t1-t0) + " seconds\n")

print("\n=========================================================================================")
print("=                       Training Adversarial SVM model w/ Mixed Dataset                 =")
print("=========================================================================================\n")
t0 = time.time()
history = train_model(mnist_mixed, config, True, True)
t1 = time.time()
history['title'] = 'Adversarial SVM with Mixed Dataset'
with open(config['train_history_dir'] + 'adversarial_mixed.json', 'w') as f:
    f.write(json.dumps(history))
print("\nRunning Time: " + str(t1-t0) + " seconds\n")

print("\n=========================================================================================")
print("=                                    Creating Attack                                    =")
print("=========================================================================================\n")

# print("Using Normal SVM Model ... ")
# t0 = time.time()
# create_attack(mnist_regular, config, False, False)
# t1 = time.time()
# print("Running Time: " + str(t1-t0) + " seconds\n")

# print("Using Normal SVM Model w/ Mixed Dataset ... ")
# t0 = time.time()
# create_attack(mnist_mixed, config, False, True)
# t1 = time.time()
# print("Running Time: " + str(t1-t0) + " seconds\n")

# print("Using Adversarial SVM Model ... ")
# t0 = time.time()
# create_attack(mnist_regular, config, True, False)
# t1 = time.time()
# print("Running Time: " + str(t1-t0) + " seconds\n")

print("Using Adversarial Model w/ Mixed Dataset ... ")
t0 = time.time()
create_attack(mnist_mixed, config, True, True)
t1 = time.time()
print("Running Time: " + str(t1-t0) + " seconds\n")

print("\n=========================================================================================")
print("=                                     Running Tests                                     =")
print("=========================================================================================\n")

# print("Testing Normal SVM Model ... ")
# t0 = time.time()
# (clean, robust) = check_and_run_attack(mnist_regular, config, False, False)
# t1 = time.time()
# print('Clean Accuracy: {:.2f}%'.format(100.0 * clean))
# print('Robust Accuracy: {:.2f}%'.format(100.0 * robust))
# print("Running Time: " + str(t1-t0) + " seconds\n")

# print("Testing Normal SVM Model w/ Mixed Dataset ... ")
# t0 = time.time()
# (clean, robust) = check_and_run_attack(mnist_mixed, config, False, True)
# t1 = time.time()
# print('Clean Accuracy: {:.2f}%'.format(100.0 * clean))
# print('Robust Accuracy: {:.2f}%'.format(100.0 * robust))
# print("Running Time: " + str(t1-t0) + " seconds\n")

# print("Testing Adversarial SVM Model ... ")
# t0 = time.time()
# (clean, robust) = check_and_run_attack(mnist_regular, config, True, False)
# t1 = time.time()
# print('Clean Accuracy: {:.2f}%'.format(100.0 * clean))
# print('Robust Accuracy: {:.2f}%'.format(100.0 * robust))
# print("Running Time: " + str(t1-t0) + " seconds\n")

print("Testing Adversarial Model w/ Mixed Dataset ... ")
t0 = time.time()
(clean, robust) = check_and_run_attack(mnist_mixed, config, True, True)
t1 = time.time()
print('Clean Accuracy: {:.2f}%'.format(100.0 * clean))
print('Robust Accuracy: {:.2f}%'.format(100.0 * robust))
print("Running Time: " + str(t1-t0) + " seconds\n")


"""
Test (clean accuracy, robust accuracy)

- pgd_attack.py for each model
- run_attack.py for each model twice
    - one for clean
    - one for robust
"""
