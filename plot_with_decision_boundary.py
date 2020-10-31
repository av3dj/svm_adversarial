import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

class Plotter(object):

    def __init__(self, pgd_attack_file_path="./pgdattack_photos", training_file_path="./train_photos"):
        self.pgd_attack_file_path = pgd_attack_file_path
        self.training_file_path = training_file_path

    def plot(self, sess, model, X, Y, train_iter=-1, pgd_attack_iter=-1, pgd_attack=False):

        # Calculate Decision Boundary things
        axes = np.arange(-3., 3., 0.01)

        A = sess.run(model.A)
        b = sess.run(model.b)
        if A[1] == 0:
            A[1] = 0.00000000001
        slope = A[0] / A[1]
        intercept = b / A[1]

        line = np.reshape(intercept - slope * axes, len(axes))
        pred = sess.run(model.prediction, feed_dict={model.x_input: np.reshape([0, intercept-1], (-1, 2))})[0][0]

        # Plot Decision Boundary
        above_color = (1, 1, 0, 0.1) if pred > 0 else (1, 0, 0, 0.1)
        below_color = (1, 0, 0, 0.1) if pred > 0 else (1, 1, 0, 0.1)
        plt.fill_between(axes, line, 10, interpolate=True, color=above_color)
        plt.fill_between(axes, -10, line, interpolate=True, color=below_color)
        
        # Plot X points
        Y_fixed = np.reshape(Y, len(Y))
        plt.plot(X[:, 0][Y_fixed==-1], X[:, 1][Y_fixed==-1], 'r1', X[:, 0][Y_fixed==1], X[:, 1][Y_fixed==1], 'y+', markersize=0.75)
        plt.plot(axes, line)
        plt.axis([-3,3,-3,3])

        # Generate Key
        red_patch = matplotlib.patches.Patch(color='red', label='"-1"')
        yellow_patch = matplotlib.patches.Patch(color='yellow', label='"1"')
        plt.legend(handles=[red_patch, yellow_patch], loc='upper left')

        # Output Picture
        if pgd_attack:
            plt.title('PGDAttack Iteration ' + str(pgd_attack_iter))
            plt.savefig(self.pgd_attack_file_path + '/pgdattack_iter_' + str(pgd_attack_iter) + '.png')
        else:
            plt.title('Train Iteration ' + str(train_iter))
            plt.savefig(self.training_file_path + '/train_iter_' + str(train_iter) + '.png')
        
        # Close instance
        plt.clf()