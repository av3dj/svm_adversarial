import scipy.stats as st
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import json

class GaussianDistributionGenerator():

    """
    Generates dataset based on Gaussian Distribution

    __init__ Parameters
    -------------------
    mu : float
        Mean for Gaussian Distribution
    
    sigma : float
        Standard deviation for Gaussian Distribution

    Call generate to create dataset

    """

    def __init__(self, mu, sigma, test_size, seed=None):

        if not seed:
            with open('config.json') as f:
                config = json.load(f)
                self.seed = config['random_seed']
        else:
            self.seed = seed

        self.test_size = test_size
        self.mu = mu
        self.sigma = sigma

    def generate(self, sample_size=10000):

        from scipy.stats import multivariate_normal
        from sklearn.model_selection import train_test_split
        
        np.random.seed(seed=self.seed)

        cov_matrix = np.diag([self.sigma**2, self.sigma**2])
        mu_matrix = np.asarray([self.mu, self.mu])

        Y_neg_sample_size = int(sample_size/2)
        Y_pos_sample_size = sample_size - Y_neg_sample_size

        Y_neg = [-1 for _ in range(Y_neg_sample_size)]
        Y_pos = [1 for _ in range(Y_pos_sample_size)]

        # Sample for y=-1
        X_neg = multivariate_normal.rvs(mean=mu_matrix*-1, cov=cov_matrix, size=Y_neg_sample_size)

        # Sample for y=1
        X_pos = multivariate_normal.rvs(mean=mu_matrix*1, cov=cov_matrix, size=Y_pos_sample_size)

        X = np.concatenate((X_neg, X_pos), axis=0)
        Y = np.concatenate((Y_neg, Y_pos), axis=0)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.seed)

        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test
        }

        return dataset

    def plot(self, X, y):
        axes = [-10, 10, -10, 10]
        plt.plot(X[:, 0][y==-1], X[:, 1][y==-1], "bs", markersize=0.5)
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", markersize=0.5)
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.show()
