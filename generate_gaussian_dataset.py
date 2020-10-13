import scipy.stats as st
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import json
import pickle

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

    def __init__(self, mu, sigma, test_ratio, dim=2, seed=None):

        if not seed:
            with open('config.json') as f:
                config = json.load(f)
                self.seed = config['random_seed']
        else:
            self.seed = seed

        self.test_ratio = test_ratio
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def generate(self, sample_size=10000):

        from scipy.stats import multivariate_normal
        from sklearn.model_selection import train_test_split
        
        np.random.seed(seed=self.seed)

        cov_matrix = np.diag([self.sigma**2 for i in range(self.dim)])
        mu_matrix = np.asarray([self.mu for i in range(self.dim)])

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

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_ratio, random_state=self.seed)

        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test
        }

        return dataset

    def plot2d(self, X, y, X_orig, Y_orig):
        axes = [-10, 10, -10, 10]
        plt.plot(X[:, 0][y==-1], X[:, 1][y==-1], "bs", X[:, 0][y==1], X[:, 1][y==1], "g^", X_orig[:, 0][Y_orig==-1], X_orig[:, 1][Y_orig==-1], 'r1', X_orig[:, 0][Y_orig==1], X_orig[:, 1][Y_orig==1], 'k+', markersize=0.5)
        plt.plot(X_orig[:, 0][Y_orig==-1], X_orig[:, 1][Y_orig==-1], 'r1', X_orig[:, 0][Y_orig==1], X_orig[:, 1][Y_orig==1], 'k+', markersize=3)
        plt.grid(True, which='both')
        plt.show()


gdg = GaussianDistributionGenerator(mu=1, sigma=0.4, dim=2, test_ratio=0.2)
dataset = gdg.generate()
with open('gaussian_train_test.npz', 'wb') as f:
    pickle.dump(dataset, f, protocol=2)