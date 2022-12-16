import random
from numpy import linalg as LA
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t


def leverage_score_syn(X, **qargs):
    """Compute leverage scores"""
    n, d = X.shape
    Ul, Sl, Vhl = np.linalg.svd(X, full_matrices=False)
    scores = [LA.norm(Ul[i]) ** 2 for i in range(0, n)]

    return scores


def generate_synthetic_non_unf_lev(n=1024, d=25):
    # generate
    min_sig, max_sig = 1.0, 5.0
    # n = 1024

    covariates = []

    for i in range(0, int(math.log(n, 2))):
        mu = i * i
        # sigma = random.uniform(min_sig, max_sig)
        for j in range(int(math.pow(2, i)), int(math.pow(2, i + 1))):
            v = np.random.normal(mu, 1.0, d)
            v = v / np.linalg.norm(v)
            covariates.append(v)

    return np.array(covariates)


def mean(d=25):
    return np.ones(d)


def cov(d=25):
    covariance_matrix = np.zeros((25, 25))
    for i in range(0, 25):
        for j in range(0, 25):
            covariance_matrix[i][j] = 2.0 * math.pow(0.5, abs(i - j))

    return covariance_matrix


def beta(d=25):
    beta_vec = np.ones(d)

    for i in range(10, d - 20):
        beta_vec[i] = 0.10

    return beta_vec


def noise(n=1024):
    sigma = 9 * np.identity(n)
    epsilon = np.diag(np.random.normal(0, sigma))


def generate_synthetic_data(n, d):
    mu = mean(d)
    covariance_matrix = cov(d)

    X = multivariate_t(mu, covariance_matrix, df=1).rvs(n)
    X = np.array(X)

    return X

