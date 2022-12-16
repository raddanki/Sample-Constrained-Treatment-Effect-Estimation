import numpy as np
import scipy
import math
import random
from data import *
from custom_plot import *

from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_boston
import pickle as pkl


def thresh_SVD(X, threshold):
    """"Threshold SVD returns a smoothed matrix by removing singular directions with low singular values"""
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    small_sing_index = len(S)
    for i in range(0, len(S)):
        if S[i] ** 2 < threshold:
            small_sing_index = i
            break
    Uplus, Splus, Vhplus = U[:, 0:small_sing_index], S[0:small_sing_index], Vh[0:small_sing_index, :]
    Xplus = Uplus @ np.diag(Splus, k=0) @ Vhplus

    return Xplus


def normalize(p):
    """Normalize it to make it a probability distribution"""
    return p / p.sum()


def leverage_score(X, **qargs):
    """Compute leverage scores"""
    n, d = X.shape
    Ul, Sl, Vhl = np.linalg.svd(X, full_matrices=False)
    scores = [LA.norm(Ul[i]) ** 2 for i in range(0, n)]

    return scores


def lev_bernoulli_sampling(X, s, gamma):
    """"Bernoulli sampling based on leverage scores multiplied by the factor gamma"""
    n, d = X.shape
    lscores = np.array(leverage_score(X))
    lscores = np.array([min(1.0, lscores[i] * float(gamma)) for i in range(0, len(lscores))])

    sampled_rows = []

    for i in range(0, len(lscores)):
        if np.random.random() <= lscores[i]:
            sampled_rows.append(i)

    return sampled_rows, lscores


def unif_sampling(X, s, gamma):
    """Uniform sampling with replacement"""
    n, d = X.shape
    uscores = np.ones(n)

    pi = normalize(uscores)
    sampled_rows = np.random.choice(range(n), s, p=pi)  # sampling with replacement

    return sampled_rows, pi


def sampled_linear_regression(X, sampled_rows, Y, s, pi):
    """Linear regression using sub-sampled matrix"""
    n, d = X.shape

    # S is a s x n matrix, D is a s x s matrix
    S = np.zeros((s, n))
    D = np.zeros((s, s))
    i = 0

    #scaling the matrix appropriately
    for row in sampled_rows:
        S[i][row] = 1
        D[i][i] = 1.0 / (math.sqrt(pi[row]))
        i += 1

    Xsampled = D @ S @ X
    Ysampled = D @ S @ Y

    beta = np.linalg.lstsq(Xsampled, Ysampled, rcond=None)[0]

    return np.array(beta)


def ITE_estimator(data_matrix, nsamples, Y1, Y0, gamma=1.0, sampling=lev_bernoulli_sampling):

    srows_0, pi_0 = sampling(data_matrix, int(nsamples / 2.0), gamma)
    srows_1, pi_1 = sampling(data_matrix, int(nsamples / 2.0), gamma)
    srows_1_dup_rmv = []

    # Remove people sampled twice from the treatment group
    for i in range(0, len(srows_1)):
        if srows_1[i] in set(srows_0):
            continue
        else:
            srows_1_dup_rmv.append(srows_1[i])

    # Solve linear regression for control group
    beta_0 = sampled_linear_regression(data_matrix, srows_0, Y0, len(srows_0), pi_0)

    # Reweighting the new probabilities, as we removed the duplicates from treatment group
    pi_1_upd = [0.0 for i in range(0, len(pi_1))]
    for i in range(0, len(pi_1)):
        pi_1_upd[i] = pi_1[i] * (1-pi_1[i])

    # Solve linear regression for treatment group
    beta_1 = sampled_linear_regression(data_matrix, srows_1_dup_rmv, Y1, len(srows_1_dup_rmv), pi_1_upd)  # changed

    data_matrix = np.array(data_matrix)
    Y1_est = data_matrix @ beta_1
    Y0_est = data_matrix @ beta_0

    ITE_est = Y1_est - Y0_est
    ITE = Y1 - Y0

    return ITE_est, ITE







