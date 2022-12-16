from ATE import *

from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_boston
import pickle as pkl

import numpy as np
import scipy
import math
import random


class ATEDesignEstimates:
    """ATEDesignEstimates is used for storing various estimator outputs and used for plotting"""
    def __init__(self, mean, std, samples_list, sampling_type):
        self.mean = mean
        self.std = std
        self.samples_list = samples_list
        self.sampling_type = sampling_type


def run_ATE_estimator(samples_list, X, Y1, Y0, estimator, trials, percentile):
    """ATE estimators for GSW and sampling"""

    npoints, ndim = X.shape
    mean_error, percentile_error = [], []

    ATE_actual = ATE(Y1, Y0)
    for samples in samples_list:
        error_trials = []

        for trial in range(0, trials):
            ATE_est = estimator(X, Y1, Y0, s=samples)
            error_trials.append(abs(ATE_actual - ATE_est))

        mean_error_trial, percentile_error_trial = np.mean(error_trials), np.percentile(error_trials, percentile)
        mean_error.append(mean_error_trial)
        percentile_error.append(percentile_error_trial)

    return mean_error, percentile_error


def run_ATE_pop_estimator(samples_list, X, Y1, Y0, estimator, trials, percentile):
    """ATE estimators at the population level, for baseline comparisons"""
    npoints, ndim = X.shape
    mean_error, percentile_error = [], []

    ATE_actual = ATE(Y1, Y0)
    error_trials = []

    for trial in range(0, trials):
        ATE_est = estimator(X, Y1, Y0)
        error_trials.append(abs(ATE_actual - ATE_est))

    mean_error_trial, percentile_error_trial = np.mean(error_trials), np.percentile(error_trials, percentile)

    for samples in samples_list:
        mean_error.append(mean_error_trial)
        percentile_error.append(percentile_error_trial)

    return mean_error, percentile_error


def run_ATE(samples_list, X, Y1, Y0, ntrials=50, percentile=70):
    """Obtain various estimators for ATE"""
    n, ndim = X.shape
    samples_percentage = []

    results = []
    for nsamples in samples_list:
        samples_percentage.append(int(float(nsamples * 100) / n))

    # coreset: Recursive gsw
    mean_error_core, percentile_error_core = run_ATE_estimator(samples_list, X, Y1, Y0, ATE_gsw_estimator, ntrials,
                                                               percentile)
    coreset = ATEDesignEstimates(mean_error_core, percentile_error_core, samples_percentage, "Recursive-GSW")
    results.append(coreset)

    # Uniform: Uniform sampling
    mean_error_unf, percentile_error_unf = run_ATE_estimator(samples_list, X, Y1, Y0, unif_estimator, ntrials,
                                                             percentile)
    unf = ATEDesignEstimates(mean_error_unf, percentile_error_unf, samples_percentage, "Uniform")
    results.append(unf)

    # gsw baseline
    mean_error_gsw, percentile_error_gsw = run_ATE_pop_estimator(samples_list, X, Y1, Y0, ATE_gsw_pop_estimator,
                                                                 ntrials,
                                                                 percentile)
    gsw = ATEDesignEstimates(mean_error_gsw, percentile_error_gsw, samples_percentage, "GSW-pop")
    results.append(gsw)

    # Uniform population baseline
    mean_error_unf_pop, percentile_error_unf_pop = run_ATE_pop_estimator(samples_list, X, Y1, Y0,
                                                                         ATE_rand_pop_estimator, ntrials,
                                                                         percentile)
    unf_pop = ATEDesignEstimates(mean_error_unf_pop, percentile_error_unf_pop, samples_percentage, "Complete Randomization")
    results.append(unf_pop)

    return results
