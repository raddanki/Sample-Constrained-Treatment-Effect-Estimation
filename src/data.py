import numpy as np
from numpy import linalg as LA
from sklearn.datasets import load_boston
import pickle as pkl
import random
# import ihdp
# import lalonde
import pandas as pd
from matplotlib.ticker import PercentFormatter

import synthetic_non_unf_lev
from synthetic_non_unf_lev import *


def extract_features(dataset):
    """Extract the covariates and the Y-vectors corresponding to treatment and control values for each dataset."""
    if dataset == 'boston':
        X, y = pkl.load(open("../datasets/boston_dataset.pkl", "rb"))
        n, d = X.shape
        Y1 = Y0 = y
    elif dataset == 'lalonde':
        X, y = lalonde()
        n, d = X.shape
        Y1 = Y0 = y
    elif dataset == 'ihdp':
        X, Y1, Y0 = ihdp()
        Y0 = Y1
        n, d = X.shape
    elif dataset == 'twins':
        X, Y1, Y0 = twins()
        n, d = X.shape
    elif dataset == 'synthetic':
        X, Y1, Y0 = synthetic()
        n, d = X.shape
    elif dataset == 'synthetic_baseline':
        X, Y1, Y0 = synthetic_baseline()
        n, d = X.shape
    elif dataset == 'synthetic_non_unf_lev':
        X = synthetic_non_unf_lev.generate_synthetic_non_unf_lev()
        n, d = X.shape
        w0 = weightvector(d)

        remaining_norm = LA.norm(X @ w0) ** 2
        eps0 = noisevector(n, d, remaining_norm)

        w1 = weightvector(d)

        remaining_norm = LA.norm(X @ w1) ** 2
        eps1 = noisevector(n, d, remaining_norm)

        # update it with the noise terms
        Y0, Y1 = X @ w0 + eps0, X @ w1 + eps1

    # Row normalize the matrix
    norm_max = 0.0
    for i in range(0, n):
        norm_max = max(norm_max, LA.norm(X[i]))

    for i in range(0, n):
        X[i] = (float(1.0) / norm_max) * X[i]

    binwidth = 8
    data = leverage_score_syn(X)
    data = sorted(data, reverse=True)

    plt.hist(leverage_score_syn(X), weights=np.ones(len(X)) / len(X))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig('../plots/' + dataset + '_lev_scores_hist.jpg')
    plt.close()

    return X, Y1, Y0


def generate_dataset(dataset='boston'):
    """Extract features from a dataset"""
    X, Y1, Y0 = extract_features(dataset)
    return X, Y1, Y0


def ihdp():
    ihdp_dataset = pd.read_csv("../datasets/ihdp_npci_1.csv", header=None)
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]

    for i in range(1, 26):
        col.append("x" + str(i))
    ihdp_dataset.columns = col
    ihdp_dataset.head()

    labelled_data = ihdp_dataset.to_numpy()
    X = labelled_data[:, 5:]

    Y1 = np.zeros(len(X))
    Y0 = np.zeros(len(X))

    for i in range(0, len(X)):
        if labelled_data[i, 0] == 1:
            Y1[i] = labelled_data[i, 1]
            Y0[i] = labelled_data[i, 2]
        else:
            Y0[i] = labelled_data[i, 1]
            Y1[i] = labelled_data[i, 2]

    return X, Y1, Y0


def lalonde():
    data = pd.read_csv('../datasets/lalonde.csv')
    labelled_data = data.to_numpy()

    _, d = labelled_data.shape
    cols = [True for i in range(0, d)]
    cols[8], cols[11] = False, False
    X = labelled_data[:, np.array(cols)]

    cols = [False for i in range(0, d)]
    cols[8] = True
    y = labelled_data[:, np.array(cols)]
    y = np.array(y[:, 0])

    return X, y


def twins(sampling=True):
    # goal is to identify the effect of weight on mortality and twins are a good approximation
    # Based on the weight of the twin, treatment is given or not and outcome variable is the mortality
    # We can use dowhy library to split each pair of twins into two datapoints and copy potential outcomes

    x = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv")

    # The outcome data contains mortality of the lighter and heavier twin
    y = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv")

    # The treatment data contains weight in grams of both the twins
    t = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv")

    # _0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin
    # lighter_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
    #                    'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
    #                    'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
    #                    'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
    #                    'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
    #                    'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
    #                    'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
    #                    'dlivord_min', 'dtotord_min', 'bord_0',
    #                    'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
    #
    # heavier_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
    #                    'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
    #                    'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
    #                    'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
    #                    'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
    #                    'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
    #                    'data_year', 'nprevistq', 'dfageq', 'feduc6',
    #                    'infant_id_1', 'dlivord_min', 'dtotord_min', 'bord_1',
    #                    'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

    data = []
    Y1 = []
    Y0 = []

    cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
            'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
            'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
            'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
            'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
            'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
            'data_year', 'nprevistq', 'dfageq', 'feduc6', 'dlivord_min', 'dtotord_min',
            'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

    for i in range(len(t.values)):

        # select only if both <=2kg
        if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
            continue

        this_instance = list(x.iloc[i][cols].values)
        data.append(this_instance)

        Y0.append(y.iloc[i].values[1])
        Y1.append(y.iloc[i].values[2])

    df = pd.DataFrame(columns=cols, data=data)

    df.fillna(value=df.mean(), inplace=True)  # filling the missing values

    # convert the data frame to a numpy array as covariates
    covariates = df.to_numpy()

    if sampling:
        X_sampled, Y1_sampled, Y0_sampled = [], [], []
        for i in range(0, len(covariates)):
            if random.random() <= 0.10:
                X_sampled.append(covariates[i])
                Y1_sampled.append(Y1[i])
                Y0_sampled.append(Y0[i])

        return np.array(X_sampled), np.array(Y1_sampled), np.array(Y0_sampled)

    return covariates, np.array(Y1), np.array(Y0)


def covariates(n=1000, d=10):
    while True:
        X = np.random.normal(0, 1, size=(n, d))
        if np.linalg.matrix_rank(X) == d:

            # for i in range(0, n):
            # 	X[i] = np.ones(d)
            # return X

            norm_max = 0.0
            for i in range(0, n):
                norm_max = max(norm_max, LA.norm(X[i]))

            for i in range(0, n):
                X[i] = (float(1.0) / norm_max) * X[i]

            return X


def weightvector(d):
    w = np.random.rand(d, 1)
    wnorm = LA.norm(w)

    for i in range(0, d):
        w[i] = float(w[i]) / float(wnorm)

    return w


def noisevector(n, d, remaining_norm):
    # eps = np.array(np.random.normal(0, 1.0/float(n**2), size=n))
    eps = np.array(np.random.normal(0, (0.01 * remaining_norm) / float(n), size=n))
    # eps = np.array(np.random.normal(0, 0.01 / float(n), size=n))
    eps = eps.reshape((n, 1))
    return eps


def synthetic(n=500, d=20):
    X = covariates(n, d)
    w0 = weightvector(d)

    remaining_norm = LA.norm(X @ w0) ** 2
    eps0 = noisevector(n, d, remaining_norm)

    w1 = weightvector(d)

    remaining_norm = LA.norm(X @ w1) ** 2
    eps1 = noisevector(n, d, remaining_norm)

    # update it with the noise terms
    Y0, Y1 = X @ w0 + eps0, X @ w1 + eps1

    return X, Y1, Y0


def synthetic_baseline(n=2000, d=25):

    X = synthetic_non_unf_lev.generate_synthetic_data(n, d)
    w0 = weightvector(d)

    remaining_norm = LA.norm(X @ w0) ** 2
    eps0 = noisevector(n, d, remaining_norm)

    w1 = weightvector(d)

    remaining_norm = LA.norm(X @ w1) ** 2
    eps1 = noisevector(n, d, remaining_norm)

    # update it with the noise terms
    Y0, Y1 = X @ w0 + eps0, X @ w1 + eps1

    return X, Y1, Y0

