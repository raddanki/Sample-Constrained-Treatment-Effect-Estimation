from typing import List

from data import *
from julia import GSWDesign


def normalize(prob: List[float]) -> List[float]:
    """Normalize the probabilities to make it a distribution"""
    return prob / prob.sum()


def GSW(X):
    """ Gram-Schmidt-Walk design -- a call to the previous implementation with robustness and balance parameter set to 0.5"""
    GSWDesign.X = np.array(X)
    GSWDesign.lamda = 0.5
    z = GSWDesign.sample_gs_walk(GSWDesign.X, GSWDesign.lamda)
    # print(z)
    return z


def rev_idx_map(X, part):
    """Reverse index map that selects a part of the dataset and stores the mapping of the original row """
    rmap = dict({})
    part_idx = 0
    Xaug = []

    # rmap stores the mapping of the new dataset Xaug to the original dataset X
    for i in range(0, len(part)):
        Xaug.append(X[part[part_idx]])
        rmap[i] = part[part_idx]
        part_idx = part_idx + 1

    return Xaug, rmap


def coreset(X, s):
    """Coreset construction uses recursive GSW and in each call selects the smallest one to recurse"""
    n = len(X)
    part = [i for i in range(0, n)]
    niter = 0
    while True:
        niter += 1

        # part refers to the partition on which we recurse in the current call
        Xaug, rmap = rev_idx_map(X, part)
        sign = GSW(Xaug)
        part1, part0 = [], []

        # we split the partition into two parts based on the sign of the assignment vector (obtained by GSW)
        for idx in range(0, len(sign)):
            if sign[idx]:
                part1.append(rmap[idx])
            else:
                part0.append(rmap[idx])

        # termination condition
        if n <= 2 * s:
            return part1[0:min(int(s / 2), len(part1))], part0[0:min(int(s / 2), len(part0))], niter

        if len(part1) >= len(part0):
            n = len(part0)
            part = part0

        if len(part1) < len(part0):
            n = len(part1)
            part = part1


def unif_sampling(X, s):
    # """Uniform sampling of rows with replacement"""
    # n, d = X.shape
    # uscores = np.ones(n)
    #
    # pi = normalize(uscores)
    # sampled_rows = np.random.choice(range(n), s, p=pi)  # sampling with replacement

    """Uniform sampling of rows with replacement"""
    n, d = X.shape
    pi = [float(s)/float(n) for i in range(0, n)]
    sampled_rows = []

    for i in range(0, n):
        if random.random() <= float(s)/float(n):
            sampled_rows.append(i)

    return list(set(sampled_rows)), pi


def unif_estimator(X, Y1, Y0, s=0):
    """Estimator for uniform sampling. This is one of the baselines"""
    n, d = X.shape
    tauS = 0

    sampled_rows, pi = unif_sampling(X, s)

    tau1, tau0 = 0.0, 0.0

    # HT estimator with inverse probability weighting to make it unbiased.
    # for i in range(0, len(sampled_rows)):
    #     if np.random.rand() <= 0.5:
    #         tau1 += (2.0 / pi[sampled_rows[i]]) * Y1[sampled_rows[i]]
    #     else:
    #         tau0 += (2.0 / pi[sampled_rows[i]]) * Y0[sampled_rows[i]]
    #
    # tauS = (1.0 / float(s * n)) * (tau1 - tau0)

    for i in range(0, len(sampled_rows)):
        if np.random.rand() <= 0.5:
            tau1 += (2.0 / pi[sampled_rows[i]]) * Y1[sampled_rows[i]]
        else:
            tau0 += (2.0 / pi[sampled_rows[i]]) * Y0[sampled_rows[i]]

    tauS = (1.0 / float(n)) * (tau1 - tau0)

    return tauS


def ATE_gsw_estimator(X, Y1, Y0, s=0):
    """ATE estimator on the coreset using just recursive calls to GSW. This is the estimator of our algorithm."""
    n, d = X.shape
    part1, part0, niter = coreset(X, s)
    niter1, niter0 = niter, niter

    tau1, tau0 = 0.0, 0.0

    for idx in part1:
        tau1 += Y1[idx]

    for idx in part0:
        tau0 += Y0[idx]

    # HT estimator
    tauS = (1.0 / float(n)) * (2 ** niter1 * (tau1) - 2 ** niter0 * (tau0))

    return tauS


def ATE_gsw_pop_estimator(X, Y1, Y0):
    """ATE estimator on the entire population using just a single call to GSW. This is one of the population level baselines for comparison."""
    n, d = X.shape
    sign = GSW(X)
    part1, part0 = [], []
    for idx in range(0, len(sign)):
        if sign[idx]:
            part1.append(idx)
        else:
            part0.append(idx)

    tau1, tau0 = 0.0, 0.0

    for idx in part1:
        tau1 += Y1[idx]

    for idx in part0:
        tau0 += Y0[idx]

    tauS = (1.0 / float(n)) * (2.0 * (tau1) - 2.0 * (tau0))

    return tauS


def ATE_rand_pop_estimator(X, Y1, Y0):
    """ATE using complete randomization. This is another population level baseline."""
    n, d = X.shape
    part1, part0 = [], []
    for idx in range(0, n):
        if np.random.rand() <= 0.5:
            part1.append(idx)
        else:
            part0.append(idx)

    tau1, tau0 = 0.0, 0.0

    for idx in part1:
        tau1 += Y1[idx]

    for idx in part0:
        tau0 += Y0[idx]

    # HT estimator using GSW on the entire population
    tauS = (1.0 / float(n)) * (2.0 * (tau1) - 2.0 * (tau0))

    return tauS


def ATE(Y1, Y0):
    """ATE assuming access to treatment and control values for the entire population"""
    n = len(Y1)
    tau = 0.0
    for i in range(0, n):
        tau += Y1[i] - Y0[i]

    tau = (float(1.0) / float(n)) * tau
    return tau
