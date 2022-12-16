from ITE import *


class ITEDesignEstimates:
    """ITEDesignEstimates is used for storing various estimator outputs and used for plotting"""

    def __init__(self, mean, std, samples_list, sampling_type):
        self.mean = mean
        self.std = std
        self.samples_list = samples_list
        self.sampling_type = sampling_type


def run_ITE_regression_estimator(samples_list, X, Y1, Y0):

    n, d = X.shape
    error_reg, error_reg_std = [], []

    beta1 = np.linalg.lstsq(X, Y1, rcond=None)[0]
    beta0 = np.linalg.lstsq(X, Y0, rcond=None)[0]

    ITE_est = X @ beta1 - X @ beta0
    ITE = Y1 - Y0

    for sample in samples_list:
        error_reg.append(LA.norm(ITE_est - ITE) * (1.0 / float(math.sqrt(n))))
        error_reg_std.append(LA.norm(ITE_est - ITE) * (1.0 / float(math.sqrt(n))))

    return error_reg, error_reg_std


def run_ITE(samples_list, X, Y1, Y0, ntrials=100, percentile=70):
    """Obtain estimators for various ITE estimators"""
    n, ndim = X.shape
    error_unf, error_lev, error_lev_nothres = [], [], []
    error_unf_std, error_lev_std, error_lev_nothres_std = [], [], []
    samples_percentage = []

    for nsamples in samples_list:

        # print("samples: " + str(nsamples) + " has started")
        samples_percentage.append(int(float(nsamples * 100) / n))

        error_unf_trials, error_lev_trials, error_lev_nothres_trials = [], [], []
        error_baseline = []

        gamma = (float(nsamples) / float(2.0 * ndim))
        # gamma = (float(2.0 * nsamples) / float(ndim))
        threshold = gamma

        Xplus = thresh_SVD(X, threshold)

        for trial in range(0, ntrials):
            ITE_est, ITE = ITE_estimator(X, nsamples, Y1, Y0, gamma, unif_sampling)
            error_unf_trials.append(LA.norm(ITE_est - ITE) * (1.0 / float(math.sqrt(n))))

            ITE_est, ITE = ITE_estimator(X, nsamples, Y1, Y0, gamma, lev_bernoulli_sampling)
            error_lev_nothres_trials.append(LA.norm(ITE_est - ITE) * (1.0 / float(math.sqrt(n))))

            ITE_est, ITE = ITE_estimator(Xplus, nsamples, Y1, Y0, gamma, lev_bernoulli_sampling)
            error_lev_trials.append(LA.norm(ITE_est - ITE) * (1.0 / float(math.sqrt(n))))

        error_unf.append(np.mean(error_unf_trials))
        error_unf_std.append(np.percentile(error_unf_trials, percentile))

        error_lev.append(np.mean(error_lev_trials))
        error_lev_std.append(np.percentile(error_lev_trials, percentile))

        error_lev_nothres.append(np.mean(error_lev_nothres_trials))
        error_lev_nothres_std.append(np.percentile(error_lev_nothres_trials, percentile))

    unf = ITEDesignEstimates(error_unf, error_unf_std, samples_percentage, "Uniform")
    lev = ITEDesignEstimates(error_lev, error_lev_std, samples_percentage, "Leverage")
    lev_nothres = ITEDesignEstimates(error_lev_nothres, error_lev_nothres_std, samples_percentage,
                                     "Leverage-nothresh")

    error_reg, error_reg_std = run_ITE_regression_estimator(samples_list, X, Y1, Y0)
    regression = ITEDesignEstimates(error_reg, error_reg_std, samples_percentage, "Lin-regression")
    return unf, lev, lev_nothres, regression
    # return lev, regression
