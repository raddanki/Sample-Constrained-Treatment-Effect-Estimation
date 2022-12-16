from custom_plot import *
from run_ATE import *
from run_ITE import *

if __name__ == '__main__':
    
    datasets_list = ['boston', 'ihdp', 'lalonde', 'twins', 'synthetic_baseline']
    ntrials = 500
 

    for dataset in datasets_list:

        X, Y1, Y0 = generate_dataset(dataset)
        npoints, _ = X.shape
        print("Processing the dataset: "+dataset)

        samples_list = []
        for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.80, 1.0]:
            samples_list.append(int(x * npoints))

        # ITE estimators
        ITE_results = run_ITE(samples_list, X, Y1, Y0, ntrials=ntrials, percentile=70)
        # pkl.dump(ITE_results, open("results-pickled/ITE_" + dataset + ".pkl", "wb"))
        plot(list(ITE_results), dataset, estimate_type='ITE')
        print("ITE estimation done for the dataset: "+dataset)

        # ATE estimators
        ATE_results = run_ATE(samples_list, X, Y1, Y0, ntrials=ntrials, percentile=70)
        # pkl.dump(ATE_results, open("results-pickled/ATE_" + dataset + ".pkl", "wb"))
        plot(list(ATE_results), dataset, estimate_type='ATE')
        print("ATE estimation done for the dataset: "+dataset)
