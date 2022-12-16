import matplotlib.pyplot as plt
import numpy as np
from ITE import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes


def plot(results, dataset='boston', estimate_type='ATE'):
    colors = ['red', 'blue', 'orange', 'green']
    color_idx = 0
    for baseline in results:

        color = colors[color_idx % 4]
        color_idx += 1

        plt.plot(baseline.samples_list, baseline.mean, label=baseline.sampling_type, color=color)
        plt.fill_between(baseline.samples_list, baseline.std, 2 * np.array(baseline.mean) - np.array(baseline.std), color=color,
                         alpha=0.1)

    plt.xlabel('Sample sizes (as a percentage of population)')
    if estimate_type == 'ITE':
        plt.ylabel('Root Mean Squared Error (RMSE)')
    else:
        plt.ylabel('Deviation Error')

    # plt.axis([0, 100, 0, 1])
    plt.legend(loc='upper right')
    plt.savefig('../plots/' + dataset + '_' + estimate_type + 'error_bar.jpg')
    plt.close()
