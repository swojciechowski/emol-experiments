import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

import numpy as np


ccmap = LinearSegmentedColormap.from_list("ccm", ['w', 'C0'], N=256)

BASE_METRIC_NAMES = ["TPR", "TNR", "PPV", "NPV"]

nbins = 100

M_comb = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
]


def f_tpr(cm):
    return cm[1, 1] / np.sum(cm[1, :])


def f_tnr(cm):
    return cm[0, 0] / np.sum(cm[0, :])


def f_ppv(cm):
    return cm[1, 1] / np.sum(cm[:, 1])


def f_npv(cm):
    return cm[0, 0] / np.sum(cm[:, 0])

n_samples = 100
levels = 5
WIDTH = 8.27

SF = 1.2

RATIOS = [0.5, 0.2, 0.1]

for min_r in RATIOS:
    n_min = int(n_samples * min_r)
    n_maj = n_samples - n_min

    print(f"TOTAL: {n_min + n_maj}: {n_min}, {n_maj}")

    f_space = []

    for t_maj in range(n_maj + 1):
        for t_min in range(n_min + 1):
            cm = np.array([[t_maj, n_maj - t_maj], [n_min - t_min, t_min]])

            f_space.append([f_tpr(cm), f_tnr(cm), f_ppv(cm), f_npv(cm)])

    f_space = np.array(f_space)

    # fix f_space
    f_space = f_space[~np.isnan(f_space).any(axis=1)]

    fig, axs = plt.subplots(2, 2, figsize=(WIDTH * SF , WIDTH * SF / 3))
    axs = iter(axs.flatten())

    fig.suptitle(f"IR: {(n_maj/n_min):.0f}", fontsize=16)

    for comb in M_comb:
        ax = next(axs)

        x, y = f_space[:, comb].T

        k = gaussian_kde(f_space[:, comb].T)
        xi, yi = np.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap=ccmap)

        # ax.hist2d(x, y, bins=10, range=[(-0.1, 1.1), (-0.1, 1.1)], density=True, cmap='Blues')
        #
        ax.scatter(x, y, facecolor='k', marker='x', s=3, linewidths=0.4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(BASE_METRIC_NAMES[comb[0]])
        ax.set_ylabel(BASE_METRIC_NAMES[comb[1]])
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"foo.png", bbox_inches='tight')
    plt.savefig(f"density_{min_r}.pdf", bbox_inches='tight')
    plt.close()
    plt.clf()
