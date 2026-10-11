import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from tqdm import tqdm
from itertools import product

ccmap = LinearSegmentedColormap.from_list("ccm", ['w', 'C0'], N=256)

BASE_METRIC_NAMES = ["TPR", "TNR", "PPV", "NPV"]

M_comb = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
]

from scipy.ndimage import gaussian_filter

def fast_2d_kde(
    x,
    y,
    grid_size=250,
    sigma=10.0,
    bounds=((0, 1), (0, 1)),
    normalize=False,
):
    H, xedges, yedges = np.histogram2d(x, y, bins=grid_size, range=bounds)
    pad_width = int(np.ceil(sigma * 3))
    padded_H = np.pad(H, pad_width=pad_width, mode="reflect")
    smoothed_padded = gaussian_filter(padded_H, sigma=sigma)
    density = smoothed_padded[pad_width:-pad_width, pad_width:-pad_width]

    return density

def confusion_matrix(y_true, y_pred, n_classes=2):
    y = n_classes * y_true + y_pred
    y = np.bincount(y, minlength=(n_classes * n_classes))
    return y.reshape(n_classes, n_classes)

def f_tpr(cm):
    return cm[1, 1] / np.sum(cm[1, :])


def f_tnr(cm):
    return cm[0, 0] / np.sum(cm[0, :])


def f_ppv(cm):
    return cm[1, 1] / np.sum(cm[:, 1])


def f_npv(cm):
    return cm[0, 0] / np.sum(cm[:, 0])

n_samples = 15
cm_samples = 100
nbins = 200

WIDTH = 8.27
SF = 1.2

RATIOS = [0.5, 0.2, 0.1]

for min_r in RATIOS:
    n_min = int(n_samples * min_r)
    n_maj = n_samples - n_min

    ir = n_maj / n_min

    y_true = np.array([0] * n_maj + [1] * n_min)

    f_space = []

    for _ in tqdm(product([0, 1], repeat=n_samples), total=2^n_samples):
        cm = confusion_matrix(y_true, _)
        f_space.append([f_tpr(cm), f_tnr(cm), f_ppv(cm), f_npv(cm)])

    f_space = np.array(f_space)

    # fix f_space
    # f_space[np.isnan(f_space).any(axis=1)] = 0
    f_space = f_space[~np.isnan(f_space).any(axis=1)]

    cm_space = []

    n_min = int(cm_samples * min_r)
    n_maj = cm_samples - n_min

    for t_maj in range(n_maj + 1):
        for t_min in range(n_min + 1):
            cm = np.array([[t_maj, n_maj - t_maj], [n_min - t_min, t_min]])

            cm_space.append([f_tpr(cm), f_tnr(cm), f_ppv(cm), f_npv(cm)])

    cm_space = np.array(cm_space)

    # fix cm_space
    # cm_space[np.isnan(cm_space).any(axis=1)] = 0
    cm_space = cm_space[~np.isnan(cm_space).any(axis=1)]



    fig, axs = plt.subplots(2, 2, figsize=(WIDTH * SF , WIDTH * SF))
    axs = iter(axs.flatten())

    fig.suptitle(f"IR: {ir:.0f}", fontsize=16)

    for comb in M_comb:
        ax = next(axs)

        x, y = f_space[:, comb].T
        cm_x, cm_y = cm_space[:, comb].T

        k = gaussian_kde(f_space[:, comb].T)

        xi, yi = np.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]

        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = zi.reshape(xi.shape)

        # zi = fast_2d_kde(x, y)

        ax.imshow(
            zi.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap=ccmap,
            aspect="equal",
        )

        # ax.hist2d(x, y, bins=100, range=[(0, 1), (0, 1)], density=True, cmap=ccmap)

        ax.scatter(cm_x, cm_y, facecolor='k', marker='x', s=3, linewidths=0.4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(BASE_METRIC_NAMES[comb[0]])
        ax.set_ylabel(BASE_METRIC_NAMES[comb[1]])
        ax.set_aspect("equal")

        plt.savefig(f"foo.png", bbox_inches='tight')

    plt.savefig(f"density_all_{min_r}.pdf", bbox_inches='tight')
    plt.close()
    plt.clf()
