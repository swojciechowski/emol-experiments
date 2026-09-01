import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import rankdata

from emol.metrics import BinaryConfusionMatrix
from emol.helpers.plot.radar import radar_factory

from config import *
from datasets import load_keel_dataset

plt.rcParams['text.usetex'] = True

BASE_METRICS = ['balanced_accuracy', 'TPR', 'NPV', 'PPV', 'TNR']
DATASETS = DATASETS_FULL

EMOL_NAME = "MEUS"
METHODS = MEUS_CONFIG

# EMOL_NAME = "MOWS"
# METHODS = MOWS_CONFIG


results_cube = np.empty((len(BASE_METRICS), len(DATASETS), len(BASE_CLASSIFIERS), len(METHODS), DEFAULT_VALIDATION.get_n_splits()))

handles = {}

for ds_idx, ds_name in enumerate(DATASETS):
    X, y = load_keel_dataset(ds_name)

    for s_idx, (train, test) in enumerate(DEFAULT_VALIDATION.split(X, y)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
            for mtd_idx, mtd_name in enumerate(METHODS):
                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}.npy')

                y_pred = np.load(results_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*BASE_METRICS)

                results_cube[:, ds_idx, clf_idx, mtd_idx, s_idx] = metrics

theta = radar_factory(len(BASE_METRICS), frame="polygon")

for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
    ranks = []

    for metric_idx, metric in enumerate(BASE_METRICS):
        rank_mean = []

        for ds_i, ds_name in enumerate(DATASETS):
            results = results_cube[metric_idx, ds_i, clf_idx]
            means = results.mean(axis=-1)
            rank = rankdata(means)
            rank_mean.append(rank)

        rank_mean = np.mean(rank_mean, axis=0)
        ranks.append(rank_mean)

    ranks = np.array(ranks)

    fig, ax = plt.subplots(figsize=(FIG_A4_WIDTH / 2.5, FIG_A4_WIDTH / 2.5), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.1, hspace=0.30, top=0.85, bottom=0.05)

    ax.set_title(f"\\textsc{{{clf_name.lower()}}}", y=-0.2, fontdict={"size": 20})

    ax.grid(ls=":")
    ax.set_ylim(1, 4)
    ax.set_rgrids([1, 2, 3, 4])
    ax.set_varlabels(['BAC'] + BASE_METRICS[1:])

    for d, method in zip(ranks.T, [METHODS_SHORT_NAMES[_] for _ in METHODS]):
        handle = ax.plot(theta, d, lw=1.5, ls="--", label=method)

    out_dir = os.path.join(FIGURES_DIR, f'radars')
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, f"{EMOL_NAME}_{clf_name}_radar.pdf")

    plt.tight_layout()
    plt.savefig(out_file_path, bbox_inches='tight', pad_inches=0.1)

    # then create a new image
    # adjust the figure size as necessary
    fig_leg = plt.figure(figsize=(FIG_A4_WIDTH, FIG_A4_HEIGHT / 15))
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncols=4, labelspacing=2, shadow=False)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.tight_layout()
    out_file_path = os.path.join(out_dir, f"{EMOL_NAME}_legend.pdf")
    fig_leg.savefig(out_file_path, bbox_inches='tight', transparent="True", pad_inches=0)

    plt.close()
    plt.clf()
