import os
import numpy as np

from emol.metrics import BinaryConfusionMatrix

from config import *
from datasets import load_keel_dataset
from functools import partial

from emol.moo.evaluation.metrics import HV, GD, DR, non_dominated_solutions
from tabulate import tabulate

import matplotlib.pyplot as plt

METRICS = ["TPR", "TNR"]
DATASETS = DATASETS_SHORT[:]
METHODS = IMBL_UNDERSAMPLERS[:]
MOO_METHOD = MEUS_SEL
CLASSIFIERS = ["GNB"]

METRIC_NAMES = ["HV", "GD", "DR"]


boxplot_style = {
    # 'whiskerprops': {'linewidth': 2, 'color': 'orange'},
    'medianprops': {'linewidth': 3, 'color': 'k'},
    'flierprops': {'marker': 'o', 'markerfacecolor': 'none', 'markeredgecolor': 'k', 'markersize': 3},
    'boxprops': {'edgecolor': 'none', 'linewidth': 1},
    'capprops': {'color': 'black', 'linewidth': 1}
}


out_dir = os.path.join(FIGURES_DIR, f'moo_metrics')
os.makedirs(out_dir, exist_ok=True)

for clf_idx, clf_name in enumerate(CLASSIFIERS):
    clf_results = []

    for ds_idx, ds_name in enumerate(DATASETS):
        X, y = load_keel_dataset(ds_name)

        ds_results = []

        fig, axs = plt.subplots(4, 3, figsize=(FIG_A4_HEIGHT * 2 / 3, FIG_A4_WIDTH), sharex=True, sharey=True)
        fig.suptitle(DATASETS_IDX[DATASETS[ds_idx]], fontsize=16, y=0.92)

        axs_i = iter(axs.T.flatten())

        for s_idx, (train, test) in enumerate(DEFAULT_VALIDATION.split(X, y)):
            ax = next(axs_i)
            ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
            ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
            ax.set_xlabel("TPR")
            ax.set_ylabel("PPV")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_xticks((0.0, 0.5, 0.75, 1.0))
            ax.set_yticks((0.0, 0.5, 0.75, 1.0))
            ax.set_aspect("equal")
            ax.grid(ls=":")

            X_test, y_test = X[test], y[test]
            results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
            record_name = f"{MOO_METHOD}_{clf_name}"
            results_file = os.path.join(results_dir, f"{record_name}_prd.npy")
            y_pred = np.load(results_file)
            moo_results = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T

            # Select non-dominated solutions
            moo_results = moo_results[non_dominated_solutions(moo_results)].T

            ax.scatter(*moo_results, c='k', s=17)

            methods_results = []

            for mtd_idx, mtd_name in enumerate(METHODS):
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f"{record_name}.npy")
                y_pred = np.load(results_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS)

                ax.scatter(metrics[0], metrics[1], marker='x', s=17, label=f"{METHODS_SHORT_NAMES[mtd_name].split("{")[1].split("}")[0]}", alpha=0.7)

                hv = HV(moo_results.T, metrics)
                gd = GD(moo_results.T, metrics, ideal_ref=True)
                dr = DR(moo_results.T, metrics)

                methods_results.append([hv, gd, dr])

            ds_results.append(methods_results)

        plt.legend(ncols=5, loc='lower center')

        gs = axs[0, 0].get_subplotspec().get_gridspec()
        # 2. Remove the individual axes you want to combine
        axs[2, 2].remove()
        axs[3, 2].remove()

        for j in range(3):
            ax = fig.add_subplot(gs[2:4, 2])

            ax.grid(ls=":")

            boxp = ax.boxplot(
                np.array(ds_results)[:, :, j],
                orientation="horizontal",
                patch_artist=True,
                **boxplot_style
            )

            for i, patch in enumerate(boxp["boxes"]):
                patch.set_facecolor(f"C{i}")
                patch.set_alpha(0.7)

            ax.set_yticks(range(1, len(METHODS) + 1))
            ax.set_yticklabels(
                [
                    f"{METHODS_SHORT_NAMES[m].split("{")[1].split("}")[0]}"
                    for m in METHODS
                ]
            )

            ax.set_xlabel(METRIC_NAMES[j])

            # 3. Create a new merged axis spanning both original slots (Row 0, Columns 0 and 1)
            # ax_merged = fig.add_subplot(gs[0, :])

            out = os.path.join(out_dir, f"{DATASETS[ds_idx]}-{METRIC_NAMES[j]}.pdf")
            plt.savefig(out, bbox_inches='tight')
            plt.savefig("foo.png", bbox_inches='tight')
            ax.remove()

        clf_results.append(ds_results)

    clf_results = np.array(clf_results)

    for m_i, m_name in enumerate(METRIC_NAMES):
        results = clf_results[..., m_i]

        fig, axs = plt.subplots(
            2,
            int(results.shape[0] / 2),
            figsize=(FIG_A4_WIDTH * 2, FIG_A4_HEIGHT / 3 * 2),
            sharex=True,
            sharey=True,
        )

        fig.suptitle(m_name, fontsize=16, y=0.95)

        axs = iter(axs.T.flatten())

        for i, d in enumerate(results):
            ax = next(axs)
            ax.grid(ls=":")

            ax.set_title(DATASETS_IDX[DATASETS[i]])

            boxp = ax.boxplot(
                d,
                orientation="horizontal",
                patch_artist=True,
                **boxplot_style
            )

            for i, patch in enumerate(boxp["boxes"]):
                patch.set_facecolor(f"C{i}")
                patch.set_alpha(0.7)

            ax.set_yticks(range(1, len(METHODS) + 1))
            ax.set_yticklabels(
                [
                    f"{METHODS_SHORT_NAMES[m].split("{")[1].split("}")[0]}"
                    for m in METHODS
                ]
            )

            plt.savefig("foo.png", bbox_inches='tight')


        out_file_dir = os.path.join(out_dir, f"{m_name}.pdf")

        plt.savefig(out_file_dir, bbox_inches='tight')
        plt.clf()
