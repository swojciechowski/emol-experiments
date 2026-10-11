import os
import sys

import numpy as np
from emol.metrics import BinaryConfusionMatrix

import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
from emol.moo.evaluation.metrics import DR, GD, HV, non_dominated_solutions

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

METRICS = ["TPR", "TNR"]
# METRICS = ["TPR", "PPV"]
DATASETS = DATASETS_SHORT[:]
METHODS = IMBL_UNDERSAMPLERS[:]
MOO_METHOD = MEUS_SEL
CLASSIFIERS = ["GNB"]

METRIC_NAMES = ["HV", "GD", "DR"]

boxplot_style = {
    "patch_artist": True,
    "boxprops": {
        "facecolor": "none",
        "edgecolor": "black",
        "linewidth": 1,
    },
    "flierprops": {
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgecolor": "k",
        "markersize": 2,
    },
    "whiskerprops": {"color": "black", "linewidth": 1},
    "capprops": {"color": "black", "linewidth": 1.5},
    "medianprops": {"color": "black", "linewidth": 2, "path_effects": [
            path_effects.withStroke(
            linewidth=3.5,
            foreground="black",
        )
        ],},
}


out_dir = os.path.join(FIGURES_DIR, f"moo_metrics")
os.makedirs(out_dir, exist_ok=True)

for clf_idx, clf_name in enumerate(CLASSIFIERS):
    clf_results = []

    for ds_idx, ds_name in enumerate(DATASETS):
        ds_results = []

        fig, axs = plt.subplots(
            3,
            5,
            figsize=(FIG_A4_WIDTH, FIG_A4_HEIGHT * 3 / 5),
            layout="constrained",
            sharey='row',
        )

        fig.suptitle(DATASETS_IDX[DATASETS[ds_idx]], fontsize=16)

        axs_i = iter(axs[:2, :5].T.flatten())


        for s_idx in range(N_SPLITS):
            ax = next(axs_i)
            # ax.set_xscale("function", functions=(partial(np.power, 10.0), np.log10))
            # ax.set_yscale("function", functions=(partial(np.power, 10.0), np.log10))
            ax.set_xlabel(METRICS[0])
            ax.set_ylabel(METRICS[1])

            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_xticks((0.0, 0.5, 0.75, 1.0))
            ax.set_yticks((0.0, 0.5, 0.75, 1.0))
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_aspect("equal")
            ax.grid(ls=":")

            y_test = np.load(
                os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy")
            )

            results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
            record_name = f"{MOO_METHOD}_{clf_name}"
            results_file = os.path.join(results_dir, f"{record_name}_prd.npy")
            y_pred = np.load(results_file)
            moo_results = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T

            # Select non-dominated solutions
            moo_results = moo_results[non_dominated_solutions(moo_results)].T

            ax.scatter(*moo_results, c="k", marker='x', s=17)

            ax.set_title(f"Split: {s_idx + 1}")


            methods_results = []

            for mtd_idx, mtd_name in enumerate(METHODS):
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f"{record_name}.npy")
                y_pred = np.load(results_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS)

                ax.scatter(
                    metrics[0],
                    metrics[1],
                    marker=".",
                    edgecolor='k',
                    linewidth=0.5,
                    # facecolor=f"C{mtd_idx}",
                    facecolor=mcolors.to_rgba(f"C{mtd_idx}", alpha=0.7),
                    s=100,
                    label=f"{METHODS_SHORT_NAMES[mtd_name]}",
                )

                hv = HV(moo_results.T, metrics)
                gd = GD(moo_results.T, metrics, ideal_ref=True)
                dr = DR(moo_results.T, metrics, strict=False)

                methods_results.append([hv, gd, dr])

            ds_results.append(methods_results)

        for ax in axs[2, :]:
            fig.delaxes(ax)

        gs = axs[0, 0].get_subplotspec().get_gridspec()
        gs = gs[2, :].subgridspec(1, 3)

        for j in range(3):
            ax = fig.add_subplot(gs[0, j])

            ax.grid(ls=":")

            boxp = ax.boxplot(
                np.array(ds_results)[..., j], orientation="horizontal", **boxplot_style
            )

            for i, patch in enumerate(boxp["medians"]):
                patch.set_color(f"C{i}")
                # patch.set_alpha(0.7)

            for i, patch in enumerate(boxp["boxes"]):
                patch.set_facecolor(f"C{i}")
                patch.set_alpha(0.7)

            ax.set_yticks(range(1, len(METHODS) + 1))
            ax.set_yticklabels([f"{METHODS_SHORT_NAMES[m]}" for m in METHODS])

            ax.set_title(METRIC_NAMES[j])

            out = os.path.join(out_dir, f"{DATASETS[ds_idx]}-{METRIC_NAMES[j]}.pdf")
            plt.savefig(out, bbox_inches="tight")
            plt.savefig("foo.png", bbox_inches="tight")

        plt.close()
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
                d[::-1], orientation="horizontal", **boxplot_style
            )

            for i, patch in enumerate(boxp["medians"]):
                patch.set_color(f"C{i}")
                # patch.set_alpha(0.7)

            for i, patch in enumerate(boxp["boxes"]):
                patch.set_facecolor(f"C{i}")
                patch.set_alpha(0.7)

            ax.set_yticks(range(1, len(METHODS) + 1))
            ax.set_yticklabels(
                [
                    METHODS_SHORT_NAMES[m]
                    for m in METHODS
                ]
            )

            plt.savefig("foo.png", bbox_inches="tight")

        out_file_dir = os.path.join(out_dir, f"{m_name}.pdf")

        plt.savefig(out_file_dir, bbox_inches="tight")
        plt.clf()
