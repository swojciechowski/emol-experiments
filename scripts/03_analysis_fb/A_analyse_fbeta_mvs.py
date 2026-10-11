import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm

from emol.fbeta.plot import fbeta_plot, fbeta_plot_combined_ref
from emol.helpers.plot.color import DEFAULT_CMAP
from emol.metrics import BinaryConfusionMatrix

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *


METRICS = ['TPR', "PPV"]
DATASETS = DATASETS_SHORT

REF_CONFIGS = [
    (MOL_METHODS[0], IMBL_UNDERSAMPLERS),
    # (MOL_METHODS[1], SMOTEV_TOP),
]

CLASSIFIERS = ["GNB"]

F_BETA_SINGLE_PLOT_DIR = os.path.join(FIGURES_DIR, 'fbeta_single')
os.makedirs(F_BETA_SINGLE_PLOT_DIR, exist_ok=True)

SPLITS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'moo_splits')
os.makedirs(SPLITS_PLOT_DIR, exist_ok=True)

REFS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'refs')
os.makedirs(REFS_PLOT_DIR, exist_ok=True)

CV_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'cv_final')
os.makedirs(CV_PLOT_DIR, exist_ok=True)


for ds_idx, ds_name in enumerate(DATASETS):
    for clf_idx, clf_name in enumerate(CLASSIFIERS):
        results = []

        for moo_mtd, ref_mtds in REF_CONFIGS:
            split_results = []
            split_ref_results = []

            for s_idx in range(N_SPLITS):
                y_test = np.load(os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy"))

                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{moo_mtd}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}_prd.npy')
                y_pred = np.load(results_file)

                indicators_file = os.path.join(results_dir, f'{record_name}_ind.npy')
                indicators = np.load(indicators_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T
                split_results.append(metrics)

                ref_results = []

                for mtd_idx, mtd_name in enumerate(ref_mtds):
                    results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                    record_name = f"{mtd_name}_{clf_name}"
                    results_file = os.path.join(results_dir, f'{record_name}.npy')
                    y_pred = np.load(results_file)
                    ref_metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS)
                    ref_results.append(ref_metrics)

                ref_results = np.array(ref_results)
                split_ref_results.append(ref_results)

                # Plot by moo splits

                fig, ax = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2, FIG_A4_HEIGHT * 1 / 10))

                fig.suptitle(DATASETS_IDX[ds_name], y=1.02)

                # colors = generate_distinct_colors(100)
                # colors = ['k'] * 100

                # cmap = plt.colormaps["coolwarm"]
                cmap = mcolors.LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
                colors = cmap(np.linspace(0, 1, len(metrics)))  # Returns an (N, 4) RGBA array
                colors = [mcolors.to_hex(c) for c in colors]

                metrics = metrics[np.lexsort((metrics[..., 0], -metrics[..., 1]))]

                labels = [f"{-1 * a:.2f}-{-1 * b:.2f}" for a,b in indicators]
                fbeta_plot(metrics, color_all=False, colors=colors, ax=ax)

                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)

                out_file = os.path.join(SPLITS_PLOT_DIR, f"{DATASETS_IDX[ds_name]}-{clf_name}-{METHODS_SHORT_NAMES[moo_mtd]}-{s_idx:02d}.pdf")

                plt.savefig(out_file, bbox_inches='tight')

                if s_idx == 0 and clf_idx == 0:
                    out_file = os.path.join(SPLITS_PLOT_DIR, f"{ds_name}.pdf")
                    plt.savefig(out_file, bbox_inches='tight')

                plt.savefig("foo.png", bbox_inches='tight')
                plt.close()

                # Plot reference

                fig, ax = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2, FIG_A4_HEIGHT * 1 / 9))

                labels = [METHODS_SHORT_NAMES[m].upper() for m in ref_mtds]

                fig.suptitle(DATASETS_IDX[ds_name], y=1.02)

                fbeta_plot_combined_ref(ref_results, metrics, color_all=False, colors=DEFAULT_CMAP, labels=labels, ax=ax)

                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)

                out_file = os.path.join(REFS_PLOT_DIR, f"{DATASETS_IDX[ds_name]}-{clf_name}-{METHODS_SHORT_NAMES[moo_mtd]}-{s_idx:02d}.pdf")

                plt.savefig(out_file, bbox_inches='tight')

                if s_idx == 0 and clf_idx == 0:
                    out_file = os.path.join(REFS_PLOT_DIR, f"{ds_name}.pdf")
                    plt.savefig(out_file, bbox_inches='tight')

                plt.savefig("foo.png", bbox_inches='tight')
                plt.close()

            results.append(split_results)


        # S = 0.75
        # W, H = FIG_A4_WIDTH / (2 * S), FIG_A4_HEIGHT / (7 * S)
        # fig, axs = plt.subplots(1, 1, figsize=(W, H))
        # fig.suptitle(DATASETS_IDX[ds_name])

        # ax = axs

        # fbeta_plot_combined_cv(results, color_all=True, colors=['red', 'blue'], ax=ax)

        # # out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}.png")
        # # plt.savefig(out_file, bbox_inches='tight')
        # out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}.pdf")
        # plt.savefig(out_file, bbox_inches='tight')
        # plt.close()
