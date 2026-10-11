import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from emol.fbeta.plot import fbeta_plot_combined_cv, fbeta_plot
from emol.metrics import BinaryConfusionMatrix

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *


METRICS = ['TPR', "PPV"]
DATASETS = DATASETS_SHORT
METHODS_CFG = MOL_METHODS
CLASSIFIERS = ["GNB"]

F_BETA_SINGLE_PLOT_DIR = os.path.join(FIGURES_DIR, 'fbeta_moo')
os.makedirs(F_BETA_SINGLE_PLOT_DIR, exist_ok=True)

# SPLITS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'splits')
# os.makedirs(SPLITS_PLOT_DIR, exist_ok=True)

DS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'cv_final')
os.makedirs(DS_PLOT_DIR, exist_ok=True)


for ds_idx, ds_name in enumerate(DATASETS):
    for clf_idx, clf_name in enumerate(CLASSIFIERS):
        results = []

        for mtd_idx, mtd_name in enumerate(METHODS_CFG):
            split_results = []

            for s_idx in range(N_SPLITS):
                y_test = np.load(os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy"))

                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}_prd.npy')
                y_pred = np.load(results_file)

                indicators_file = os.path.join(results_dir, f'{record_name}_ind.npy')
                indicators = np.load(indicators_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T
                split_results.append(metrics)


            results.append(split_results)


        W, H = FIG_A4_WIDTH / 2, FIG_A4_HEIGHT / 9
        fig, axs = plt.subplots(1, 1, figsize=(W, H))
        fig.suptitle(DATASETS_IDX[ds_name], y=1.05)

        ax = axs

        fbeta_plot_combined_cv(results, color_all=True, colors=['red', 'blue'], labels=["MEUS", "MOWS"], ax=ax)

        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        ax.legend()

        legend = ax.get_legend()
        if legend is None:
            raise ValueError("The provided axis does not have a legend to export.")

        handles, labels = ax.get_legend_handles_labels()

        # 2. Create a standalone figure for the legend
        fig_leg = plt.figure(figsize=(FIG_A4_WIDTH, FIG_A4_HEIGHT / 30))
        leg = fig_leg.legend(handles, labels, loc='center', ncol=5, frameon=True)
        frame = leg.get_frame()
        frame.set_boxstyle("square")
        plt.axis('off')

        # 3. Save the legend as a PNG
        fig_leg.savefig("legend.pdf", bbox_inches='tight', transparent=True, pad_inches=0.1)
        fig_leg.savefig("legend.png", bbox_inches='tight', transparent=True, pad_inches=0.1)
        plt.close(fig_leg)

        # 4. Remove the legend from the original axis
        legend.remove()

        # out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}.png")
        # plt.savefig(out_file, bbox_inches='tight')
        out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}.pdf")
        plt.savefig(out_file, bbox_inches='tight')
        plt.savefig("foo.png", bbox_inches='tight')
        plt.close()
