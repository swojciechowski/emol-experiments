import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from emol.fbeta.plot import fbeta_plot, fbeta_plot_cv
from emol.metrics import BinaryConfusionMatrix

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

# Custom config
METRICS = ['TPR', "PPV"]
DATASETS = DATASETS_SHORT
METHODS = SMOTEV_FULL
CLASSIFIERS = ["GNB", 'KNN', 'CRT', 'MLP', "SVM"]

F_BETA_SINGLE_PLOT_DIR = os.path.join(FIGURES_DIR, 'fbeta_sv')
os.makedirs(F_BETA_SINGLE_PLOT_DIR, exist_ok=True)

SPLITS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'splits')
os.makedirs(SPLITS_PLOT_DIR, exist_ok=True)

DS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'cv')
os.makedirs(DS_PLOT_DIR, exist_ok=True)

for ds_idx, ds_name in enumerate(DATASETS):

    for clf_idx, clf_name in enumerate(CLASSIFIERS):
        split_results = []

        for s_idx in range(N_SPLITS):
            y_test = np.load(os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy"))

            results = []

            for mtd_idx, mtd_name in enumerate(METHODS):
                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}.npy')
                y_pred = np.load(results_file)
                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS)

                results.append(metrics)

            split_results.append(results)

            results = np.array(results)

            fig, axs = plt.subplots(1, 1, figsize=(6, 4))
            ax = axs

            fbeta_plot(results, color_all=False, colors=None, labels=METHODS, ax=ax)

            out_file = os.path.join(SPLITS_PLOT_DIR, f"{ds_name}-{clf_name}-{s_idx:02d}.png")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()

        split_results = np.array(split_results)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4))

        fig.suptitle(ds_name)

        ax = axs

        fbeta_plot_cv(split_results, color_all=False, colors=None, labels=METHODS, ax=ax)

        out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}-{clf_name}.png")
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
