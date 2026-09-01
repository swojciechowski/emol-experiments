import os
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

from tqdm import tqdm

from emol.fbeta.plot import fbeta_plot_combined_cv, fbeta_plot
from emol.metrics import BinaryConfusionMatrix

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

plt.rcParams["text.usetex"] = True

METRICS = ['TPR', "PPV"]
DATASETS = DATASETS_SHORT
METHODS_CFG = [
    ("MEUS", MEUS_CONFIG),
    ("MOWS", MOWS_CONFIG)
]
CLASSIFIERS = ["GNB", 'KNN', 'CRT', 'MLP', "SVM"]

F_BETA_SINGLE_PLOT_DIR = os.path.join(FIGURES_DIR, 'fbeta_moo')
os.makedirs(F_BETA_SINGLE_PLOT_DIR, exist_ok=True)

SPLITS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'splits')
os.makedirs(SPLITS_PLOT_DIR, exist_ok=True)

DS_PLOT_DIR = os.path.join(F_BETA_SINGLE_PLOT_DIR, 'cv')
os.makedirs(DS_PLOT_DIR, exist_ok=True)

for moo_method, moo_config in METHODS_CFG:
    for ds_idx, ds_name in enumerate(DATASETS):
        for clf_idx, clf_name in enumerate(CLASSIFIERS):
            results = []

            for mtd_idx, mtd_name in enumerate(moo_config):
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

                    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                    ax = axs

                    labels = [f"{i}" for i in indicators]

                    fbeta_plot(metrics, color_all=False, labels=labels, ax=ax)

                    out_file = os.path.join(SPLITS_PLOT_DIR, f"{ds_name}-{clf_name}-{METHODS_SHORT_NAMES[mtd_name]}-{s_idx:02d}.png")
                    plt.savefig(out_file, bbox_inches='tight')
                    plt.close()


                results.append(split_results)

            fig, axs = plt.subplots(1, 1, figsize=(6, 4))

            ax = axs

            labels = [METHODS_SHORT_NAMES[n] for n in moo_config]

            fbeta_plot_combined_cv(results, color_all=True, colors=['red', 'blue', 'green', 'orange'], labels=labels, ax=ax)

            out_file = os.path.join(DS_PLOT_DIR, f"{ds_name}-{clf_name}-{moo_method}.png")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
