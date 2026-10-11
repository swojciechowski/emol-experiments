from critdd import Diagrams
import numpy as np
import pandas as pd
import os
import sys

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from helpers import load_keel_dataset
from emol.metrics import BinaryConfusionMatrix


METRICS = ["balanced_accuracy", "gmean", "f1_score"]
DATASETS = DATASETS_FULL

# EMOL_NAME = "MEUS"
# METHODS = MEUS_RANKING

EMOL_NAME = "MOWS"
METHODS = MOWS_RANKING

results_cube = np.empty((len(METRICS), len(DATASETS), len(BASE_CLASSIFIERS), len(METHODS), DEFAULT_VALIDATION.get_n_splits()))

for ds_idx, ds_name in enumerate(DATASETS):
    for s_idx in range(N_SPLITS):
        y_test = np.load(
            os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy")
        )

        for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
            for mtd_idx, mtd_name in enumerate(METHODS):
                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}.npy')

                y_pred = np.load(results_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS)

                results_cube[:, ds_idx, clf_idx, mtd_idx, s_idx] = metrics


for m_idx, (metric_name, metric_results) in enumerate(zip(METRICS, results_cube)):
    results = np.mean(results_cube[m_idx], axis=-1)

    Xs = [] # collect an (n,k)-shaped matrix for each diagram

    for clf_i, clf_name in enumerate(BASE_CLASSIFIERS):
        V = results[:, clf_i, :]
        Xs.append(V)

    T_NAMES = [METHODS_SHORT_NAMES[m] for m in METHODS]

    mdiagram = Diagrams(
        np.stack(Xs),
        diagram_names = BASE_CLASSIFIERS,
        treatment_names = T_NAMES,
        maximize_outcome = False
    )

    out_dir = os.path.join(FIGURES_DIR, f'rankings')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{EMOL_NAME}_{metric_name}_rank.tikz')

    with open(out_file, 'w') as fp:
        fp.write(mdiagram.to_str(alpha = .05, adjustment = "holm", axis_options = {
            "cycle list": """
    {red,mark=*},
    {black,mark=o},
    {black,mark=square},
    {black,mark=triangle},
    {black,mark=diamond},
    {black,mark=pentagon},
    {blue,mark=o},
    {blue,mark=square},
    {blue,mark=triangle},
    {blue,mark=diamond},
    {blue,mark=pentagon},
            """,

            "width": "\\axisdefaultwidth",
            "height": "\\axisdefaultheight",
            "title": f"{metric_name.replace('_', '-')}"
        }))

    # print(mdiagram.to_str(alpha = .05, adjustment = "holm",))

    # mdiagram.to_file(
    #     f"{metric_name}.png",
    #     alpha = .05,
    #     adjustment = "holm",

    #     preamble = "\n".join([
    #         "\\definecolor{color1}{HTML}{84B818}",
    #     ]),
    #     axis_options = {
    #         "cycle list": ",".join([
    #             "{color1,mark=*}",
    #         ]),
    #         "width": "\\axisdefaultwidth",
    #         "height": "\\axisdefaultheight",
    #         "title": f"{metric_name}"
    #     },
    # )
