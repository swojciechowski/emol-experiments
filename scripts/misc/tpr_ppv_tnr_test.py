import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from tqdm import tqdm

from emol.fbeta.fbeta import fbeta_analysis, fbeta_analysis_aprx, fbeta
from emol.misc.generator.random_clusters import random_clusters
from emol.metrics.binary_confusion_matrix import confusion_matrix

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def calc_tnr(tpr, ppv, ir):
    return 1 - (1 / ir) * tpr * ((1 - ppv) / ppv)

def f_tpr(cm):
    return cm[1, 1] / np.sum(cm[1, :])


def f_tnr(cm):
    return cm[0, 0] / np.sum(cm[0, :])


def f_ppv(cm):
    return cm[1, 1] / np.sum(cm[:, 1])


# N_SAMPLES = [50, 100, 200, 500, 1000, 2000]
N_SAMPLES = [10, 20]
RATIOS = [0.1, 0.5, 0.8, 0.9]

for n_samples in N_SAMPLES:
    for min_r in RATIOS:
        n_min = int(n_samples * min_r)
        n_maj = n_samples - n_min

        ir = n_maj / n_min
        print("IR", ir)
        y_true = np.array([0] * n_maj + [1] * n_min)

        results = []

        for _ in tqdm(product([0, 1], repeat=n_samples), total=2^n_samples):
            cm = confusion_matrix(y_true, _)
            tpr, ppv, tnr = f_tpr(cm), f_ppv(cm), f_tnr(cm)

            if ppv == 0 or not np.isfinite(ppv):
                continue

            results.append([tpr, ppv, tnr])

        results = np.array(results)
        res_tnr = calc_tnr(results[:, 0], results[:, 1], ir)

        close = np.allclose(results[:, 2], res_tnr)

        print(close)
