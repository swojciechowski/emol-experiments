import os
import numpy as np
import sys
from itertools import combinations

from tabulate import tabulate

from emol.metrics import BinaryConfusionMatrix
from emol.helpers.latex import latex_table
from emol.misc.ftest import F_test

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

latex_symbols = [
    "\\textcolor{c3}{\\ding{170}}",
    "\\textcolor{c1}{\\ding{169}}",
    "\\textcolor{c2}{\\ding{171}}",
    "\\textcolor{c0}{\\ding{168}}",
]

colors = np.array(["c3", "c1", "c2", "c0"])

def to_mcul(tab):
    return f"\\mcul{{{','.join(colors[tab])}}}{{"

def stat_test_mat(series, alpha=0.05):
    s_mat = np.zeros((len(series), len(series)), dtype=bool)
    p_mat = np.zeros((len(series), len(series)), dtype=bool)

    # Check each pair
    for s1, s2 in combinations(range(len(series)), 2):
        stat, p = F_test(series[s1], series[s2], cv=(5, 2))

        p_mat[(s1, s2), (s2, s1)] = p < alpha
        s_mat[s1, s2] = np.mean(series[s1]) > np.mean(series[s2])
        s_mat[s2, s1] = np.mean(series[s1]) < np.mean(series[s2])

    return s_mat, p_mat

def stat_results_to_list(stat_tesults):
    return [np.argwhere(row).flatten() for row in stat_tesults]

def stat_lists_to_str(lists):
    n_all = len(lists) - 1

    def translate_row(a):
        if len(a) == n_all:
            return '\\textbf{all}'
        if len(a) == 0:
            return ''
        else:
            symbols = list(map(lambda x: f"{latex_symbols[x]}", a))
            return ' '.join(symbols)

    return list(map(translate_row, lists))

BASE_METRICS = ['TPR', 'TNR', 'PPV', 'NPV']
METRICS = ["balanced_accuracy"]
# METRICS = ["balanced_accuracy", "gmean", "f1_score"]
DATASETS = DATASETS_FULL

METHODS = MEUS_CONFIG + IMBL_UNDERSAMPLERS
# METHODS = MOWS_CONFIG + SMOTEV_TOP

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

for metric_name, metric_results in zip(METRICS, results_cube):
    for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
        table = []
        for ds_i, (ds_name, dataset_results) in enumerate(zip(DATASETS, metric_results)):
            results = dataset_results[clf_idx]

            mv = results.mean(axis=-1)
            mv = np.char.mod('%.2f', mv)

            sd = results.std(axis=-1)
            sd = np.char.mod('%.2f', sd)

            s, p = stat_test_mat(results)

            combined_mat = np.logical_and(s, p)

            ss = stat_results_to_list(combined_mat[:, 0:4])
            is_worse = combined_mat[0:4, :].T
            ss = stat_lists_to_str(ss)

            # table_row = [f"{v} $\\pm$ {d} \n {s}" for v, d, s in zip(mv, sd, ss)]
            table_row = [f"{to_mcul(w) if np.any(w) and i > 3 else ''}{v} $\\pm$ {d[1:]}{'}' if np.any(w) and i > 3 else ''} \n {s}" for i, (v, d, s, w) in enumerate(zip(mv, sd, ss, is_worse))]
            index = DATASETS_IDX[ds_name]
            table.append([f"\\textit{{{index}}}", *table_row])

        print(clf_name, metric_name)
        print(tabulate(table))

        out_dir = os.path.join(TABLES_DIR, f'full')
        os.makedirs(out_dir, exist_ok=True)
        out_file_dir = os.path.join(out_dir, f"MEUS-{metric_name}-{clf_name}.tex")

        with open(out_file_dir, 'w') as fp:
            fp.write(latex_table(table, headers=[f"\\textsc{{{METHODS_SHORT_NAMES[cn]}}}{' ' + latex_symbols[i] if i < 4 else ''}" for i, cn in enumerate(METHODS)], width=1000))
