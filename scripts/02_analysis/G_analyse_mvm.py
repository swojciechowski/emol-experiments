import os
import sys
import numpy as np

from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tabulate import tabulate

from emol.metrics import BinaryConfusionMatrix
from emol.helpers.latex import latex_table
from emol.helpers.ascii import ascii_underline, ansi_to_latex, ascii_bold
from emol.misc.ftest import F_test

from emol.moo.evaluation.metrics import HV, GD

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

def stat_test_mat(series, alpha=0.05):
    s_mat = np.zeros((len(series), len(series)), dtype=bool)
    p_mat = np.zeros((len(series), len(series)), dtype=bool)

    # Check each pair
    for s1, s2 in combinations(range(len(series)), 2):
        stat, p = F_test(series[s1], series[s2], cv=(5, 2))

        p_mat[(s1, s2), (s2, s1)] = p < alpha
        s_mat[s1, s2] = np.mean(series[s1]) > np.mean(series[s2])
        s_mat[s2, s1] = np.mean(series[s1]) < np.mean(series[s2])
        # s_mat[s1, s2] = np.mean(series[s1]) < np.mean(series[s2])
        # s_mat[s2, s1] = np.mean(series[s1]) > np.mean(series[s2])

    return s_mat, p_mat

def stat_results_to_list(stat_tesults):
    return [np.argwhere(row).flatten() for row in stat_tesults]

# MOO_METRIC = "GD"
MOO_METRIC = "HV"
#
DATASETS = DATASETS_FULL
# DATASETS = DATASETS_SHORT

# METHODS = MOWS_CONFIG + MEUS_CONFIG
# METHODS = MOWS_CONFIG
METHODS = MOL_METHODS
METHODS_CMP_IND = []

# METRICS = ["TPR", "PPV"]
METRICS = ["TPR", "TNR"]

results_cube = np.zeros((len(DATASETS), len(BASE_CLASSIFIERS), N_SPLITS, len(METHODS)))

for ds_idx, ds_name in enumerate(DATASETS):
    for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
        for s_idx in range(N_SPLITS):

            y_test = np.load(os.path.join(RESULTS_DIR, ds_name, str(s_idx), "y_true.npy"))

            fronts = []

            for mtd_idx, mtd_name in enumerate(METHODS):
                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"

                results_file = os.path.join(results_dir, f'{record_name}_prd.npy')
                y_pred = np.load(results_file)

                results = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T
                fronts.append(results)

                results_cube[ds_idx, clf_idx, s_idx, mtd_idx] = HV(results, np.array([0., 0.]))
                # results_cube[ds_idx, clf_idx, s_idx, mtd_idx] = GD(results, np.array([1., 1.]))

# Tables
table = []

wdl_table = np.zeros((len(BASE_CLASSIFIERS), 3))

for ds_i, (ds_name, dataset_results) in enumerate(zip(DATASETS, results_cube)):
    index = DATASETS_IDX[ds_name]
    row = [index]

    for clf_idx, clf_name in enumerate(BASE_CLASSIFIERS):
        results = dataset_results[clf_idx].T

        mv = results.mean(axis=-1)
        mv = np.char.mod('%.2f', mv)

        if mv[0] > mv[1]:
            wdl_table[clf_idx, 0] += 1
        elif mv[0] < mv[1]:
            wdl_table[clf_idx, 2] += 1
        else:
            wdl_table[clf_idx, 1] += 1

        sd = results.std(axis=-1)
        sd = np.char.mod('%.2f', sd)

        s, p = stat_test_mat(results)

        combined_mat = np.logical_and(s, p)

        ss = stat_results_to_list(combined_mat)
        ss = stat_lists_to_str(ss)

        # table_row = [f"{v} $\\pm$ {d} \n {s}" for v, d, s in zip(mv, sd, ss)]
        table_row = [ascii_bold(ascii_underline(f"{v} ± {d[1:]}")) if s else f"{v} ± {d[1:]}"  for v, d, s in zip(mv, sd, ss)]
        row.extend(table_row)

    table.append(row)


fig, ax = plt.subplots(figsize=(FIG_A4_WIDTH / 2, FIG_A4_HEIGHT / 4))

y = np.arange(len(BASE_CLASSIFIERS))
wdl_table = wdl_table.T

fig.suptitle(MOO_METRIC)

ax.barh(y, wdl_table[0], color="C0", edgecolor="k", alpha=0.8, label="MEUS win")
ax.barh(y, wdl_table[1], left=wdl_table[0], color="gray", alpha=0.8, edgecolor="k", label="draw")
ax.barh(y, wdl_table[2], left=wdl_table[0] + wdl_table[1], color="C3", alpha=0.8, edgecolor="k", label="MOWS win")

ax.set_yticks(y)
ax.set_yticklabels(BASE_CLASSIFIERS, fontsize=9)
ax.set_xlim(0, len(DATASETS))
ax.invert_yaxis()

ax.legend(
    ncol=3,
)

legend = ax.get_legend()
handles, labels = ax.get_legend_handles_labels()

fig_leg = plt.figure(figsize=(FIG_A4_WIDTH, FIG_A4_HEIGHT / 30))
leg = fig_leg.legend(handles, labels, loc='center', ncol=3, frameon=False)
frame = leg.get_frame()
frame.set_boxstyle("square")
plt.axis('off')

out_dir = os.path.join(FIGURES_DIR, f'mvm')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "legend.pdf")

fig_leg.savefig(out_file, bbox_inches='tight', transparent=True, pad_inches=0)
plt.close(fig_leg)

legend.remove()

plt.savefig("foo.png", bbox_inches="tight")

out_file = os.path.join(out_dir, f"{MOO_METRIC}.pdf")
plt.savefig(out_file, bbox_inches="tight")
plt.close()

out_dir = os.path.join(TABLES_DIR, f'moo')
os.makedirs(out_dir, exist_ok=True)
out_file_dir = os.path.join(out_dir, f"{MOO_METRIC}.tex")

with open(out_file_dir, 'w') as fp:
    fp.write(latex_table([[f"\\textit{{{t[0]}}}", *[ansi_to_latex(v) for v in t[1:]]] for t in table], headers=[f"{METHODS_SHORT_NAMES[cn]}" for i, cn in enumerate(METHODS)], width=1000))
