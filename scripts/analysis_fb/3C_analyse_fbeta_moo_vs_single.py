import os
import numpy as np

from emol.metrics import BinaryConfusionMatrix

from config import *
from datasets import load_keel_dataset
from emol.moo.evaluation.metrics import non_dominated_solutions

from functools import partial

import matplotlib.pyplot as plt
from emol.fbeta.plot import fbeta_plot
from emol.fbeta.fbeta import fbeta_rank

METRICS = ['TPR', "TNR", "PPV"]
DATASETS = DATASETS_SHORT
METHODS = MOL_METHODS[0:]
CLASSIFIERS = ["GNB"]

for ds_idx, ds_name in enumerate(DATASETS):
    X, y = load_keel_dataset(ds_name)

    for clf_idx, clf_name in enumerate(CLASSIFIERS):
        for s_idx, (train, test) in enumerate(DEFAULT_VALIDATION.split(X, y)):
            X_test, y_test = X[test], y[test]

            for mtd_idx, mtd_name in enumerate(METHODS):
                print(f"{ds_name}-{clf_name}-{mtd_name.split('-')[0]}")

                results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
                record_name = f"{mtd_name}_{clf_name}"
                results_file = os.path.join(results_dir, f'{record_name}_prd.npy')
                indicators_file = os.path.join(results_dir, f'{record_name}_ind.npy')

                y_pred = np.load(results_file)
                indicators = -1 * np.load(indicators_file)

                metrics = BinaryConfusionMatrix(y_test, y_pred).get_metrics(*METRICS).T
                labels = [
                    f"[{'|'.join(map(lambda x: f"{x:.3f}", ind))}]"
                    for i, ind in enumerate(metrics[:, [0, 2]])
                ]

                print(ds_name, clf_name, mtd_name, s_idx)
                # print(metrics[:, [0, 2]])
                # exit()

                fbeta_best = np.array(fbeta_rank(metrics[:, [0, 2]], precision=0.0001))[:, 1].astype(int)
                ind_nd_map = non_dominated_solutions(indicators)
                pnr_nd_map = non_dominated_solutions(metrics[:, [0, 1]])
                fbt_nd_map = non_dominated_solutions(metrics[:, [0, 2]])

                fig, axs = plt.subplots(1, 3, figsize=(4 + FIG_FBETA_WIDTH, FIG_FBETA_HEIGHT))

                ax = axs[0]

                for artist in ax.get_children():
                    if hasattr(artist, 'set_clip_on'):
                        artist.set_clip_on(False)

                plot_metrics = metrics[:, [0, 1]]

                ax.set_aspect("equal")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_xlabel("TPR")
                ax.set_ylabel("TNR")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(ls=":")

                ax.scatter(*indicators.T, c='grey', s=30, clip_on=False)

                for bi, best in enumerate(fbeta_best):
                    ax.scatter(*indicators[best], edgecolor='none', facecolor=f'C{bi}', s=15)

                # ax.scatter(*plot_metrics.T, c='grey', s=10)
                ax.scatter(*plot_metrics[pnr_nd_map].T, edgecolor='k', facecolor='none', s=30)

                for bi, best in enumerate(fbeta_best):
                    ax.scatter(*plot_metrics[best], edgecolor='none', facecolor=f'C{bi}', s=25)

                # ax.scatter(*plot_metrics[fbeta_best].T, edgecolor='none', facecolor='g', s=55)
                # ax.scatter(*plot_metrics[pnr_nd_map].T, edgecolor='r', facecolor='none', s=15)

                ax = axs[1]

                for artist in ax.get_children():
                    if hasattr(artist, 'set_clip_on'):
                        artist.set_clip_on(False)

                plot_metrics = metrics[:, [0, 2]]

                ax.set_aspect("equal")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_xlabel("TPR")
                ax.set_ylabel("PPV")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(ls=":")

                ax.scatter(*plot_metrics.T, c='grey', s=10)

                ax.scatter(*plot_metrics[fbt_nd_map].T, edgecolor='k', facecolor='none', s=30)
                # ax.scatter(*plot_metrics[fbeta_best].T, edgecolor='none', facecolor='g', s=55)

                for bi, best in enumerate(fbeta_best):
                    ax.scatter(*plot_metrics[best], edgecolor='none', facecolor=f'C{bi}', s=25)
                # ax.scatter(*plot_metrics[pnr_nd_map].T, edgecolor='r', facecolor='none', s=15)

                ax = axs[2]

                fbeta_plot(metrics[:, [0, 2]], ax=ax, labels=labels, N=200, add_legend=False)

                # ax.set_title(DATASETS_IDX[ds_name], y=1.20, fontsize=FBETA_TITLE_SIZE)

                out_dir = os.path.join(FIGURES_DIR, f'fbeta_moo')
                os.makedirs(out_dir, exist_ok=True)
                # out_file_dir = os.path.join(out_dir, f"{ds_name}-{clf_name}-{mtd_name.split('-')[0]}.pdf")
                out_file_dir = os.path.join(out_dir, f"{ds_name}.pdf")


                plt.savefig(out_file_dir, bbox_inches='tight')
                plt.savefig("foo.png", bbox_inches='tight')

                plt.clf()
                plt.close()

            break
