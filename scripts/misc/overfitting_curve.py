import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold

from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from emol.metrics import BinaryConfusionMatrix

from functools import partial

from emol.meus import MEUS
from emol.mows import MOWS

from config import *

from emol.fitting import (
    HoldOutFitting,
    CrossValidationFitting,
    DummyFitting,
    BootstrapFitting,
)

from emol.moo.evaluation.metrics import *

RADOM_STATE = 26600

fittings = [
    # HoldOutFitting(random_state=RADOM_STATE),
    # CrossValidationFitting(random_state=RADOM_STATE),
    DummyFitting(random_state=RADOM_STATE),
    BootstrapFitting(random_state=RADOM_STATE),
]

metrics = [
    ["TPR", "TNR"],
    # ["PPV", "NPV"],
]

snap_frames = np.array([
    1, 2, 5,
    10, 20, 50,
    # 100, 200, 500,
])

# emols = [MOWS, MEUS]
emols = [MOWS]
constrained = [True]
# constrained = [False]
# proba = [True, False]
proba = [False]

def get_methods(estimator):
    for emol in emols:
        for fitting in fittings:
            for p in proba:
                for metric in metrics:
                    for c in constrained:
                        yield (
                            (emol.__name__, type(fitting).__name__, 'cc' if c else 'uc', 'pp' if p else 'pc'),
                            emol(
                                estimator=estimator,
                                fitting=fitting,
                                metrics=metric,
                                random_state=RADOM_STATE,
                                constrained=c,
                                use_proba=p,
                                verbose=True
                            ),
                        )


X, y = make_classification(
    n_samples=200,
    flip_y=0,
    weights=[0.8, 0.2],
    class_sep=0.8,
    random_state=RADOM_STATE,
)

est_name, estimator = "GNB", GaussianNB()
# est_name, estimator = "CRT", DecisionTreeClassifier()

# Train Baseline on CV -- those are expected metrics with deviation
scores = []

for train, test in RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=RADOM_STATE).split(X, y):
    clf = clone(estimator)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(BinaryConfusionMatrix(y[test], y_pred).get_metrics(*["TPR", "TNR", "balanced_accuracy"]))

baseline_m = np.mean(scores)
baseline_d = np.std(scores)

# Prepare single trail
train, test = next(
    StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RADOM_STATE).split(
        X, y
    )
)

NAMES = {
    "LeakingFitting": "Leaking",
    "BootstrapFitting": "Synthetic",
}

X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

configs = [*get_methods(estimator)]

for config, emol in configs:
    # Gather results
    history_indicators = []
    history_test = []

    def step_eval_cb(model, solver):
        solutions = solver.pop.get("X")
        history_indicators.append(-1 * solver.pop.get("F"))

        models = solver.pop.get("A")
        models = models.ravel() if models.shape[1] == 1 else [model.train_estimator(s) for s in solutions]

        test_scores = []

        for m in models:
            # Evaluate predictions
            # y_pred = m.predict_proba(X_test) if emol.use_proba else m.predict(X_test)
            y_pred = m.predict(X_test)
            test_scores.append(BinaryConfusionMatrix(y_test, y_pred).get_metrics(*emol.metrics))

        history_test.append(test_scores)


    emol.fit(
        X_train,
        y_train,
        step_cb=step_eval_cb,
    )

    history_indicators = np.array(history_indicators)
    history_test = np.array(history_test)

    fig = plt.figure()
    fig.set_size_inches(FIG_A4_WIDTH, FIG_A4_HEIGHT * (5/6))
    gs = GridSpec(3, 3, figure=fig, height_ratios=(3, 3, 2))

    fig.suptitle(f"{est_name} {NAMES[config[1]]}", y=0.92, fontsize=16)

    # plot_snaps
    snap_axs = []
    share_ax = None

    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], sharex=share_ax)
            snap_axs.append(ax)

            if j == 0:
                share_ax = ax

    snap_axs = iter(snap_axs)

    for frame in snap_frames - 1:

        ax = next(snap_axs)

        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
        ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
        ax.set_xticks([0, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.5, 0.75, 1.0])
        ax.set_aspect("equal")
        ax.grid(ls=":")
        ax.set_xlabel("TNR")
        ax.set_ylabel("TPR")
        ax.set_title(f"iter: {frame + 1}")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        hindi = history_indicators[frame]
        htest = history_test[frame]

        nds = non_dominated_solutions(hindi)

        ax.scatter(*hindi.T, edgecolor='C0', facecolor='none', s=15, lw=0.5)
        ax.scatter(*htest.T, edgecolor='C3', facecolor='none', s=15, lw=0.5)

        ax.scatter(*hindi[nds].T, facecolor='C0', edgecolor='none', s=15, lw=0.5)
        ax.scatter(*htest[nds].T, facecolor='C3', edgecolor='none', s=15, lw=0.5)

    # Plot curve

    # hv_train_history = [HV(x, np.array([0, 0])) for x in history_indicators]
    # hv_test_history = [HV(x, np.array([0, 0])) for x in history_test]

    # ax = fig.add_subplot(gs[4, :])

    # ax.set_title("MOO Indicators")

    # ax.grid(ls=":")
    # ax.set_xlim(0, len(history_indicators) + 1)
    # # ax.set_ylim(0, 1.5)
    # ax.spines[["right", "top"]].set_visible(False)
    # ax.set_xticks([0, 20, 50, 100, 200, 300, 400, 500])

    # hv_train_history = [HV(x, np.array([0, 0])) for x in history_indicators]
    # hv_test_history = [HV(x, np.array([0, 0])) for x in history_test]

    # ax.plot(np.arange(1, len(history_test) + 1), hv_train_history, c='orange', lw=1)
    # ax.plot(np.arange(1, len(history_test) + 1), hv_test_history, c='orange', ls='--', lw=1)
    # ax.set_ylabel("HV")
    # ax.set_ylim(0, 1.01)

    # # hv_train_history = [GD(x, np.array([1, 1])) for x in history_indicators]
    # # hv_test_history = [GD(x, np.array([1, 1])) for x in history_test]

    # # ax.plot(np.arange(len(history_test)), hv_train_history, c='orange', lw=1, label="train GD")
    # # ax.plot(np.arange(len(history_test)), hv_test_history, c='orange', ls='--', lw=1, label="test GD")

    # hv_train_history = [ONVGR(x) for x in history_indicators]
    # hv_test_history = [ONVGR(x) for x in history_test]

    # axl = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    # axl.plot(np.arange(1, len(history_test) + 1), hv_train_history, c='green', lw=1)
    # axl.plot(np.arange(1, len(history_test) + 1), hv_test_history, c='green', ls='--', lw=1)
    # axl.set_ylabel("ONVGR")


    # # hv_train_history = [SP(x) for x in history_indicators]
    # # hv_test_history = [SP(x) for x in history_test]

    # # axl.plot(np.arange(len(history_test)), hv_train_history, c='m', lw=1, label="train SP")
    # # axl.plot(np.arange(len(history_test)), hv_test_history, c='m', ls='--', lw=1, label="test SP")

    # ax.spines[["top"]].set_visible(False)

    ax = fig.add_subplot(gs[2, :])

    # ax.set_title("BAC Curve")
    ax.set_ylim(0.7, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, len(history_indicators))

    # bac
    indicators = np.mean(history_indicators, axis=-1)
    test = np.mean(history_test, axis=-1)
    best = np.argmax(indicators, axis=1)

    for i in range(len(history_test)):
        ax.scatter(np.repeat(i, indicators.shape[1]), indicators[i], c='C0', s=5, edgecolors='none', alpha=0.1)
        ax.scatter(np.repeat(i, test.shape[1]), test[i], c='C3', s=5, edgecolors='none', alpha=0.1)

    # ax.hlines(
    #     baseline_m, ls="-", color="w", xmin=0, xmax=len(test) - 1, lw=3,
    # )

    # ax.hlines(
    #     baseline_m, ls="-", color="k", xmin=0, xmax=len(test) - 1, lw=1, label="BAC (CV)"
    # )

    # ax.hlines(
    #     baseline_m + baseline_d,
    #     ls="--",
    #     color="k",
    #     xmin=0,
    #     xmax=len(test) - 1,
    #     lw=1,
    # )

    # ax.hlines(
    #     baseline_m - baseline_d,
    #     ls="--",
    #     color="k",
    #     xmin=0,
    #     xmax=len(test) - 1,
    #     lw=1,
    # )

    ax.plot(range(1, 51), indicators[np.arange(len(test)), best], c='w', lw=3)
    ax.plot(range(1, 51), indicators[np.arange(len(test)), best], c='C0', lw=1, label="train")
    ax.plot(range(1, 51), test[np.arange(len(test)), best], c='w', lw=3)
    ax.plot(range(1, 51), test[np.arange(len(test)), best], c='C3', lw=1, label="test")

    ax.set_xlabel("iter")
    ax.set_ylabel("BAC")
    # ax.set_xticks([0, 20, 50, 100, 200, 300, 400, 500])
    ax.set_xticks([1, 5, 10, 20, 50])


    ax.grid(ls=":")
    ax.legend(loc="lower right")


    out_dir = os.path.join(FIGURES_DIR, f'overfitting_curve')
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, f"{est_name}_{'-'.join(config)}_fitting.pdf")

    plt.savefig("foo.png", bbox_inches='tight')
    plt.savefig(out_file_path, bbox_inches='tight')

    plt.close()
    plt.clf()
