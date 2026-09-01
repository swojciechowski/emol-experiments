import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold

from sklearn.base import clone

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from emol.metrics import BinaryConfusionMatrix
from emol.helpers.plot.animation import Animation

from emol.meus import MEUS
from emol.mows import MOWS

import os

from emol.fitting import (
    HoldOutFitting,
    CrossValidationFitting,
    DummyFitting,
    BootstrapFitting,
)

RADOM_STATE = 26600

fittings = [
    HoldOutFitting(random_state=RADOM_STATE),
    # CrossValidationFitting(random_state=RADOM_STATE),
    # DummyFitting(random_state=RADOM_STATE),
    BootstrapFitting(random_state=RADOM_STATE),
]

metrics = [
    ["TPR", "TNR"],
    # ["PPV", "NPV"],
    # ["balanced_accuracy"],
]

emols = [MOWS]
constrained = [True, False]
proba = [True, False]

def get_methods(estimator):
    for emol in emols:
        for fitting in fittings:
            for p in proba:
                for metric in metrics:
                    for c in constrained:
                        yield (
                            (emol.__name__, type(fitting).__name__, type(estimator).__name__, '+'.join(metric), 'c' if c else 'uc', 'pp' if p else 'p'),
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
    n_samples=300,
    flip_y=0,
    weights=[0.8, 0.2],
    class_sep=0.8,
    random_state=RADOM_STATE,
)

# est_name, estimator = "GNB", GaussianNB()
est_name, estimator = "CART", DecisionTreeClassifier()
# Train Baseline on CV -- those are expected metrics with deviation
scores = []

for train, test in RepeatedStratifiedKFold(random_state=RADOM_STATE).split(X, y):
    clf = clone(estimator)
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(BinaryConfusionMatrix(y[test], y_pred).get_metrics(*["TPR", "TNR", "balanced_accuracy"]))

baseline_m = np.mean(scores)
baseline_d = np.std(scores)


# Prepare single trail
train, test = next(
    StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RADOM_STATE).split(
        X, y
    )
)

X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

configs = [*get_methods(estimator)]


for config, emol in configs:
    os.makedirs(f"{'_'.join(config)}", exist_ok=True)

    a = Animation(tmp_path=f"{'_'.join(config)}")

    def step_eval_cb(model, solver):
        fig, ax = plt.subplots(1, 1)

        solutions = solver.pop.get("X")
        indicators = -1 * solver.pop.get("F")
        models = solver.pop.get("A")

        models = models.ravel() if models.shape[1] == 1 else [model.train_estimator(s) for s in solutions]

        test_scores = []

        for m in models:
            # Evaluate predictions
            y_pred = m.predict_proba(X_test) if emol.use_proba else m.predict(X_test)
            test_scores.append(BinaryConfusionMatrix(y_test, y_pred).get_metrics(*emol.metrics))

        test_scores = np.array(test_scores)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.scatter(*indicators.T, c='k')
        ax.scatter(*test_scores.T, c='r')
        ax.grid(ls=":")

        a.add_frame(fig)

        plt.close()
        plt.clf()

    emol.fit(
        X_train,
        y_train,
        step_cb=step_eval_cb,
    )

    a.export(f"{'_'.join(config)}.gif")

#     indicators = np.mean(np.array(history_indicators), axis=-1)
#     test = np.mean(np.array(history_indicators), axis=-1)

#     ax = next(axs)

#     ax.set_ylim(0.5, 1)

#     best = np.argmax(indicators, axis=1)

#     for i in range(len(history_test)):
#         ax.scatter(np.repeat(i, indicators.shape[1]), indicators[i], c='b', s=1, edgecolors='none', alpha=0.15)
#         ax.scatter(np.repeat(i, test.shape[1]), test[i], c='r', s=1, edgecolors='none', alpha=0.15)

#     ax.plot(indicators[np.arange(len(test)), best], c='w', lw=2)
#     ax.plot(indicators[np.arange(len(test)), best], c='b', lw=1, label="train")
#     ax.plot(test[np.arange(len(test)), best], c='w', lw=2)
#     ax.plot(test[np.arange(len(test)), best], c='r', lw=1, label="test")

#     ax.hlines(
#         baseline_m, ls="-", color="k", xmin=0, xmax=len(test) - 1, lw=1
#     )
#     ax.hlines(
#         baseline_m + baseline_d,
#         ls="--",
#         color="k",
#         xmin=0,
#         xmax=len(test) - 1,
#         lw=1,
#     )
#     ax.hlines(
#         baseline_m - baseline_d,
#         ls="--",
#         color="k",
#         xmin=0,
#         xmax=len(test) - 1,
#         lw=1,
#     )

#     ax.grid(ls=":")
#     ax.legend()
#     ax.set_title('-'.join(config))

# plt.tight_layout()
# plt.savefig(f"{est_name}_fitting.png")

# plt.close()
# plt.clf()
