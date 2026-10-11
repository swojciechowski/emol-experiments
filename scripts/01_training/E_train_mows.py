import os
import sys
import numpy as np

from rich.progress import Progress, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from helpers import load_keel_dataset
from base_defaults import DEF_BASE_CLASSIFIERS

from emol.mows import MOWS
from emol.fitting import LeakingFitting, SyntheticFitting


metrics = [
    ["TPR", "TNR"],
]

fitting = [
    LeakingFitting,
    SyntheticFitting
]

constrained = [
    True, False
]

def get_methods(estimator):
    for f in fitting:
        for c in constrained:
            yield (
                ("MOWS", f.__name__, '+'.join(["TPR", "TNR"]), 'constrained' if c else 'unconstrained'),
                MOWS(
                    estimator=estimator,
                    fitting=f(random_state=RANDOM_STATE),
                    metrics=["TPR", "TNR"],
                    random_state=RANDOM_STATE,
                    constrained=c,
                ),
            )

N_CONFIGS = 4
UPDATE_STEP = 100 / (N_SPLITS * len(DEF_BASE_CLASSIFIERS) * N_CONFIGS)

prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
)

def experiment_iter(ds_name):
    prog_task = prog.add_task(ds_name)

    X, y = load_keel_dataset(ds_name, keel_path=os.path.join(PROJECT_ROOT, 'keel'))
    for s_idx, (train, test) in enumerate(DEFAULT_VALIDATION.split(X, y)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
        os.makedirs(results_dir, exist_ok=True)

        for clf_name, base_clf in DEF_BASE_CLASSIFIERS:
            for config, emol in get_methods(base_clf):
                config_s = '-'.join(config)
                emol.fit(X_train, y_train)

                y_pred = emol.predict(X_test, pref=[0.5, 0.5])
                # Save predictions
                np.save(os.path.join(results_dir, f"{config_s}_{clf_name}.npy"), y_pred)

                # Save all predictions and indicators
                y_pred = emol.predict(X_test)

                np.save(os.path.join(results_dir, f"{config_s}_{clf_name}_prd.npy"), y_pred)
                np.save(os.path.join(results_dir, f"{config_s}_{clf_name}_ind.npy"), emol._indicators)

                prog.update(prog_task, advance=UPDATE_STEP)


def done_handler(task):
    result = task.result()
    if result:
        prog.console.print(result)


if __name__ == "__main__":
    prog.start()

    with ThreadPoolExecutor(max_workers=N_JOBS) as e:
        for ds_name in DATASETS:
            task = e.submit(experiment_iter, ds_name)
            task.add_done_callback(done_handler)

    prog.stop()
