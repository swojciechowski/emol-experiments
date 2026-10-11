import os
import sys

import numpy as np

from sklearn.base import clone
from rich.progress import Progress, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from base_defaults import DEF_BASE_CLASSIFIERS
from helpers import load_keel_dataset


UPDATE_STEP = 100 / (N_SPLITS * len(DEF_BASE_CLASSIFIERS))

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

        # Save true labels
        np.save(os.path.join(results_dir, f"y_true.npy"), y_test)

        for clf_name, base_clf in DEF_BASE_CLASSIFIERS:
            clf = clone(base_clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Save results
            np.save(os.path.join(results_dir, f"BASE_{clf_name}.npy"), y_pred)

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
