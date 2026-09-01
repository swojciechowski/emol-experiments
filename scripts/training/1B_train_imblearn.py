import os
import numpy as np

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor

from config import *
from base_defaults import BASE_CLASSIFIERS
from imblearn_defaults import IMBL_UNDERSAMPLERS

DATASETS = DATASETS_FULL
CLASSIFIERS = BASE_CLASSIFIERS
METHODS = IMBL_UNDERSAMPLERS

from datasets import load_keel_dataset

UPDATE_STEP = 100 / (N_SPLITS * len(CLASSIFIERS) * len(METHODS))

prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
)


def experiment_iter(ds_name):
    prog_task = prog.add_task(ds_name)

    X, y = load_keel_dataset(ds_name)
    for s_idx, (train, test) in enumerate(DEFAULT_VALIDATION.split(X, y)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        results_dir = os.path.join(RESULTS_DIR, ds_name, str(s_idx))
        os.makedirs(results_dir, exist_ok=True)

        for rsmpl_name, rsmpl_base in IMBL_RESAMPERS:
            rsmpl = clone(rsmpl_base)
            _X_train, _y_train = rsmpl.fit_resample(X_train, y_train)

            for clf_name, base_clf in CLASSIFIERS:
                results_file = os.path.join(results_dir, f"{rsmpl_name}_{clf_name}.npy")

                clf = clone(base_clf)
                clf.fit(_X_train, _y_train)
                y_pred = clf.predict(X_test)

                # Save results
                np.save(results_file, y_pred)

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
