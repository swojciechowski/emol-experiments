import os
import numpy as np

from rich.progress import Progress, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor

from config import *
from datasets import load_keel_dataset
from base_defaults import BASE_CLASSIFIERS

from emol.meus import MEUS
from emol.fitting import HoldOutFitting


metrics = [
    ["TPR", "TNR"],
    ["TPR", "PPV"],
    ["TPR", "TNR", "PPV"],
    ["balanced_accuracy"],
]

def get_methods(estimator):
    for m in metrics:
        yield (
            ("MEUS", '+'.join(m), 'constrained'),
            MEUS(
                estimator=estimator,
                fitting=HoldOutFitting(random_state=RANDOM_STATE),
                metrics=m,
                random_state=RANDOM_STATE,
                constrained=True,
            ),
        )

N_CONFIGS = 4
UPDATE_STEP = 100 / (N_SPLITS * len(BASE_CLASSIFIERS) * N_CONFIGS)

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

        for clf_name, base_clf in BASE_CLASSIFIERS:
            for config, emol in get_methods(base_clf):
                config_s = '-'.join(config)
                emol.fit(X_train, y_train)

                pref = np.ones(len(emol.metrics)) / len(emol.metrics)
                y_pred = emol.predict(X_test, pref=pref)
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
