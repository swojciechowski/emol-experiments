import numpy as np
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from operator import itemgetter

import problexity as px
import problexity.classification as c

from emol.helpers.keel import parse_keel_dat

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from experiments.config import *

from tqdm import tqdm

DS_NAMES = [ds_fname.split(".")[0] for ds_fname in os.listdir("keel")]
DS_NAMES = list(sorted(DS_NAMES))

DATASETS = {
    relation: {"data": data, "target": target}
    for relation, data, target in [
        parse_keel_dat(os.path.join("keel", f"{ds_name}.dat")) for ds_name in DS_NAMES
    ]
}

table = []

for ds in tqdm(DATASETS):
    X, y = DATASETS[ds]["data"], DATASETS[ds]["target"]
    y = LabelEncoder().fit_transform(y)
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()

    n_samples = len(X)
    n_features = len(X.T)
    n_categorical = len(categorical_features)
    c_lab, c_num = np.unique(y, return_counts=True)
    min_class_samples = np.min(c_num)
    ir = np.max(c_num) / min_class_samples

    # ir >= 2
    if ir < 2:
        continue

    # No categorical features
    if n_categorical > 0:
        continue

    # At least 50 minority class samples
    if min_class_samples < 25:
        continue

    cc = px.ComplexityCalculator(
        metrics=[
            c.f1,
            c.n2,
            c.lsc,
            c.t4,
            c.c2
        ]
    ).fit(X, y)

    table.append([ds, n_samples, n_features, ir, min_class_samples])

headers = [
    "datasets",
    "n_samples",
    "n_features",
    "ir",
    "min_class_samples",
]

table = sorted(table, key=itemgetter(0), reverse=True)
datasets = list(map(itemgetter(0), table))

print(datasets)

with open("datasets.tbl", "w") as fp:
    fp.write(f"Total: {len(table)}\n")
    fp.write(tabulate(table, tablefmt="grid", headers=headers))

# Sphinx / PC shell
