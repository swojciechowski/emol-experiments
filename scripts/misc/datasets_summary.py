import numpy as np

import os

from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from collections import defaultdict

import problexity as px
import problexity.classification as c

from emol.helpers.keel import parse_keel_dat
from emol.helpers.latex import latex_table

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import ClassicalMDS
from sklearn.preprocessing import StandardScaler

import sys

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from tqdm import tqdm

import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def build_xxnn_color_map(category_counts, start_hue=0.0, as_hex=True):
    """Generates a dict mapping 'XXNN' element keys to distinct sequential colors.

    :param category_counts: Dict of {category_prefix: count}
    :param start_hue: Hue offset. Accepts float [0.0, 1.0] or degrees [0, 360]
                       (e.g., 0.0 = Red, 0.33/120° = Green, 0.66/240° = Blue).
    :param as_hex: Return Hex string if True, RGB float tuple if False
    :return: Dict of {'XXNN': color}
    """
    element_color_map = {}
    num_categories = len(category_counts)

    # Convert degrees (> 1.0) to float scale [0.0, 1.0]
    hue_offset = (
        (start_hue % 360.0) / 360.0 if start_hue > 1.0 else (start_hue % 1.0)
    )

    for i, (cat, count) in enumerate(category_counts.items()):
        # Shift hue by start offset and wrap around the color wheel
        hue = (hue_offset + (i / num_categories)) % 1.0

        if count == 1:
            saturations = [0.85]
            values = [0.90]
        else:
            saturations = np.linspace(0.30, 0.95, count)
            values = np.linspace(0.95, 0.50, count)

        for order in range(1, count + 1):
            s = saturations[order - 1]
            v = values[order - 1]

            r, g, b = colorsys.hsv_to_rgb(hue, s, v)
            key = f"{cat}{order:02d}"

            if as_hex:
                element_color_map[key] = (
                    f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                )
            else:
                element_color_map[key] = (
                    round(r, 4),
                    round(g, 4),
                    round(b, 4),
                )

    return element_color_map


# Category counts dictionary
data = {
    "ec": 6,
    "gl": 3,
    "ha": 1,
    "ir": 1,
    "le": 1,
    "ne": 1,
    "pa": 2,
    "po": 2,
    "se": 1,
    "sh": 2,
    "ve": 4,
    "vo": 1,
    "wi": 2,
    "ye": 13,
}


# Build complete dictionary
xxnn_colors = build_xxnn_color_map(data, start_hue=0.75, as_hex=True)

DATASETS = sorted(DATASETS_FULL)
CMPLX_RESULTS_FILE = 'datasets.npy'

if not os.path.isfile(CMPLX_RESULTS_FILE):
    table = []
    complexities = []
    dataset_indexes = []
    datasets_dict = {}
    ds_family_counters = defaultdict(lambda: 1)

    for ds in tqdm(DATASETS):
        _, X, y = parse_keel_dat(os.path.join(DEFAULT_KEEL_DIR, f"{ds}.dat"))
        y = LabelEncoder().fit_transform(y)
        categorical_features = X.select_dtypes(include=["category"]).columns.tolist()

        n_samples = len(X)
        n_features = len(X.T)
        n_categorical = len(categorical_features)
        c_lab, c_num = np.unique(y, return_counts=True)
        min_class_samples = int(np.min(c_num))
        ir = np.max(c_num) / min_class_samples

        cc = px.ComplexityCalculator(
            metrics=[
                c.f1,
                c.n2,
                c.lsc,
                c.t4,
            ]
        ).fit(X, y)

        complexities.append([ir, *cc.complexity])

        complexities_str = list(map(lambda x: "%.2f" % x, [ir, *cc.complexity]))

        ds_short_name = ds[:2]
        ds_fam_idx = ds_family_counters[ds_short_name]
        ds_family_counters[ds_short_name] += 1


        ds_idx = ds_short_name + str(ds_fam_idx)
        dataset_indexes.append(ds_idx)
        datasets_dict[ds] = ds_idx

        table.append(
            [
                ds_idx,
                ds,
                str(n_features),
                str(n_samples),
                str(min_class_samples),
                *complexities_str,
            ]
        )

    print(dataset_indexes)

    headers = [
        "dataset",
        "n_features",
        "n_samples",
        "min_class_samples",
        "ir",
        "f1",
        "n2",
        "lsc",
        "t4",
    ]

    # Make table
    with open("datasets.tex", "w") as fp:
        fp.write(latex_table(table, headers=headers))

    X = np.array(complexities)
    np.save(CMPLX_RESULTS_FILE, X)
else:
    X = np.load(CMPLX_RESULTS_FILE)

# Make plot
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[1].axis('off')

ax = axs[0]

X = StandardScaler().fit_transform(X)
X = ClassicalMDS(n_components=2).fit_transform(X)
colors = plt.cm.jet(np.linspace(0, 1, len(DATASETS)))

values, counts = np.unique([x[:2] for x in DATASETS_IDX], return_counts=True)
groups = dict(zip(values.tolist(), counts.tolist()))
print(groups)

for d, s in zip(DATASETS, X):
    l = DATASETS_IDX[d]
    c = xxnn_colors[l]
    if d in DATASETS_SHORT:
        ax.scatter(*s, c=c, label=l, edgecolor='k', s=60, lw=1, zorder=6)
    else:
        ax.scatter(*s, label=l, alpha=0.3, facecolor=c, edgecolor='none', s=60)

ax.grid(ls=":")
ax.set_aspect("equal")

ax.set_xlabel('$MDS^1$')
ax.set_ylabel('$MDS^2$')

ax.legend(
    loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=4,  prop=dict(family="monospace")
)

plt.savefig("foo.png", bbox_inches="tight")
plt.savefig("foo.pdf", bbox_inches="tight")
plt.close()
