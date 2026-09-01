import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from emol.fbeta.fbeta import fbeta_analysis
from emol.misc.generator.random_clusters import random_clusters
from emol.moo.evaluation.metrics import non_dominated_solutions


# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def calc_tnr(tpr, ppv, ir):
    return 1 - (1 / ir) * tpr * ((1 - ppv) / ppv)

METRICS = ['TPR', "PPV"]

IR = [2.0, 9.0, 18.0]

methods_num = []

seed = np.random.SeedSequence().entropy
seed = 42
print("Seed", seed)

results, _ = random_clusters(num_centers=100, rel_var_range=[0.001, 0.01], seed=seed)

best, point = fbeta_analysis(results)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs = iter(axs)

ax = next(axs)

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.01)
# ax.set_yscale('function', functions=(lambda x: 10**x, np.log10))
# ax.set_xscale('function', functions=(lambda x: 10**x, np.log10))
ax.set_xticks([0, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.5, 0.75, 1.0])
ax.set_aspect("equal")
ax.grid(ls=":")
ax.set_xlabel("TPR")
ax.set_ylabel("PPV")

ax.scatter(*results.T, c='grey', s=5)

for b in best:
    ax.scatter(*results[b], s=25, marker='x')

nds = non_dominated_solutions(results)
ax.scatter(*results[nds].T, s=35, marker='o', facecolor='none', edgecolor='k')

for ir in IR:
    print(ir)
    tnr = calc_tnr(results[:, 0], results[:, 1], ir=ir)

    ax = next(axs)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    # ax.set_yscale('function', functions=(lambda x: 10**x, np.log10))
    # ax.set_xscale('function', functions=(lambda x: 10**x, np.log10))
    ax.set_xticks([0, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 0.75, 1.0])
    ax.set_aspect("equal")
    ax.grid(ls=":")
    ax.set_xlabel("TPR")
    ax.set_ylabel("TNR")

    ax.scatter(results[:, 0], tnr, c='grey', s=5)

    for b in best:
        ax.scatter(results[b, 0], tnr[b], s=15, marker='x')

    new_res = np.array([results[:, 0], tnr]).T

    print(new_res.shape)

    nds = non_dominated_solutions(new_res)
    ax.scatter(*new_res[nds].T, s=35, marker='o', facecolor='none', edgecolor='k')

plt.savefig("foo.png", bbox_inches='tight')
plt.clf()
plt.close()
