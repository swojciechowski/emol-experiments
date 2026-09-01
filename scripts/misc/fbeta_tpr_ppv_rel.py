import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from emol.fbeta.fbeta import fbeta_analysis, fbeta, fbeta_combined_curve_helper
from emol.fbeta.plot import fbeta_plot, DEFAULT_CMAP
from emol.misc.generator.random_predictions import random_results


# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS = ['TPR', "PPV"]

seed = np.random.SeedSequence().entropy
# seed = 42
print("Seed", seed)

results, _ = random_results(n_samples=100, ir=1.0, n_methods=7, n_repeats=1, random_state=seed)

results = results[:, 0, (0, 2)]

best, point = fbeta_analysis(results)
# best, point = fbeta_analysis_aprx(results)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})

ax = axs[0]

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

colors = DEFAULT_CMAP

for b in best:
    ax.scatter(*results[b], s=170, edgecolor='k', facecolor='k', marker='.')

for r, c in zip(results, colors):
    ax.scatter(*r, c=c, s=75, marker='.')

ax = axs[1]

fbeta_plot(results, color_all=True, colors=None, ax=ax)

lim, n = 1000, 301
betas = np.geomspace(1 / lim, lim, n)
f_betas = fbeta(results[:, 0], results[:, 1], betas=betas)
combined_fbeta = fbeta_combined_curve_helper(f_betas)

ax.plot(betas, combined_fbeta, ls='--', c='k', lw=2.5)

plt.savefig("foo.png", bbox_inches='tight')
plt.clf()
plt.close()
