import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from emol.fbeta.fbeta import fbeta
from emol.fbeta.plot import fbeta_plot_combined_cv
from emol.helpers.plot.color import generate_distinct_colors, DEFAULT_CMAP
from emol.misc.generator.random_predictions import random_results, default_config_gen

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS = ['TPR', "PPV"]

methods_num = []

seed = np.random.SeedSequence().entropy
# seed = 42
print("Seed", seed)

results = [
    random_results(n_samples=100, n_methods=5, n_repeats=10, ir=2.0, config_gen=default_config_gen(pos_err_exp_range=(0.5, 0.7), random_state=seed) ,random_state=seed)[0][..., (0, 2)].swapaxes(0, 1),

    random_results(n_samples=100, n_methods=7, n_repeats=10, ir=2.0, config_gen=default_config_gen(neg_err_exp_range=(0.5, 0.7), random_state=seed), random_state=seed)[0][..., (0, 2)].swapaxes(0, 1),
]

fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})

ax = axs[0]
ax.set_visible(False)

# ax.spines[["right", "top"]].set_visible(False)
# ax.set_xlim(0, 1.01)
# ax.set_ylim(0, 1.01)
# # ax.set_yscale('function', functions=(lambda x: 10**x, np.log10))
# # ax.set_xscale('function', functions=(lambda x: 10**x, np.log10))
# ax.set_xticks([0, 0.5, 0.75, 1.0])
# ax.set_yticks([0, 0.5, 0.75, 1.0])
# ax.set_aspect("equal")
# ax.grid(ls=":")
# ax.set_xlabel("TPR")
# ax.set_ylabel("PPV")

colors = [DEFAULT_CMAP[1], DEFAULT_CMAP[2]]

# for r, c in zip(results, colors):
#     ax.scatter(*r.T, c=c, s=5)

ax = axs[1]

print(len(results), results[0].shape)

fbeta_plot_combined_cv(results, ax=ax, color_all=True, colors=colors, labels=["A", "B"])

plt.savefig("foo.png", bbox_inches='tight')
# plt.savefig("foo.png")
plt.clf()
plt.close()
