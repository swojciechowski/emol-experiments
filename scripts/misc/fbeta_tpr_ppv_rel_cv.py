import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from emol.fbeta.fbeta import fbeta
from emol.fbeta.plot import fbeta_plot_cv
from emol.helpers.plot.color import generate_distinct_colors, DEFAULT_CMAP
from emol.misc.generator.random_predictions import random_results

# Import config
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


seed = np.random.SeedSequence().entropy
# seed = 42
seed = 175488963808876867041187382716138152518
print("Seed", seed)

results, _ = random_results(n_samples=100, n_methods=7, n_repeats=10, ir=1.0, random_state=seed)
results = results[..., (0, 2)]

# lim, n = 100, 301
# betas = np.geomspace(1 / lim, lim, n)
# f_betas = fbeta(results[..., 0], results[..., 1], betas=betas)

# macro_avg_fbetas = np.mean(f_betas, axis=1)
# macro_std_fbetas = np.std(f_betas, axis=1)

# micro_avg_fbetas = fbeta(results_mean[..., 0], results_mean[..., 1], betas=betas)
# micro_std_fbetas = fbeta(results_std[..., 0], results_std[..., 1], betas=betas)

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

colors = generate_distinct_colors(len(results))

for r, c in zip(results, colors):
    ax.scatter(*r.T, marker='.', c=c, s=5, alpha=0.5)


for r, c in zip(np.mean(results, axis=1), colors):
    ax.scatter(*r.T, marker='x', c=c, s=25)

ax = axs[1]

labels = [f"{a:.2f}+{b:.2f}" for a,b in zip(*results.mean(axis=1).T)]

fbeta_plot_cv(results, ax=ax, color_all=False, colors=colors, labels=labels)

plt.savefig("foo.png", bbox_inches='tight')
plt.clf()
plt.close()
