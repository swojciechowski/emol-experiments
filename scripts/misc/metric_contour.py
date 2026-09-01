import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "Gmean": [
        lambda a, b: np.sqrt(a * b),
        lambda x, b: (x ** 2) / b,
    ],

    "BAC": [
        lambda a, b: (a + b) / 2,
        lambda x, b: (2 * x) - b,
    ],
    "IBA" : [
        lambda a, b: (1 + a - b) * (a * b),
        lambda x, b: (-1 + b + np.sqrt(b - 2 * b ** 2 + b ** 3 + 4 * x) / np.sqrt(b)) / 2,
    ]
}

P = [0.2, 0.4, 0.6, 0.8]
N = 1000

space = np.linspace(0 + np.finfo(float).eps, 1.0, N, endpoint=True)

FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

for m in METRICS:
    fig, axs = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2.5, FIG_A4_WIDTH / 2.5))

    fig.suptitle(f"{m}", y=0.92, fontsize=12)
    ax = axs

    f, fy = METRICS[m]

    for p in P:
        y = fy(p, space)
        y_map = np.logical_and(y >= 0.0, y <= 1.0)

        ax.plot(space[y_map], y[y_map], c='k', ls="-.")
        ax.text(p, fy(p, p), str(p), horizontalalignment='left',
                verticalalignment='bottom', size='medium')

    ax.set_ylabel("TNR")
    ax.set_xlabel("TPR")

    ax.tick_params(top=True, right=True, which='both')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(ls=':')

    plt.tight_layout()
    plt.savefig(f'contour_{m}.pdf', bbox_inches='tight')
    plt.savefig("foo.png", bbox_inches='tight')
    plt.close()
    plt.clf()
