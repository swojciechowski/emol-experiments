import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "GM": [
        lambda a, b: np.sqrt(a * b),
        lambda x, b: (x ** 2) / b,
    ],

    "BAC": [
        lambda a, b: (a + b) / 2,
        lambda x, b: (2 * x) - b,
    ],
    "$IBA_1$" : [
        lambda a, b: (1 + a - b) * (a * b),
        # a = 1/2 (sqrt(b^3 - 2 b^2 + b + 4 x)/sqrt(b) + b - 1) and b!=0
        # lambda x, b: (-1 + b + np.sqrt(b - 2 * b ** 2 + b ** 3 + 4 * x) / np.sqrt(b)) / 2,
        lambda x, b: 0.5 * (np.sqrt(b**3 - (2 * (b**2)) + b + (4 * x)) / np.sqrt(b) + b - 1),
    ]
}

P = [0.2, 0.4, 0.6, 0.8]
N = 1000

space = np.linspace(0 + np.finfo(float).eps, 1.0, N, endpoint=True)

FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

for m in METRICS:
    fig, axs = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2.5, FIG_A4_WIDTH / 2.5))

    fig.suptitle(f"{m}", fontsize=12)
    ax = axs

    f, fy = METRICS[m]

    for p in P:
        y = fy(p, space)
        y_map = np.logical_and(y >= 0.0, y <= 1.0)

        ax.plot(y[y_map], space[y_map], c='k', ls="-.")
        ax.text(fy(p, p), p, str(p), horizontalalignment='left',
                verticalalignment='bottom', size='medium')

    ax.set_xlabel("TPR")
    ax.set_ylabel("TNR")

    ax.tick_params(top=True, right=True, which='both')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    ax.grid(ls=':')

    plt.savefig(f'contour_{m}.pdf', bbox_inches='tight')
    plt.close()
