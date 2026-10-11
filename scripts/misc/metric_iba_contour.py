import matplotlib.pyplot as plt
import numpy as np

def iba(TPR, TNR, alpha=1):
    return (1 + (alpha * (TPR - TNR))) * TPR * TNR


def TNR_aiba(iba, TPR, alpha=1):
    numerator = (
        np.sqrt(alpha**2 * TPR**3 + 2 * alpha * TPR**2 + TPR - 4 * alpha * iba)
        / np.sqrt(TPR)
        + alpha * TPR
        + 1
    )
    denominator = 2 * alpha

    return numerator / denominator


def TPR_aiba(iba, TNR, alpha=1):
    numerator = (
        np.sqrt(alpha**2 * TNR**3 - 2 * alpha * TNR**2 + TNR + 4 * alpha * iba)
        / np.sqrt(TNR)
        - alpha * TNR
        + 1
    )
    denominator = 2 * alpha

    return numerator / denominator

P = [0.2, 0.4, 0.6, 0.8]
N = 10000
alphas = [0.9, 1, 100]

space = np.linspace(0.1, 1.0, N, endpoint=True)

FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

for alpha in alphas:
    fig, axs = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2.5, FIG_A4_WIDTH / 2.5))
    ax = axs

    fig.suptitle(f"$\\alpha$ = {alpha:.2f}", y=0.92, fontsize=12)

    ax.set_xlabel("TPR")
    ax.set_ylabel("PPV")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.tick_params(top=True, right=True, which='both')

    for p in P:
        y = TPR_aiba(p, space, alpha=alpha)

        print(y)

        y_map = np.logical_and(y <= 1.0, y > 0.0)

        ax.plot(space[y_map], y[y_map], c='k', ls="-.")

    ax.grid(ls=':')

    plt.savefig(f'contour_IBA_{alpha}.pdf', bbox_inches='tight')
    plt.close()
