import matplotlib.pyplot as plt
import numpy as np

def fbeta(PPV, TPR, beta=1):
    beta_sqr = beta**2
    return (1 + beta_sqr)  * ((PPV * TPR) / (beta_sqr * PPV + TPR))


def PPV_fbeta(fbeta, TPR, beta=1):
    beta_sqr = beta**2
    return (fbeta * TPR) / (-fbeta * beta_sqr + TPR + beta_sqr * TPR)


def TPR_fbeta(fbeta, PPV, beta=1):
    beta_sqr = beta**2
    return -(fbeta * beta_sqr * PPV) / (fbeta - (1 + beta_sqr) * PPV)

P = [0.2, 0.4, 0.6, 0.8]
N = 10000
betas = [0.5, 1, 2]

space = np.linspace(0, 1.0, N, endpoint=True)

FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

for beta in betas:
    fig, axs = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH / 2.5, FIG_A4_WIDTH / 2.5))
    ax = axs

    fig.suptitle(f"$\\beta$ = {beta:.2f}", y=0.92, fontsize=12)

    ax.set_xlabel("TPR")
    ax.set_ylabel("PPV")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.tick_params(top=True, right=True, which='both')

    for p in P:
        y = TPR_fbeta(p, space, beta=beta)

        y_map = np.logical_and(y <= 1.0, y > 0.0)

        ax.plot(y[y_map], space[y_map], c='k', ls="-.")
        ax.text(p + 0.01, p + 0.01, str(p), size='medium')

    ax.grid(ls=':')

    plt.savefig(f'contour_F{beta}.pdf', bbox_inches='tight')
    plt.close()
