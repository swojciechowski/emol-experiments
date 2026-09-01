import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

BASE_METRIC_NAMES = ["TPR", "TNR", "PPV", "NPV"]

nbins = 100

M_comb = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
]


def f_tpr(cm):
    return cm[1, 1] / np.sum(cm[1, :]) if np.sum(cm[1, :]) != 0 else 0


def f_tnr(cm):
    return cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) != 0 else 0


def f_ppv(cm):
    return cm[1, 1] / np.sum(cm[:, 1]) if np.sum(cm[:, 1]) != 0 else 0


def f_npv(cm):
    return cm[0, 0] / np.sum(cm[:, 0]) if np.sum(cm[:, 0]) != 0 else 0


def fbeta(PPV, TPR, beta=1.0):
    beta_sqr = beta**2
    return (1 + beta_sqr) * ((PPV * TPR) / (beta_sqr * PPV + TPR))


METRICS = (
    ("BAC", lambda tpr, tnr, ppv, npv: (tpr + tnr) / 2),
    ("GM", lambda tpr, tnr, ppv, npv: np.sqrt(tpr * tnr)),
    ("IBA", lambda tpr, tnr, ppv, npv: (1 + tpr - tnr) * (tpr * tnr)),
    ("MD", lambda tpr, tnr, ppv, npv: ppv + npv - 1),
    ("F1", lambda tpr, tnr, ppv, npv: fbeta(ppv, tpr, 1.0)),
    ("F01", lambda tpr, tnr, ppv, npv: fbeta(ppv, tpr, 0.1)),
    ("F10", lambda tpr, tnr, ppv, npv: fbeta(ppv, tpr, 10)),
)

n_samples = 100
levels = 5
SF = 6

RATIOS = [0.5]

for min_r in RATIOS:
    n_min = int(n_samples * min_r)
    n_maj = n_samples - n_min

    print(f"TOTAL: {n_min + n_maj}: {n_min}, {n_maj}")

    f_space = []

    for t_maj in range(n_maj + 1):
        for t_min in range(n_min + 1):
            cm = np.array([[t_maj, n_maj - t_maj], [n_min - t_min, t_min]])

            f_space.append([f_tpr(cm), f_tnr(cm), f_ppv(cm), f_npv(cm)])

    f_space = np.array(f_space)

    for m_name, m in METRICS:
        fig, axs = plt.subplots(2, 2, figsize=(2 * SF, 2 * SF))

        fig.suptitle(m_name)
        axs = iter(axs.flatten())

        z = m(*f_space.T)
        z[np.isnan(z)] = 0

        for comb in M_comb:
            ax = next(axs)

            x, y = f_space[:, comb].T

            ax.tricontourf(x, y, z, levels=100)
            ax.scatter(x, y, edgecolor='w', c=z)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(BASE_METRIC_NAMES[comb[0]])
            ax.set_ylabel(BASE_METRIC_NAMES[comb[1]])
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(f"space_{m_name}_{min_r}.png")
        plt.close()
        plt.clf()
