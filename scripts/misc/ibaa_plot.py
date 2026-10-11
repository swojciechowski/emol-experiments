import matplotlib.pyplot as plt
import numpy as np

def iba(TPR, TNR, alpha=1):
    # IBAα = (1 + α· (TPrate− TNrate))· TPrate· TNrate
    return (1 + (alpha * (TPR - TNR))) * TPR * TNR

N = 100

PLOTS = [
    [0.1, 1.0],
    [0.25, 0.75],
    [0.5, 0.5],
    [0.75, 0.25],
    [1.0, 0.1],
]

space = np.linspace(0.0, 10.0, N, endpoint=True)


FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

fig, axs = plt.subplots(1, 1, figsize=(FIG_A4_WIDTH, FIG_A4_WIDTH))
ax = axs

# fig.suptitle(f"$\\alpha$ = {alpha:.2f}", y=0.92, fontsize=12)

ax.set_aspect('equal')
ax.tick_params(top=True, right=True, which='both')

for tpr, tnr in PLOTS:
    y = iba(tpr, tnr, alpha=space)
    ax.plot(space, y)

ax.grid(ls=':')

plt.savefig(f'foo.png')
plt.close()
