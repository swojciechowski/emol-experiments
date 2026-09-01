import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

from emol.mows import MOWS


RADOM_STATE = 26600

c = np.array(['blue', 'red'])

colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # R -> G -> B
cmap = LinearSegmentedColormap.from_list("name", colors, N=10000)

X, y = make_classification(
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    n_samples=200,
    flip_y=0.0,
    weights=[0.8, 0.2],
    class_sep=0.2,
    random_state=RADOM_STATE,
)

X = MinMaxScaler().fit_transform(X)

N = 200
lx = np.linspace(0, 1, N)
ly = np.linspace(0, 1, N)
xx, yy = np.meshgrid(lx, ly)
space = np.c_[xx.ravel(), yy.ravel()]

# So we know what is the name of the game
X_train = X
y_train = y

from emol.misc.knn import SDKNN
from emol.fitting import SyntheticFitting


clf = SDKNN()
emol = MOWS(estimator=clf, fitting=SyntheticFitting(), use_proba=True, verbose=True, metrics=["PPV", "NPV"])

emol.fit(
    X_train,
    y_train,
)

clf.fit(X_train, y_train)

# print(np.sum(emol._solutions * emol.budget, axis=1))
# print(emol._indicators)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

ax = axs[0, 0]
# ax.scatter(*X_test.T, edgecolor=c.take(y_test), c='none', s=7, lw=0.4)
ax.scatter(*X_train.T, c=c.take(y_train), s=6)
ax.set_aspect("equal")


ax = axs[0, 1]

for cl in np.unique(y):
    kernel = gaussian_kde(X[y==cl].T)
    Z = kernel(space.T)
    # Z = np.round(Z, 2)
    Z = (Z / np.max(Z)) * 0.5
    # ax.scatter(*space.T, alpha=Z, c=c.take(cl), edgecolor='none')
    ax.tricontour(*space.T, Z, levels=4, linewidths=0.5, colors=c.take(cl))

ax.set_aspect("equal")

ax = axs[0, 2]

preds = clf.predict_proba(space)
preds = preds[:, 0] - preds[:, 1]
ax.scatter(*space.T, c=preds, cmap=cmap, edgecolor='none')

ax.set_aspect("equal")


ax = axs[1, 0]

preds = emol.predict_proba(space, pref=[0.01, 0.99])

preds = preds[:, 0] - preds[:, 1]
print(preds.max(), preds.min())

# ax.scatter(*space.T, c=preds, cmap=cmap, edgecolor='none')
img = preds.reshape(N, N)
ax.imshow(img, vmin=-1.0,
    vmax=1.0, extent=[0, 1, 0, 1],
    origin="lower", cmap=cmap)

ax.set_aspect("equal")


ax = axs[1, 1]

preds = emol.predict_proba(space, pref=[0.5, 0.5])

preds = preds[:, 0] - preds[:, 1]
img = preds.reshape(N, N)
ax.imshow(img, vmin=-1.0,
    vmax=1.0, extent=[0, 1, 0, 1],
    origin="lower", cmap=cmap)

ax.set_aspect("equal")

ax = axs[1, 2]

preds = emol.predict_proba(space, pref=[0.99, 0.01])

preds = preds[:, 0] - preds[:, 1]

img = preds.reshape(N, N)
ax.imshow(img, vmin=-1.0,
    vmax=1.0, extent=[0, 1, 0, 1],
    origin="lower", cmap=cmap)

ax.set_aspect("equal")

plt.savefig("foo.png", bbox_inches='tight')
plt.close()
