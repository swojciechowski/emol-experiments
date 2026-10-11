from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Replace sklearn KNN with KNN supporting weighting
# from sklearn.neighbors import KNeighborsClassifier
from emol.misc.knn import KNN as KNeighborsClassifier


from config import RANDOM_STATE

# Disable Convergence Warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


DEF_BASE_CLASSIFIERS = [
    ("GNB", GaussianNB()),
    ("KNN", KNeighborsClassifier()),
    ("CRT", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("SVM", SVC(random_state=RANDOM_STATE, tol=0.005, max_iter=100)),
    (
        "MLP",
        MLPClassifier(
            random_state=RANDOM_STATE,
            tol=0.005,
            max_iter=100,
        ),
    ),
]
