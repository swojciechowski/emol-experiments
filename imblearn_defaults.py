from imblearn.under_sampling import *
from imblearn.over_sampling import *

from config import RANDOM_STATE



DEF_IMBL_OVERSAMPLERS = [
        ("ADASYN", ADASYN(random_state=RANDOM_STATE)),
        ("RandomOverSampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("SMOTE", SMOTE(random_state=RANDOM_STATE)),
        ("KMeansSMOTE", KMeansSMOTE(random_state=RANDOM_STATE, cluster_balance_threshold=0.0)),
        ("BorderlineSMOTE", BorderlineSMOTE(random_state=RANDOM_STATE)),
        ("SVMSMOTE", SVMSMOTE(random_state=RANDOM_STATE)),
        ("SMOTENC", SMOTENC(categorical_features="auto", random_state=RANDOM_STATE)),
        ("SMOTEN", SMOTEN(random_state=RANDOM_STATE)),
    ]

DEF_IMBL_UNDERSAMPLERS = [
        ("RandomUnderSampler", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("NearMiss", NearMiss()),
        ("TomekLinks", TomekLinks()),
        ("CondensedNearestNeighbour", CondensedNearestNeighbour(random_state=RANDOM_STATE)),
        ("EditedNearestNeighbours", EditedNearestNeighbours()),
        ("ClusterCentroids", ClusterCentroids(random_state=RANDOM_STATE)),
        ("InstanceHardnessThreshold", InstanceHardnessThreshold(random_state=RANDOM_STATE)),
        ("OneSidedSelection", OneSidedSelection(random_state=RANDOM_STATE)),
        ("NeighbourhoodCleaningRule", NeighbourhoodCleaningRule()),
        ("AllKNN", AllKNN()),
    ]

DEF_IMBL_RESAMPERS = DEF_IMBL_OVERSAMPLERS + DEF_IMBL_UNDERSAMPLERS
