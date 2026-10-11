import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__))

# Default directories
DEFAULT_KEEL_DIR = os.path.join(PROJECT_ROOT, "keel")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "_results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "_figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "_tables")

# Constants
FIG_A4_HEIGHT, FIG_A4_WIDTH = (11.69, 8.27)

# Executor config
N_JOBS = 1

# Validation Protocol
from sklearn.model_selection import RepeatedStratifiedKFold

RANDOM_STATE = 90210
DEFAULT_VALIDATION = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=RANDOM_STATE)
N_SPLITS = DEFAULT_VALIDATION.get_n_splits()

# Method configs
BASE_CLASSIFIERS = ["GNB", "KNN", "CRT", "SVM", "MLP"]

# Dataset configs
DATASETS_FULL = [
    "ecoli-0-1-4-7_vs_2-3-5-6",
    "ecoli-0-1-4-7_vs_5-6",
    "ecoli-0-3-4-7_vs_5-6",
    "ecoli1",
    "ecoli2",
    "ecoli3",
    "glass-0-1-2-3_vs_4-5-6",
    "glass0",
    "glass6",
    "haberman",
    "iris0",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "new-thyroid1",
    "page-blocks-1-3_vs_4",
    "page-blocks0",
    "poker-8-9_vs_5",
    "poker-8-9_vs_6",
    "segment0",
    "shuttle-2_vs_5",
    "shuttle-c0-vs-c4",
    "vehicle0",
    "vehicle1",
    "vehicle2",
    "vehicle3",
    "vowel0",
    "winequality-red-4",
    "winequality-white-3-9_vs_5",
    "yeast-0-2-5-6_vs_3-7-8-9",
    "yeast-0-2-5-7-9_vs_3-6-8",
    "yeast-0-3-5-9_vs_7-8",
    "yeast-0-5-6-7-9_vs_4",
    "yeast-1-2-8-9_vs_7",
    "yeast-1-4-5-8_vs_7",
    "yeast-1_vs_7",
    "yeast-2_vs_4",
    "yeast1",
    "yeast3",
    "yeast4",
    "yeast5",
    "yeast6",
]

DATASETS_SHORT = [
    "ecoli-0-1-4-7_vs_2-3-5-6",
    "glass-0-1-2-3_vs_4-5-6",
    "haberman",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "new-thyroid1",
    # "iris0",
    "page-blocks-1-3_vs_4",
    "poker-8-9_vs_5",
    "segment0",
    "shuttle-2_vs_5",
    "vehicle0",
    "vowel0",
    "winequality-red-4",
    "winequality-white-3-9_vs_5",
    "yeast-0-2-5-6_vs_3-7-8-9",
]

DATASETS = DATASETS_FULL # Select value for experiments

# SMOTEV configs
SMOTEV_FULL = [
    "A_SUWO",
    "ADASYN",
    "ADG",
    "ADOMS",
    "AHC",
    "AMSCO",
    "AND_SMOTE",
    "ANS",
    "ASMOBD",
    "Assembled_SMOTE",
    "Borderline_SMOTE1",
    "Borderline_SMOTE2",
    "CBSO",
    "CCR",
    "CE_SMOTE",
    "cluster_SMOTE",
    "CURE_SMOTE",
    "DBSMOTE",
    "DE_oversampling",
    "DEAGO",
    "distance_SMOTE",
    "DSMOTE",
    "DSRBF",
    "E_SMOTE",
    "Edge_Det_SMOTE",
    "G_SMOTE",
    "GASMOTE",
    "Gaussian_SMOTE",
    "polynom_fit_SMOTE_star",
    "polynom_fit_SMOTE_bus",
    "polynom_fit_SMOTE_poly",
    "polynom_fit_SMOTE_mesh",
    "Gazzah",
    "IPADE_ID",
    "ISMOTE",
    "ISOMAP_Hybrid",
    "KernelADASYN",
    "kmeans_SMOTE",
    "Lee",
    "LLE_SMOTE",
    "LN_SMOTE",
    "LVQ_SMOTE",
    "MCT",
    "MDO",
    # "MOT2LD",
    "MSYN",
    "MWMOTE",
    "NDO_sampling",
    "NEATER",
    "NRAS",
    "NRSBoundary_SMOTE",
    "NT_SMOTE",
    "OUPS",
    "PDFOS",
    "ProWSyn",
    "Random_SMOTE",
    "ROSE",
    "RWO_sampling",
    "Safe_Level_SMOTE",
    "SDSMOTE",
    "Selected_SMOTE",
    "SL_graph_SMOTE",
    "SMMO",
    "SMOBD",
    "SMOTE_Cosine",
    "SMOTE_D",
    "SMOTE_ENN",
    "SMOTE_FRST_2T",
    "SMOTE_IPF",
    "SMOTE_OUT",
    "SMOTE_PSO",
    "SMOTE_PSOBAT",
    "SMOTE_RSB",
    "SMOTE_TomekLinks",
    "SMOTE",
    "SN_SMOTE",
    "SOI_CJ",
    "SOMO",
    "SPY",
    "SSO",
    "Stefanowski",
    "SUNDO",
    "Supervised_SMOTE",
    "SVM_balance",
    "SYMPROD",
    "TRIM_SMOTE",
    "V_SYNTH",
    "VIS_RST",
    "MSMOTE",
    "SMOTE_AMSR",
    "SMOTEWB",
]

SMOTEV_TOP = [
    "polynom_fit_SMOTE_star",
    "ProWSyn",
    "SMOTE_IPF",
    "Lee",
    "SMOBD",
    "G_SMOTE",
    "CCR",
    "LVQ_SMOTE",
    "Assembled_SMOTE",
    "SMOTE_TomekLinks",
]

SMOTEV_RESAMPLERS = SMOTEV_TOP # Select value for experiments

# imbalanced-learn config
IMBL_OVERSAMPLERS = [
    "RandomOverSampler",
    "KMeansSMOTE",
    "SMOTE",
    "BorderlineSMOTE",
    "SVMSMOTE",
    # Not Applicable
    # "ADASYN",
    # "SMOTENC",
    # "SMOTEN",
]

IMBL_UNDERSAMPLERS = [
    "RandomUnderSampler",
    "ClusterCentroids",
    "InstanceHardnessThreshold",
    "TomekLinks",
    "EditedNearestNeighbours",
    "NearMiss",
    "AllKNN",
    "OneSidedSelection",
    "CondensedNearestNeighbour",
    "NeighbourhoodCleaningRule",
]

IMBL_RESAMPERS = IMBL_OVERSAMPLERS + IMBL_UNDERSAMPLERS # Select value for experiments

# emol configs
MEUS_CONFIG = [
    "MEUS-HoldOutFitting-balanced_accuracy-c",
    "MEUS-HoldOutFitting-TPR+PPV-c",
    "MEUS-HoldOutFitting-TPR+TNR-c",
    "MEUS-HoldOutFitting-TPR+TNR+PPV-c",
]

# MOWS config use old splitting names, change to new name for reproduction
# MOWS_CONFIG = [
#     "MOWS-SyntheticFitting-TPR+TNR-c",
#     "MOWS-SyntheticFitting-TPR+TNR-uc",
#     "MOWS-LeakingFitting-TPR+TNR-c",
#     "MOWS-LeakingFitting-TPR+TNR-uc",
# ]

MOWS_CONFIG = [
    "MOWS-BootstrapFitting-TPR+TNR-c",
    "MOWS-BootstrapFitting-TPR+TNR-uc",
    "MOWS-DummyFitting-TPR+TNR-c",
    "MOWS-DummyFitting-TPR+TNR-uc",
]

# Configuration for results
MEUS_SEL = "MEUS-HoldOutFitting-TPR+TNR-c"
MOWS_SEL = "MOWS-DummyFitting-TPR+TNR-c"
MOWS_RANKING = [MOWS_SEL] + SMOTEV_TOP
MEUS_RANKING = [MEUS_SEL] + IMBL_UNDERSAMPLERS
MOL_METHODS = [MEUS_SEL, MOWS_SEL]

# Name tables for short names
DATASETS_IDX = {
    "ecoli-0-1-4-7_vs_2-3-5-6": "ec01",
    "ecoli-0-1-4-7_vs_5-6": "ec02",
    "ecoli-0-3-4-7_vs_5-6": "ec03",
    "ecoli1": "ec04",
    "ecoli2": "ec05",
    "ecoli3": "ec06",
    "glass-0-1-2-3_vs_4-5-6": "gl01",
    "glass0": "gl02",
    "glass6": "gl03",
    "haberman": "ha01",
    "iris0": "ir01",
    "led7digit-0-2-4-5-6-7-8-9_vs_1": "le01",
    "new-thyroid1": "ne01",
    "page-blocks-1-3_vs_4": "pa01",
    "page-blocks0": "pa02",
    "poker-8-9_vs_5": "po01",
    "poker-8-9_vs_6": "po02",
    "segment0": "se01",
    "shuttle-2_vs_5": "sh01",
    "shuttle-c0-vs-c4": "sh02",
    "vehicle0": "ve01",
    "vehicle1": "ve02",
    "vehicle2": "ve03",
    "vehicle3": "ve04",
    "vowel0": "vo01",
    "winequality-red-4": "wi01",
    "winequality-white-3-9_vs_5": "wi02",
    "yeast-0-2-5-6_vs_3-7-8-9": "ye01",
    "yeast-0-2-5-7-9_vs_3-6-8": "ye02",
    "yeast-0-3-5-9_vs_7-8": "ye03",
    "yeast-0-5-6-7-9_vs_4": "ye04",
    "yeast-1-2-8-9_vs_7": "ye05",
    "yeast-1-4-5-8_vs_7": "ye06",
    "yeast-1_vs_7": "ye07",
    "yeast-2_vs_4": "ye08",
    "yeast1": "ye09",
    "yeast3": "ye10",
    "yeast4": "ye11",
    "yeast5": "ye12",
    "yeast6": "ye13",
}

SMOTEV_NAME = {
    "polynom_fit_SMOTE_star": "Poly-SMOTE",
    "ProWSyn": "PWSyn",
    "SMOTE_IPF": "SMOTE-IPF",
    "Lee": "LEE",
    "SMOBD": "SMOBD",
    "G_SMOTE": "G-SMOTE",
    "CCR": "CCR",
    "LVQ_SMOTE": "LVQ-SMOTE",
    "Assembled SMOTE": "ASMOTE",
    "SMOTE_TomekLinks": "SMOTE-TL",
}

METHODS_SHORT_NAMES = {
    "RandomUnderSampler": "rus",
    "ClusterCentroids": "cct",
    "InstanceHardnessThreshold": "iht",
    "TomekLinks": "tln",
    "EditedNearestNeighbours": "enn",
    "NearMiss": "nms",
    "AllKNN": "akn",
    "OneSidedSelection": "oss",
    "CondensedNearestNeighbour": "cnn",
    "NeighbourhoodCleaningRule": "ncr",
    #
    "polynom_fit_SMOTE_star" : "pfs",
    "ProWSyn" : "pws",
    "SMOTE_IPF" : "ipf",
    "Lee" : "lee",
    "SMOBD" : "sbd",
    "G_SMOTE" : "gsm",
    "CCR" : "ccr",
    "LVQ_SMOTE" : "lvq",
    "Assembled_SMOTE" : "ass",
    "SMOTE_TomekLinks" : "stl",
    #
    "MEUS-HoldOutFitting-balanced_accuracy-c": "s-bac",
    "MEUS-HoldOutFitting-TPR+PPV-c": "m-ss",
    "MEUS-HoldOutFitting-TPR+TNR-c": "m-sp",
    "MEUS-HoldOutFitting-TPR+TNR+PPV-c": "m-ssp",
    #
    "MOWS-BootstrapFitting-TPR+TNR-c": "synt-c",
    "MOWS-BootstrapFitting-TPR+TNR-uc": "synt-u",
    "MOWS-DummyFitting-TPR+TNR-c": "leak-c",
    "MOWS-DummyFitting-TPR+TNR-uc": "leak-u",
}
