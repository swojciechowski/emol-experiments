from smote_variants.oversampling import *
from smote_variants.base import OverSamplingBase

# Disable sv logging
import logging
logger = logging.getLogger("smote_variants")
logger.disabled = True

from config import RANDOM_STATE

SMOTEV_DEFAULTS = {
    "A_SUWO": (A_SUWO, {}),
    "ADASYN": (ADASYN, {}),
    "ADG": (ADG, {}),
    "ADOMS": (ADOMS, {}),
    "AHC": (AHC, {}),
    "AMSCO": (AMSCO, {}),
    "AND_SMOTE": (AND_SMOTE, {}),
    "ANS": (ANS, {}),
    "ASMOBD": (ASMOBD, {}),
    "Assembled_SMOTE": (Assembled_SMOTE, {}),
    "Borderline_SMOTE1": (Borderline_SMOTE1, {}),
    "Borderline_SMOTE2": (Borderline_SMOTE2, {}),
    "CBSO": (CBSO, {}),
    "CCR": (CCR, {}),
    "CE_SMOTE": (CE_SMOTE, {}),
    "cluster_SMOTE": (cluster_SMOTE, {}),
    "CURE_SMOTE": (CURE_SMOTE, {}),
    "DBSMOTE": (DBSMOTE, {}),
    "DE_oversampling": (DE_oversampling, {}),
    "DEAGO": (DEAGO, {}),
    "distance_SMOTE": (distance_SMOTE, {}),
    "DSMOTE": (DSMOTE, {}),
    "DSRBF": (DSRBF, {}),
    "E_SMOTE": (E_SMOTE, {}),
    "Edge_Det_SMOTE": (Edge_Det_SMOTE, {}),
    "G_SMOTE": (G_SMOTE, {}),
    "GASMOTE": (GASMOTE, {}),
    "Gaussian_SMOTE": (Gaussian_SMOTE, {}),
    "polynom_fit_SMOTE_star": (polynom_fit_SMOTE_star, {}),
    "polynom_fit_SMOTE_bus": (polynom_fit_SMOTE_bus, {}),
    "polynom_fit_SMOTE_poly": (polynom_fit_SMOTE_poly, {}),
    "polynom_fit_SMOTE_mesh": (polynom_fit_SMOTE_mesh, {}),
    "Gazzah": (Gazzah, {}),
    "IPADE_ID": (IPADE_ID, {}),
    "ISMOTE": (ISMOTE, {}),
    "ISOMAP_Hybrid": (ISOMAP_Hybrid, {}),
    "KernelADASYN": (KernelADASYN, {}),
    "kmeans_SMOTE": (kmeans_SMOTE, {}),
    "Lee": (Lee, {}),
    "LLE_SMOTE": (LLE_SMOTE, {}),
    "LN_SMOTE": (LN_SMOTE, {}),
    "LVQ_SMOTE": (LVQ_SMOTE, {}),
    "MCT": (MCT, {}),
    "MDO": (MDO, {}),
    "MOT2LD": (MOT2LD, {}),
    "MSYN": (MSYN, {}),
    "MWMOTE": (MWMOTE, {}),
    "NDO_sampling": (NDO_sampling, {}),
    "NEATER": (NEATER, {}),
    "NRAS": (NRAS, {}),
    "NRSBoundary_SMOTE": (NRSBoundary_SMOTE, {}),
    "NT_SMOTE": (NT_SMOTE, {}),
    "OUPS": (OUPS, {}),
    "PDFOS": (PDFOS, {}),
    "ProWSyn": (ProWSyn, {}),
    "Random_SMOTE": (Random_SMOTE, {}),
    "ROSE": (ROSE, {}),
    "RWO_sampling": (RWO_sampling, {}),
    "Safe_Level_SMOTE": (Safe_Level_SMOTE, {}),
    "SDSMOTE": (SDSMOTE, {}),
    "Selected_SMOTE": (Selected_SMOTE, {}),
    "SL_graph_SMOTE": (SL_graph_SMOTE, {}),
    "SMMO": (SMMO, {}),
    "SMOBD": (SMOBD, {}),
    "SMOTE_Cosine": (SMOTE_Cosine, {}),
    "SMOTE_D": (SMOTE_D, {}),
    "SMOTE_ENN": (SMOTE_ENN, {}),
    "SMOTE_FRST_2T": (SMOTE_FRST_2T, {}),
    "SMOTE_IPF": (SMOTE_IPF, {}),
    "SMOTE_OUT": (SMOTE_OUT, {}),
    "SMOTE_PSO": (SMOTE_PSO, {}),
    "SMOTE_PSOBAT": (SMOTE_PSOBAT, {}),
    "SMOTE_RSB": (SMOTE_RSB, {}),
    "SMOTE_TomekLinks": (SMOTE_TomekLinks, {}),
    "SMOTE": (SMOTE, {}),
    "SN_SMOTE": (SN_SMOTE, {}),
    "SOI_CJ": (SOI_CJ, {}),
    "SOMO": (SOMO, {}),
    "SPY": (SPY, {}),
    "SSO": (SSO, {}),
    "Stefanowski": (Stefanowski, {}),
    "SUNDO": (SUNDO, {}),
    "Supervised_SMOTE": (Supervised_SMOTE, {}),
    "SVM_balance": (SVM_balance, {}),
    "SYMPROD": (SYMPROD, {}),
    "TRIM_SMOTE": (TRIM_SMOTE, {}),
    "V_SYNTH": (V_SYNTH, {}),
    "VIS_RST": (VIS_RST, {}),
    "MSMOTE": (MSMOTE, {}),
    "SMOTE_AMSR": (SMOTE_AMSR, {}),
    "SMOTEWB": (SMOTEWB, {}),
}

def make_smotev_resampler(resampler_name, random_state=RANDOM_STATE):
    rsmpl_config = SMOTEV_DEFAULTS.get(resampler_name, None)

    if not rsmpl_config:
        return None

    rsmpl_cls, rsmpl_params = rsmpl_config
    return rsmpl_cls(random_state=RANDOM_STATE, **rsmpl_params)

def is_transforming_resampler(resampler):
    return OverSamplingBase.cat_dim_reduction in resampler.categories