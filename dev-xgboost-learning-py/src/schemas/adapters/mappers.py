import pandas as pd
from typing import Sequence

from core.enums import RiskLevel
from schemas.common_ml.feature_schema import Feature
from schemas.training_ml.target_schema import Target

TARGET_MAPPING = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2
}

FEATURE_ORDER = [
    "edad",
    "ingresos_mensuales",
    "antiguedad_meses",
    "incidentes_previos",
    "ratio_deuda_ingresos",
    "num_productos",
    "canal"
]

NUMERIC_FEATURES = [
    "edad",
    "ingresos_mensuales",
    "antiguedad_meses",
    "incidentes_previos",
    "ratio_deuda_ingresos",
    "num_productos"
]

CATEGORICAL_FEATURES = [
    "canal"
]

META_COLUMNS = ["client_id"]
DF_COLUMNS = META_COLUMNS + FEATURE_ORDER


# LOGICA CONVERSION ----> Objeto Pydantic -> Panda/DataFrame
def feature_training_mapper(features: Sequence[Feature]) -> pd.DataFrame:
    if not features:
        raise ValueError("feature empty")

    feature_df = pd.DataFrame([f.model_dump() for f in features])
    feature_df = feature_df.reindex(columns=FEATURE_ORDER)

    if feature_df.isnull().any().any():
        raise ValueError("Features contienen valores nulos")

    return feature_df


def feature_prediction_mapper(features: Sequence[Feature]) -> pd.DataFrame:
    if not features:
        raise ValueError("feature empty")

    feature_df = pd.DataFrame([f.model_dump() for f in features])
    feature_df["client_id"] = [f.client_id for f in features]
    feature_df = feature_df.reindex(columns=DF_COLUMNS)

    if feature_df.isnull().any().any():
        raise ValueError("Features contienen valores nulos")

    return feature_df


def target_training_mapper(target: Sequence[Target]) -> pd.Series:
    target_df = pd.Series(
        [TARGET_MAPPING[label.risk_level] for label in target],
        name="target"
    )
    return target_df
