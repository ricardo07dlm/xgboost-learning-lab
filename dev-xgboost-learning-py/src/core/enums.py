from enum import Enum


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    MULTI_XGBOOST = "multi:softprob"


class TrainingStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TRAINED = "trained"


class PredictionType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskClass(int, Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
