from dataclasses import dataclass
from sklearn.pipeline import Pipeline

from ml.xgb.dto.metrics_dto import MetricsDto


@dataclass
class TrainingDto:
    trained_model: Pipeline
    metrics_model: MetricsDto
