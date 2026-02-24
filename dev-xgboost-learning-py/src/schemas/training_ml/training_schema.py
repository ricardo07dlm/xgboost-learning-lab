from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from schemas.training_ml.target_schema import Target
from schemas.common_ml.feature_schema import Feature


class Params(BaseModel):
    n_estimators: int
    learning_rate: float
    max_depth: int




class TrainRequest(BaseModel):
    features: List[Feature]
    target: List[Target]
    params: Optional[Params] = None

    @classmethod
    def validate_labels_length(cls, labels, info):
        features = info.data.get("features") or []
        if len(labels) != len(features):
            raise ValueError("labels debe tener la misma longitud que features")
        return labels


# CLASS RESPONSE
class MetricsResponse(BaseModel):
    accuracy: str = Field(..., description="Overall model accuracy (percentage)")
    precision: str = Field(..., description="Model precision (percentage)")
    recall: str = Field(..., description="Model recall (percentage)")
    f1_score: str = Field(..., description="Model F1-score (percentage)")


class ModelInfoResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the trained model")
    lifecycle: str = Field(..., description="Lifecycle Process ML")


class ModelInfoResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the trained model")
    lifecycle: str


class TrainingSummaryResponse(BaseModel):
    samples_used: int = Field(..., description="Number of dataset samples used during training")
    feature_dimension: int = Field(..., description="Dimensionality of the model feature space")


class TrainResponse(BaseModel):
    model: ModelInfoResponse
    performance: MetricsResponse
    training_summary: TrainingSummaryResponse
    features: List[str]
