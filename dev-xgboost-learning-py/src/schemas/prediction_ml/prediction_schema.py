from pydantic import BaseModel, Field
from typing import List
from schemas.common_ml.feature_schema import Feature


class FeaturePredict(Feature):
    client_id: str = Field(..., description="Identificación del cliente")


class TopClass(BaseModel):
    risk: str
    score: str


class PredictResult(BaseModel):
    client_id: str = Field(..., description="Identificacion Cliente")
    age: int = Field(..., description="Idade Cliente")
    risk: str = Field(..., description="Etiqueta predicha")
    score: str = Field(..., description="Probabilidad de la clase predicha")
    risk_ranking: List[TopClass] = Field(...,description="Ranking de clases más probables")


class PredictRequest(BaseModel):
    model_id: str
    features: List[FeaturePredict]


class PredictResponse(BaseModel):
    predictions: List[PredictResult]


