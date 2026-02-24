from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from schemas.common_ml.repository_schema import ModelRepositoryResponse
from schemas.prediction_ml.prediction_schema import PredictResponse, PredictRequest
from schemas.training_ml.training_schema import TrainRequest, TrainResponse
from sklearn.pipeline import Pipeline


class TrainingService(ABC):
    @abstractmethod
    def process(self, request: TrainRequest) -> TrainResponse: ...


class PredictingService(ABC):
    @abstractmethod
    def process(self, predict_obj: PredictRequest) -> PredictResponse: ...


class ProbabilityService(ABC):
    @abstractmethod
    def process(self, X: Any) -> np.ndarray: ...


class RepositoryService(ABC):

    def save_model(self, pipe_model: Pipeline, name_model: str) -> None: ...

    @abstractmethod
    def load_model(self, name_model: str) -> Pipeline: ...

    @abstractmethod
    def list_all_models(self) -> list[ModelRepositoryResponse]: ...


class AssessmentService(ABC):
    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]: ...
