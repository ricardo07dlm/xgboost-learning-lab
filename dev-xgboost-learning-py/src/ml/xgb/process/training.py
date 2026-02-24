from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from core.exceptions.training_exceptions import (
    TrainingError,
    TrainingDataSetEmpty_Error,
    TrainingDataSetClass_Error,
    TrainingSplitInvalid_Error,
    TrainingFit_Error,
    TrainingEvaluation_Error, TrainingUnexpected_Error)
from ml.xgb.dto.metrics_dto import MetricsDto
from ml.xgb.dto.training_dto import TrainingDto
from schemas.training_ml.training_schema import Params
from sklearn.pipeline import Pipeline
from typing import Tuple
import logging
from ml.xgb.builder.preprocessor import build_preprocessor
logger = logging.getLogger(__name__)


class Training:

    def execute(self, x: DataFrame, y: Series, parameter: Params) -> TrainingDto:
        logger.info("Execute :: Training & Validation Phase")
        try:
            # (1) Split
            x_train, x_test, y_train, y_test = self._split_dataset(x_train=x, y_train=y)
            # (2) FIT - Training
            pipeline_trained = self._fit_training(x_train, y_train, parameter)
            # (3) Evaluación
            metricsDto = self._evaluation_model(x_test=x_test, y_test=y_test, pipe=pipeline_trained)
            # (4) Return Object DTO
            return TrainingDto(
                trained_model=pipeline_trained,
                metrics_model=metricsDto
            )
        except TrainingError as e:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error Training & Validation Phase {e}")
            raise TrainingUnexpected_Error(str(e)) from e


    @staticmethod
    def _split_dataset(x_train: DataFrame, y_train: Series) -> Tuple[DataFrame, DataFrame, Series, Series]:
        logger.info("Execute :: Dataset Splitting Process")

        if x_train is None or len(x_train) == 0:
            raise TrainingDataSetEmpty_Error("Features dataset vacío")

        if y_train is None or len(y_train) == 0:
            raise TrainingDataSetEmpty_Error("Target dataset vacío")

        # clases insuficientes (ej. clasificación)
        if getattr(y_train, "nunique", None) and y_train.nunique() < 2:
            raise TrainingDataSetClass_Error(f"n_classes={y_train.nunique()}")

        try:
            return train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )
        except ValueError as e:
            raise TrainingSplitInvalid_Error(
                f"{str(e)} | X shape={x_train.shape}, y shape={y_train.shape}"
            ) from e


    @staticmethod
    def _fit_training(x_train: DataFrame, y_train: Series, xgb_parameter: Params) -> Pipeline:
        logger.info("Execute :: Model Fitting Process")
        try:
            preprocessor = build_preprocessor()

            # 2) Modelo XGBoost - Simples
            model = XGBClassifier(  # objective=ModelType.MULTI_XGBOOST,
                n_estimators=xgb_parameter.n_estimators,
                learning_rate=xgb_parameter.learning_rate,
                max_depth=xgb_parameter.max_depth)

            # 3) sklearn Pipeline completo: preprocess -> model
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            # 4) Entrenar: el pipeline se encarga de transformar y entrenar
            pipe.fit(x_train, y_train)
            return pipe
        except Exception as e:
            raise TrainingFit_Error(
                f"X shape={x_train.shape}, y shape={y_train.shape}"
            )

    @staticmethod
    def _evaluation_model(x_test: DataFrame, y_test: Series, pipe: Pipeline) -> MetricsDto:
        try:
            y_pre = pipe.predict(x_test)
            return MetricsDto(
                accuracy=round(accuracy_score(y_test, y_pre), 4),
                precision=round(precision_score(y_test, y_pre, average="weighted"), 4),
                recall=round(recall_score(y_test, y_pre, average="weighted"), 4),
                f1_score=round(f1_score(y_test, y_pre, average="weighted"), 4),
            )
        except Exception as e:
            raise TrainingEvaluation_Error(str(e)) from e
