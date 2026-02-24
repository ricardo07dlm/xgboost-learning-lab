import logging

import numpy as np
from pandas import DataFrame
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.exceptions.predict_proba_exceptions import (
    PredictProbaError, PredictInputInvalid_Error,
    PredictProbaNotSupported_Error,
    PredictOutputShape_Error, PredictNotFitted_Error, PredictInvalidProba_Error, PredictProbaUnexpected_Error
)

logger = logging.getLogger(__name__)


class Predictor:

    @staticmethod
    def execute(x_new: DataFrame, pipeline: Pipeline):
        logger.info("Execute Predict & Probability Phase")
        try:
            return Predictor._predict_proba_safe(pipe=pipeline, x_df=x_new)
        except PredictProbaError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in Predict & Probability Phase")
            raise PredictProbaUnexpected_Error(str(e)) from e


    @staticmethod
    def _predict_proba_safe(pipe: Pipeline, x_df: DataFrame, expected_n_classes: int | None = None) -> np.ndarray:
        logger.info("Execute _predict_proba_safe()")

        # 1) Verificar fit (sin “probar y fallar”)
        try:
            check_is_fitted(pipe)
        except NotFittedError as e:
            raise PredictNotFitted_Error(str(e)) from e

        # 2) Ejecutar predict_proba
        if not hasattr(pipe, "predict_proba"):
            raise PredictProbaNotSupported_Error("Pipeline/estimador sin predict_proba")

        # Validaciones
        if x_df is None or len(x_df) == 0:
            raise PredictInputInvalid_Error("DataFrame vacío")

        proba = pipe.predict_proba(x_df)

        # 3) Validar forma (binary vs multiclass)
        if not hasattr(proba, "shape") or len(proba.shape) != 2:
            raise PredictOutputShape_Error(f"shape={getattr(proba, 'shape', None)}")

        n_samples, n_cols = proba.shape

        # Si no te pasan expected_n_classes, intenta inferirlo del modelo final
        if expected_n_classes is None:
            model = getattr(pipe, "named_steps", {}).get("model", None)
            classes_ = getattr(model, "classes_", None)
            if classes_ is not None:
                expected_n_classes = len(classes_)

        if expected_n_classes is not None and n_cols != expected_n_classes:
            raise PredictOutputShape_Error(
                f"expected_n_classes={expected_n_classes}, got={n_cols}, proba_shape={proba.shape}"
            )

        # 4) Validar valores de probabilidad
        if not np.isfinite(proba).all():
            raise PredictInvalidProba_Error("Se detectó NaN/Inf en proba")

        if (proba < -1e-9).any() or (proba > 1 + 1e-9).any():
            raise PredictInvalidProba_Error("Se detectaron valores fuera de [0,1]")

        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise PredictInvalidProba_Error(f"Suma por fila != 1 (min={row_sums.min():.6f}, max={row_sums.max():.6f})")

        return proba
