class PredictProbaError(RuntimeError):
    """Raised when training pipeline fails"""

    def __init__(self, message: str, *, message_key: str, status_code: int):
        super().__init__(message)
        self.message_key = message_key
        self.status_code = status_code


class PredictInputInvalid_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Entrada inválida para predicción"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_INPUT_INVALID",
            status_code=400
        )


class PredictSchemaMismatch_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Esquema/columnas incompatibles con el modelo"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_SCHEMA_MISMATCH",
            status_code=400
        )


class PredictPreprocess_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Fallo en el preprocesado"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_PREPROCESS_ERROR",
            status_code=400
        )


class PredictNaNInf_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Datos contienen NaN/Inf"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_NAN_INF",
            status_code=400
        )


class PredictProbaNotSupported_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "El modelo no soporta predict_proba"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_PROBA_NOT_SUPPORTED",
            status_code=400
        )


class PredictOutputShape_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Salida de predict_proba con forma inesperada"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_OUTPUT_SHAPE_ERROR",
            status_code=500
        )


class PredictNotFitted_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Modelo no entrenado (NotFitted)"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_NOT_FITTED",
            status_code=500
        )


class PredictInvalidProba_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Probabilidades inválidas (NaN/Inf o fuera de rango)"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_INVALID_PROBA",
            status_code=500
        )


class PredictProbaUnexpected_Error(PredictProbaError):
    def __init__(self, details: str | None = None):
        msg = "Error inesperado en predict_proba"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="PREDICT_PROBA_UNEXPECTED_ERROR",
            status_code=500
        )
