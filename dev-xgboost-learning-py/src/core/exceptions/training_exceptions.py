class TrainingError(RuntimeError):
    """Raised when training pipeline fails"""

    def __init__(self, message: str, *, message_key: str, status_code: int):
        super().__init__(message)
        self.message_key = message_key
        self.status_code = status_code


class TrainingDataSetEmpty_Error(TrainingError):
    def __init__(self, details: str | None = None):
        message = "Dataset vacío"
        super().__init__(
            message if not details else f"{message}: {details}",
            message_key="DATASET_ERROR_EMPTY",
            status_code=400
        )


class TrainingDataSetClass_Error(TrainingError):
    def __init__(self, details: str | None = None):
        message = "Clases insuficientes en dataset"
        super().__init__(
            message if not details else f"{message}: {details}",
            message_key="DATASET_INSUFFICIENT_CLASSES",
            status_code=400
        )


class TrainingSplitInvalid_Error(TrainingError):
    def __init__(self, details: str | None = None):
        message = "Split inválido"
        super().__init__(
            message if not details else f"{message}: {details}",
            message_key="SPLIT_INVALID",
            status_code=400
        )


class TrainingFit_Error(TrainingError):
    def __init__(self, details: str | None = None):
        message = "Fallo en entrenamiento del modelo"
        super().__init__(
            message if not details else f"{message}: {details}",
            message_key="FIT_FAIL_ERROR",
            status_code=500
        )


class TrainingEvaluation_Error(TrainingError):
    def __init__(self, details: str | None = None):
        message = "Fallo en evaluación del modelo"
        super().__init__(
            message if not details else f"{message}: {details}",
            message_key="EVALUATION_FAILURE",
            status_code=500
        )


class TrainingUnexpected_Error(TrainingError):
    def __init__(self, details: str | None = None):
        msg = "Error inesperado en training"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="TRAINING_UNEXPECTED_ERROR",
            status_code=500
        )