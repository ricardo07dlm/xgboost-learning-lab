class RepositoryError(RuntimeError):

    def __init__(self, message: str, *, message_key: str, status_code: int):
        super().__init__(message)
        self.message_key = message_key
        self.status_code = status_code


class ModelNotFound_Error(RepositoryError):
    def __init__(self, details: str | None = None):
        msg = "Modelo no encontrado"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_NOT_FOUND",
            status_code=404
        )


class ModelLoad_Error(RepositoryError):
    def __init__(self, details: str | None = None):
        msg = "Error cargando el modelo"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_LOAD_ERROR",
            status_code=500
        )


class ModelPersistence_Error(RepositoryError):
    def __init__(self, details: str | None = None):
        msg = "Error en persistencia del modelo"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_PERSISTENCE_ERROR",
            status_code=500
        )


class ModelType_Error(RepositoryError):
    def __init__(self, details: str | None = None):
        msg = "Artefacto cargado incompatible"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_TYPE_ERROR",
            status_code=500
        )


class ModelList_Error(RepositoryError):
    def __init__(self, details: str | None = None):
        msg = "Error listando modelos del repositorio"
        super().__init__(
            msg if not details else f"{msg}: {details}",
            message_key="MODEL_LIST_ERROR",
            status_code=500
        )
