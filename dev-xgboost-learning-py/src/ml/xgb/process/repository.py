import logging
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
from core.exceptions.repository_exceptions import (
    RepositoryError,
    ModelPersistence_Error,
    ModelLoad_Error,
    ModelType_Error,
    ModelNotFound_Error,
    ModelList_Error,
)
from schemas.common_ml.repository_schema import ModelRepositoryResponse

logger = logging.getLogger(__name__)


class Repository:

    def __init__(self):
        self.models_cache: dict[str, Pipeline] = {}

    def save(self, model: Pipeline, name_model: str, path_dir: Path):
        try:
            if path_dir.exists() and not path_dir.is_dir():
                raise ModelPersistence_Error(f"Path no es directorio: {path_dir}")

            path_dir.mkdir(parents=True, exist_ok=True)

            path = path_dir / f"{name_model}.pkl"
            joblib.dump(model, path)
            logger.info("Model saved: %s", path)

        except RepositoryError:
            raise
        except OSError as e:
            raise ModelPersistence_Error(str(e)) from e
        except Exception as e:
            raise ModelPersistence_Error(f"Unexpected error: {e}") from e

    def load(self, model_id: str, path_dir: str, *, reload: bool = False) -> Pipeline:
        try:
            if (not reload) and (model_id in self.models_cache):
                return self.models_cache[model_id]

            model_file = Path(path_dir) / f"{model_id}.pkl"
            if not model_file.exists():
                raise ModelNotFound_Error(f"path={model_file}")

            try:
                obj = joblib.load(model_file)
            except Exception as e:
                raise ModelLoad_Error(str(e)) from e

            if not isinstance(obj, Pipeline):
                raise ModelType_Error(f"type={type(obj).__name__}")

            self.models_cache[model_id] = obj
            logger.info("Model loaded: %s", model_file)
            return obj

        except RepositoryError:
            raise
        except OSError as e:
            raise ModelLoad_Error(str(e)) from e
        except Exception as e:
            raise ModelLoad_Error(f"Unexpected load error: {e}") from e

    def list_all(self, path_dir: str) -> list[ModelRepositoryResponse]:
        try:
            models_path = Path(path_dir)

            if not models_path.exists():
                raise ModelList_Error(f"Directorio no existe: {models_path}")
            if not models_path.is_dir():
                raise ModelList_Error(f"Path no es directorio: {models_path}")

            models_list: list[ModelRepositoryResponse] = []

            for file in models_path.glob("*.pkl"):
                stat = file.stat()
                models_list.append(
                    ModelRepositoryResponse(
                        model_id=file.stem,
                        filename=file.name,
                        size_kb=round(stat.st_size / 1024, 2),
                        created_at=datetime.fromtimestamp(stat.st_mtime),  # más consistente en Linux
                    )
                )

            models_list.sort(key=lambda m: m.created_at, reverse=True)
            return models_list

        except RepositoryError:
            raise
        except OSError as e:
            raise ModelList_Error(str(e)) from e
        except Exception as e:
            raise ModelList_Error(f"Unexpected error: {e}") from e