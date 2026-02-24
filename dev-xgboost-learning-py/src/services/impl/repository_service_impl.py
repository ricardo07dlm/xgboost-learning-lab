from sklearn.pipeline import Pipeline

from core.exceptions.service_error import ServiceError
from src.core.exceptions.repository_exceptions import RepositoryError
from schemas.common_ml.repository_schema import ModelRepositoryResponse
from src.services.ml_service import RepositoryService
from core.env.settings_config import load_config
from core.env.environment_config import AppConfig
from pathlib import Path
import logging
from src.ml.xgb.process.repository import Repository

logger = logging.getLogger(__name__)


class RepositoryServiceImpl(RepositoryService):

    def __init__(self, *, config: AppConfig | None = None, repository: Repository | None = None):
        # Allow dependency injection
        self.config_env: AppConfig = config or load_config().application
        self.repository: Repository = repository or Repository()
        self.models_dir: Path = Path(self.config_env.url_vector_db)

    def save_model(self, pipe_model: Pipeline, model_id: str) -> None:
        logger.info("Execute[Repository Phase] :: save_model | model_id=%s | dir=%s", model_id, self.models_dir)
        try:
            self.repository.save(model=pipe_model, name_model=model_id, path_dir=self.models_dir)
        except RepositoryError as e:
            logger.exception("Save Model failed in service layer | model_id=%s", model_id)
            raise ServiceError(
                message=e.message_key,
                status_code=e.status_code,
                details=str(e)
            ) from e

    def load_model(self, model_id: str) -> Pipeline:
        logger.info("Execute[Repository Phase] :: load_model | model_id=%s | dir=%s", model_id, self.models_dir)
        try:
            return self.repository.load(model_id=model_id, path_dir=self.models_dir)
        except RepositoryError as e:
            logger.exception("Load Model failed in service layer | model_id=%s", model_id)
            raise ServiceError(
                message=e.message_key,
                status_code=e.status_code,
                details=str(e)
            ) from e

    def list_all_models(self) -> list[ModelRepositoryResponse]:
        logger.info("Execute[Repository Phase] :: list_all_models | dir=%s", self.models_dir)
        try:
            return self.repository.list_all(path_dir=self.models_dir)
        except RepositoryError as e:
            logger.exception("List ALL Model directory failed in service layer")
            raise ServiceError(
                message=e.message_key,
                status_code=e.status_code,
                details=str(e)
            ) from e
