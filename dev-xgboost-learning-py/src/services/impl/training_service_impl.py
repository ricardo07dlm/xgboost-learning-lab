from pandas import DataFrame

from core.exceptions.training_exceptions import TrainingError
from ml.xgb.dto.training_dto import TrainingDto
from core.exceptions.service_error import ServiceError
from services.impl.repository_service_impl import RepositoryServiceImpl
from services.ml_service import TrainingService
from schemas.training_ml.training_schema import TrainRequest, TrainResponse
from core.env.settings_config import load_config
from core.env.environment_config import AppConfig
from ml.xgb.process.training import Training
from datetime import datetime
from schemas.adapters.mappers import feature_training_mapper, target_training_mapper
from schemas.adapters.builders import training_response_build
import logging

logger = logging.getLogger(__name__)


class TrainingServiceImpl(TrainingService):

    def __init__(self):
        self.config_env: AppConfig = load_config().application

    def process(self, request: TrainRequest) -> TrainResponse:
        logger.info("Execute[Training & Validation Phase] :: process")
        try:
            # (1) Transform SCHEMA > Panda / DataFrame - Series - COMPLETO
            x_df: DataFrame = feature_training_mapper(features=request.features)
            y_df = target_training_mapper(target=request.target)

            # (2) Call Training XGBoost
            training_obj = Training()
            model_id = datetime.now().strftime("xgb_model_%Y%m%d_%H%M%S")
            trainingDto: TrainingDto = training_obj.execute(x=x_df, y=y_df, parameter=request.params)

            # (3) Prepare save Directory
            model_rep = RepositoryServiceImpl(config=self.config_env)
            model_rep.save_model(pipe_model=trainingDto.trained_model, model_id=model_id)

            # (4) Build TrainingResponse
            return training_response_build(x_df, trainingDto.metrics_model, model_id)

        except TrainingError as e:
            logger.exception("Training failed in service layer")
            raise ServiceError(
                message=e.message_key,
                status_code=e.status_code,
                details=str(e)
            ) from e
