from core.exceptions.predict_proba_exceptions import PredictProbaError
from core.exceptions.service_error import ServiceError
from schemas.prediction_ml.prediction_schema import PredictResponse, PredictRequest
from services.impl.repository_service_impl import RepositoryServiceImpl
from services.ml_service import PredictingService
from core.env.settings_config import load_config
from core.env.environment_config import AppConfig
from schemas.adapters.mappers import feature_prediction_mapper, FEATURE_ORDER
from schemas.adapters.builders import prediction_response_build
import pandas as pd
from ml.xgb.process.predictor import Predictor
import logging

logger = logging.getLogger(__name__)


class PredictServiceImpl(PredictingService):

    def __init__(self, *, config: AppConfig | None = None, repository_service: RepositoryServiceImpl | None = None):
        self.config_env: AppConfig = config or load_config().application
        self.repository_service = repository_service or RepositoryServiceImpl()

    def process(self, predict_obj: PredictRequest) -> PredictResponse:
        logger.info("Execute[PredictProba Phase] :: process | model_id=%s", predict_obj.model_id)

        try:
            # (1) Transform SCHEMA > Panda / DataFrame - Series - COMPLETO
            x_df: pd.DataFrame = feature_prediction_mapper(features=predict_obj.features)
            x_df_predict = x_df[FEATURE_ORDER]  # Delete Client_ID field in DF

            # (2) Load Trained Model in *.pkl format
            model_trained = self.repository_service.load_model(model_id=predict_obj.model_id)

            # (3) Execute predict method in model trained
            proba = Predictor.execute(x_new=x_df_predict, pipeline=model_trained)

            # (4) Build Response
            predict_resp = prediction_response_build(x_df=x_df, proba=proba, pipeline=model_trained, top_n=3)
            return predict_resp

        except PredictProbaError as e:
            logger.exception("Training failed in service layer")
            raise ServiceError(
                message=e.message_key,
                status_code=e.status_code,
                details=str(e)
            ) from e

