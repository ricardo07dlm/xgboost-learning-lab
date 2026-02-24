from fastapi import APIRouter
from schemas.common_ml.repository_schema import ModelRepositoryResponse
from services.impl.predict_service_impl import PredictServiceImpl
from services.impl.repository_service_impl import RepositoryServiceImpl
from schemas.training_ml.training_schema import TrainRequest, TrainResponse
from schemas.prediction_ml.prediction_schema import PredictResponse, PredictRequest
from src.services.impl.training_service_impl import TrainingServiceImpl

router = APIRouter()


@router.post(path="/api/training",
             response_model_exclude_none=True,
             response_model=TrainResponse,
             summary="Training Model API",
             responses={200: {"description": "Success", },
                        404: {"description": "Not Found.", }, }, )
async def training_model(training_obj: TrainRequest):
    trainingService = TrainingServiceImpl()
    return trainingService.process(training_obj)


@router.post(path="/api/predictproba",
             response_model_exclude_none=True,
             response_model=PredictResponse,
             summary="Predict Model API",
             responses={200: {"description": "Success", },
                        404: {"description": "Not Found.", }, }, )
async def predict_model(predict_obj: PredictRequest):
    predictService = PredictServiceImpl()
    return predictService.process(predict_obj)



@router.get(path="/api/repository",
            response_model_exclude_none=True,
            response_model=list[ModelRepositoryResponse],
            summary="Repository Model API",
            responses={200: {"description": "Success", },
                       404: {"description": "Not Found.", }, }, )
async def predict_model():
    repService = RepositoryServiceImpl()
    return repService.list_all_models()
