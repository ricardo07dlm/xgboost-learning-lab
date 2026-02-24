from core.enums import TrainingStatus, RiskClass, RiskLevel
from ml.xgb.dto.metrics_dto import MetricsDto
from schemas.prediction_ml.prediction_schema import PredictResponse, TopClass, PredictResult
import numpy as np
from pandas import DataFrame
from schemas.training_ml.training_schema import TrainResponse, ModelInfoResponse, TrainingSummaryResponse, \
    MetricsResponse

RISK_FROM_CLASS = {
    RiskClass.LOW: RiskLevel.LOW,
    RiskClass.MEDIUM: RiskLevel.MEDIUM,
    RiskClass.HIGH: RiskLevel.HIGH,
}


def prediction_response_build(x_df, proba, pipeline, top_n: int = 3) -> PredictResponse:
    # Obtener clases del modelo
    classes = pipeline.classes_
    results = []

    # Iterar sobre cada muestra
    for i, probs in enumerate(proba):
        probs_array = np.array(probs, dtype=float)
        # ordenar índices por probabilidad descendente - Convierte en índices ordenados:
        sorted_idx = probs_array.argsort()[::-1]
        # clase con mayor probabilidad
        best_idx = int(sorted_idx[0])

        risk_ranking = [
            TopClass(
                risk=RISK_FROM_CLASS[RiskClass(int(classes[idx]))],
                score=f"{round(float(probs_array[idx]) * 100, 2)}%"
            )
            for idx in sorted_idx[1:top_n]
        ]

        # datos del cliente a partir del input (features[i])
        client_id = x_df.iloc[i]["client_id"]
        age = x_df.iloc[i]["edad"]

        results.append(
            PredictResult(
                client_id=str(client_id),
                age=int(age),
                risk=RISK_FROM_CLASS[RiskClass(int(classes[best_idx]))],
                score=f"{round(float(probs_array[best_idx]) * 100, 2)}%",
                risk_ranking=risk_ranking
            )
        )
    return PredictResponse(predictions=results)


def training_response_build(x_df: DataFrame, metrics_dto: MetricsDto, model_id: str) -> TrainResponse:
    metricObj = MetricsResponse(
        accuracy=f"{metrics_dto.accuracy * 100:.2f}%",
        precision=f"{metrics_dto.precision * 100:.2f}%",
        recall=f"{metrics_dto.recall * 100:.2f}%",
        f1_score=f"{metrics_dto.f1_score * 100:.2f}%",
    )
    modelInfoObj = ModelInfoResponse(
        id=model_id,
        lifecycle=TrainingStatus.TRAINED
    )
    trainingSummaryObj = TrainingSummaryResponse(
        samples_used=len(x_df),
        feature_dimension=x_df.shape[1]
    )
    trainingResp = TrainResponse(
        model=modelInfoObj,
        performance=metricObj,
        training_summary=trainingSummaryObj,
        features=list(x_df.columns),
    )
    return trainingResp

