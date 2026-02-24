from dataclasses import dataclass


@dataclass
class MetricsDto:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
