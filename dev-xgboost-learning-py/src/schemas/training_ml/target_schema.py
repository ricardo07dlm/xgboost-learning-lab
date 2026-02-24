from pydantic import BaseModel, Field

from core.enums import RiskLevel


class Target(BaseModel):
    risk_level: RiskLevel = Field(..., description="Nivel de riesgo asociado al registro",example="bajo")
