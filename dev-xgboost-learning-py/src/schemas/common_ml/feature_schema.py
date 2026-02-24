from pydantic import BaseModel, Field


class Feature(BaseModel):
    edad: int = Field(..., ge=18, description="Edad del cliente")
    ingresos_mensuales: float = Field(..., gt=0, description="Ingresos mensuales del cliente")
    antiguedad_meses: int = Field(..., ge=0, description="Antigüedad del cliente en meses")
    incidentes_previos: int = Field(..., ge=0, description="Número de incidentes previos registrados")
    ratio_deuda_ingresos: float = Field(..., ge=0, le=1, description="Ratio entre deuda e ingresos (valor entre 0 y 1)")
    num_productos: int = Field(..., ge=0, description="Número de productos contratados")
    canal: str = Field(..., description="Canal de contratación o interacción (Web, Oficina, CallCenter, etc.)")
