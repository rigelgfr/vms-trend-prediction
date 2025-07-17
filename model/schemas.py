from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PredictionData(BaseModel):
    Date: str
    Visits: int
    Peak_1: str
    Peak_2: str


class ForecastResponse(BaseModel):
    status: str
    forecast: List[PredictionData]
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scalers_loaded: bool
    database_configured: bool


class DataTestResponse(BaseModel):
    status: str
    data_shape: tuple
    columns: List[str]
    sample_data: Optional[dict]
    date_range: dict
