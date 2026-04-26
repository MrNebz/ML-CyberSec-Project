"""Pydantic request/response schemas for the FastAPI service."""
from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Accept either one feature dict or a batch; micro-batching is handled server-side.
    rows: Union[dict[str, float], list[dict[str, float]]] = Field(
        ...,
        description="Either a single feature dict or a list of feature dicts.",
    )


class PredictionItem(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    variant: str
    n_predictions: int
    predictions: list[PredictionItem]


class ModelInfo(BaseModel):
    key: str
    display_name: str
    variant: str
    loader_type: str
    available: bool


class ModelsResponse(BaseModel):
    available: list[ModelInfo]
    unavailable: list[ModelInfo]
