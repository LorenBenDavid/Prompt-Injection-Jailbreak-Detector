"""Pydantic v2 request/response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class ShapToken(BaseModel):
    token: str
    importance: float


class LayerScores(BaseModel):
    heuristic_score: float
    heuristic_triggered_rules: list[str]
    heuristic_latency_ms: float
    embedding_score: float | None
    embedding_nearest_attacks: list[dict[str, Any]]
    embedding_latency_ms: float | None
    bert_score: float | None
    bert_shap_tokens: list[ShapToken]
    bert_latency_ms: float | None


class AnalyzeResponse(BaseModel):
    text: str
    is_attack: bool
    risk_level: str
    ensemble_score: float
    short_circuited: bool
    explanation: str
    layers: LayerScores
    total_latency_ms: float


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    total_latency_ms: float


class DatasetRow(BaseModel):
    id: str
    prompt: str
    label: int
    attack_type: str
    attack_subtype: str
    source: str
    severity: int
    language: str
    created_at: str


class DatasetResponse(BaseModel):
    rows: list[DatasetRow]
    total: int
    page: int
    page_size: int
    attack_count: int
    benign_count: int


class ModelInfo(BaseModel):
    name: str
    version: str
    layers: list[str]
    weights: dict[str, float]
    base_model: str
    embedding_model: str


class MetricsResponse(BaseModel):
    embedding: dict[str, float] | None
    bert: dict[str, float] | None


class GalleryItem(BaseModel):
    prompt: str
    label: int
    risk_level: str
    ensemble_score: float
    attack_subtype: str
    source: str


class GalleryResponse(BaseModel):
    items: list[GalleryItem]


class StatsResponse(BaseModel):
    total_analyzed: int
    attack_count: int
    benign_count: int
    risk_distribution: dict[str, int]
    avg_latency_ms: float
    dataset_stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    dataset_loaded: bool
