"""FastAPI application — 8 endpoints for the prompt injection detector."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.analyzer import (
    analyze,
    analyze_batch,
    load_dataset,
    load_gallery,
    load_metrics,
    load_models,
)
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    DatasetResponse,
    GalleryResponse,
    HealthResponse,
    LayerScores,
    MetricsResponse,
    ModelInfo,
    ShapToken,
    StatsResponse,
)

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)

# In-memory stats accumulator
_stats: dict = {
    "total_analyzed": 0,
    "attack_count": 0,
    "benign_count": 0,
    "risk_distribution": {"SAFE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
    "latencies": [],
}

_models_loaded = False
_gallery_cache: list[dict] | None = None


async def _precompute_gallery() -> None:
    global _gallery_cache
    try:
        _gallery_cache = await asyncio.to_thread(load_gallery, 20)
        log.info("Gallery pre-computed (%d items).", len(_gallery_cache))
    except Exception as exc:
        log.warning("Gallery pre-computation failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    try:
        load_models()
        _models_loaded = True
        import torch
        torch.set_num_threads(2)
        log.info("Models loaded successfully.")
    except Exception as exc:
        log.error("Startup error: %s", exc)
    yield


app = FastAPI(
    title="Prompt Injection Detector",
    description="3-layer ML ensemble for detecting prompt injection and jailbreak attacks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensemble_result_to_response(text: str, result) -> AnalyzeResponse:
    h = result.heuristic
    e = result.embedding
    b = result.bert

    layers = LayerScores(
        heuristic_score=h.score,
        heuristic_triggered_rules=h.triggered_rules,
        heuristic_latency_ms=h.latency_ms,
        embedding_score=e.score if e else None,
        embedding_nearest_attacks=e.nearest_attacks if e else [],
        embedding_latency_ms=e.latency_ms if e else None,
        bert_score=b.score if b else None,
        bert_shap_tokens=[ShapToken(token=t["token"], importance=t["importance"]) for t in (b.shap_tokens[:10] if b else [])],
        bert_latency_ms=b.latency_ms if b else None,
    )
    return AnalyzeResponse(
        text=text[:500],
        is_attack=result.is_attack,
        risk_level=result.risk_level.value,
        ensemble_score=round(result.ensemble_score, 4),
        short_circuited=result.short_circuited,
        explanation=result.explanation,
        layers=layers,
        total_latency_ms=round(result.latency_ms, 2),
    )


def _update_stats(result: AnalyzeResponse) -> None:
    _stats["total_analyzed"] += 1
    if result.is_attack:
        _stats["attack_count"] += 1
    else:
        _stats["benign_count"] += 1
    _stats["risk_distribution"][result.risk_level] = (
        _stats["risk_distribution"].get(result.risk_level, 0) + 1
    )
    _stats["latencies"].append(result.total_latency_ms)
    if len(_stats["latencies"]) > 1000:
        _stats["latencies"] = _stats["latencies"][-1000:]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = await asyncio.to_thread(analyze, request.text)
    response = _ensemble_result_to_response(request.text, result)
    _update_stats(response)
    return response


@app.post("/api/analyze/batch", response_model=BatchAnalyzeResponse)
async def api_analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    t0 = time.perf_counter()
    results = await asyncio.to_thread(analyze_batch, request.texts)
    responses = [_ensemble_result_to_response(t, r) for t, r in zip(request.texts, results)]
    for r in responses:
        _update_stats(r)
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchAnalyzeResponse(results=responses, total_latency_ms=round(total_ms, 2))


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats() -> StatsResponse:
    import pandas as pd

    def _load_dataset_stats() -> dict:
        stats: dict = {}
        for split in ("train", "val", "test"):
            p = ROOT / "data" / "final" / f"{split}.csv"
            if p.exists():
                df = pd.read_csv(p)
                stats[split] = {
                    "total": len(df),
                    "attacks": int((df["label"] == 1).sum()),
                    "benign": int((df["label"] == 0).sum()),
                }
        return stats

    dataset_stats = await asyncio.to_thread(_load_dataset_stats)

    latencies = _stats["latencies"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return StatsResponse(
        total_analyzed=_stats["total_analyzed"],
        attack_count=_stats["attack_count"],
        benign_count=_stats["benign_count"],
        risk_distribution=_stats["risk_distribution"],
        avg_latency_ms=round(avg_latency, 2),
        dataset_stats=dataset_stats,
    )


@app.get("/api/dataset", response_model=DatasetResponse)
async def api_dataset(
    split: str = Query(default="test", pattern="^(train|val|test)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    label: int | None = Query(default=None),
    attack_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> DatasetResponse:
    data = load_dataset(split=split, page=page, page_size=page_size,
                        label=label, attack_type=attack_type, source=source)
    return DatasetResponse(**data)


@app.get("/api/model/info", response_model=ModelInfo)
async def api_model_info() -> ModelInfo:
    return ModelInfo(
        name="PromptInjectionDetector",
        version="1.0.0",
        layers=["heuristic", "embedding", "bert"],
        weights={"heuristic": 0.20, "embedding": 0.30, "bert": 0.50},
        base_model="distilbert-base-uncased",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def api_metrics() -> MetricsResponse:
    m = load_metrics()
    return MetricsResponse(
        embedding=m.get("embedding"),
        bert=m.get("bert"),
    )


@app.get("/api/gallery", response_model=GalleryResponse)
async def api_gallery(n: int = Query(default=20, ge=1, le=100)) -> GalleryResponse:
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    items = (_gallery_cache or [])[:n]
    return GalleryResponse(items=items)


@app.get("/api/health", response_model=HealthResponse)
async def api_health() -> HealthResponse:
    dataset_loaded = (ROOT / "data" / "final" / "test.csv").exists()
    return HealthResponse(
        status="ok",
        models_loaded=_models_loaded,
        dataset_loaded=dataset_loaded,
    )
