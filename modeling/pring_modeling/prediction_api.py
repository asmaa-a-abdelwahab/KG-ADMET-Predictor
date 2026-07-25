from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .prediction_service import PRINGPredictionService


class PredictionRequest(BaseModel):
    compounds: list[str] = Field(min_length=1, max_length=25)
    targets: list[str] = Field(min_length=1, max_length=25)


@lru_cache(maxsize=1)
def get_service() -> PRINGPredictionService:
    return PRINGPredictionService()


def _service_status() -> dict[str, Any]:
    """Return readiness details without converting them into an HTTP error."""
    try:
        status = get_service().status()
        status["ready"] = status.get("status") == "ready"
        return status
    except Exception as exc:
        return {
            "status": "not_ready",
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


app = FastAPI(
    title="PRING Production Interaction Predictor",
    version="2.0.1",
    description=(
        "Production API using precomputed component-model scores, a locked calibrated "
        "ensemble, and Neo4j evidence retrieval. Full heterogeneous-graph scoring is an "
        "offline batch/HPC responsibility."
    ),
)


@app.get("/live")
def live() -> dict[str, str]:
    """Liveness probe: 200 means the API process is running."""
    return {"status": "alive"}


@app.get("/health")
def health() -> dict[str, Any]:
    """Diagnostic status: always returns JSON so missing assets are visible."""
    return _service_status()


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness probe: 503 until all production-serving assets are usable."""
    status = _service_status()
    if not status.get("ready"):
        raise HTTPException(status_code=503, detail=status)
    return status


@app.get("/model")
def model_info() -> dict[str, Any]:
    return _service_status()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    status = _service_status()
    if not status.get("ready"):
        raise HTTPException(status_code=503, detail=status)

    try:
        return get_service().predict_many(request.compounds, request.targets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
