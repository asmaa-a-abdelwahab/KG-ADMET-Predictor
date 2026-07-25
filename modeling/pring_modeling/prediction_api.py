from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .prediction_service import PRINGPredictionService
from .pyg_runtime import initialize_pyg_runtime


class PredictionRequest(BaseModel):
    compounds: list[str] = Field(min_length=1, max_length=25)
    targets: list[str] = Field(min_length=1, max_length=25)


@lru_cache(maxsize=1)
def get_service() -> PRINGPredictionService:
    return PRINGPredictionService()


def _service_status() -> dict[str, Any]:
    try:
        return get_service().status()
    except Exception as exc:
        return {
            "status": "not_ready",
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Import PyG once, in a controlled order, before health checks and prediction
    # requests can import separate PyG submodules concurrently.
    initialize_pyg_runtime()
    service = get_service()
    yield
    service.close()


app = FastAPI(
    title="PRING Hybrid Interaction Predictor",
    version="3.0.0",
    description=(
        "Precomputed-first prediction API. Existing component scores are reused; "
        "missing compound-CYP450 pairs are scored by the deployable Stage 1, R-GCN "
        "and HGT components and persisted to the production score cache."
    ),
    lifespan=lifespan,
)


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health")
def health() -> dict[str, Any]:
    """Diagnostic endpoint; always returns JSON while the API process is alive."""
    return _service_status()


@app.get("/ready")
def ready() -> dict[str, Any]:
    status = _service_status()
    if status.get("status") != "ready":
        raise HTTPException(status_code=503, detail=status)
    return status


@app.get("/model")
def model_info() -> dict[str, Any]:
    return _service_status()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    status = _service_status()
    if status.get("status") != "ready":
        raise HTTPException(status_code=503, detail=status)
    try:
        return get_service().predict_many(request.compounds, request.targets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error_type": type(exc).__name__, "error": str(exc)},
        ) from exc
