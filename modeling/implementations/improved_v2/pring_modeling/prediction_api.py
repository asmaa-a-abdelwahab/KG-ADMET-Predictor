from __future__ import annotations

import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .prediction_service import PRINGPredictionService
from .pyg_runtime import initialize_pyg_runtime

logger = logging.getLogger(__name__)


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


def require_api_key(
    supplied_key: str | None = Header(default=None, alias="X-PRING-API-Key"),
) -> None:
    """Require a shared API key when PREDICTION_API_KEY is configured."""
    expected_key = os.getenv("PREDICTION_API_KEY", "")
    if expected_key and (
        supplied_key is None
        or not hmac.compare_digest(supplied_key.encode("utf-8"), expected_key.encode("utf-8"))
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing prediction API key.")


def _internal_error(exc: Exception) -> HTTPException:
    error_id = uuid.uuid4().hex
    logger.exception("Prediction API failure [%s]", error_id)
    detail: dict[str, Any] = {
        "error": "Prediction processing failed.",
        "error_id": error_id,
    }
    if os.getenv("PREDICTION_DEBUG_ERRORS", "").strip().lower() in {"1", "true", "yes"}:
        detail.update({"error_type": type(exc).__name__, "debug_error": str(exc)})
    return HTTPException(status_code=500, detail=detail)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_pyg_runtime()
    service = get_service()
    yield
    service.close()


app = FastAPI(
    title="PRING Validated Hybrid Interaction Predictor",
    version="4.0.0",
    description=(
        "Validated-reference-first prediction API with a separate production cache, "
        "parity-guarded Stage 1/R-GCN/HGT live inference, calibrated ensemble scoring, "
        "applicability-domain diagnostics, and Neo4j evidence reconstruction."
    ),
    lifespan=lifespan,
)


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health", dependencies=[Depends(require_api_key)])
def health() -> dict[str, Any]:
    return _service_status()


@app.get("/ready")
def ready() -> dict[str, Any]:
    status = _service_status()
    if status.get("status") != "ready":
        raise HTTPException(status_code=503, detail="Prediction service is not ready.")
    return {"status": "ready", "ready": True}


@app.get("/model", dependencies=[Depends(require_api_key)])
def model_info() -> dict[str, Any]:
    return _service_status()


@app.post("/validate-live-parity", dependencies=[Depends(require_api_key)])
def validate_live_parity(
    force: bool = Query(default=False, description="Re-run parity validation even if a result is cached."),
) -> dict[str, Any]:
    try:
        result = get_service().validate_live_parity(force=force)
    except Exception as exc:
        raise _internal_error(exc) from exc
    if result.get("status") != "passed":
        raise HTTPException(status_code=409, detail=result)
    return result


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest) -> dict[str, Any]:
    status = _service_status()
    if status.get("status") != "ready":
        raise HTTPException(status_code=503, detail=status)
    try:
        return get_service().predict_many(request.compounds, request.targets)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _internal_error(exc) from exc
