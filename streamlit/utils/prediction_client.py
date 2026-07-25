from __future__ import annotations

import os
from typing import Any

import requests

PREDICTION_API_URL = os.getenv("PREDICTION_API_URL", "http://predictor:8000").rstrip("/")
CONNECT_TIMEOUT = int(os.getenv("PREDICTION_CONNECT_TIMEOUT_SECONDS", "10"))
READ_TIMEOUT = int(os.getenv("PREDICTION_TIMEOUT_SECONDS", "1800"))


class PredictionAPIError(RuntimeError):
    pass


def _response_detail(response: requests.Response) -> Any:
    try:
        body = response.json()
        return body.get("detail", body)
    except Exception:
        return response.text


def get_prediction_status() -> dict[str, Any]:
    try:
        response = requests.get(
            f"{PREDICTION_API_URL}/health",
            timeout=(CONNECT_TIMEOUT, 30),
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {
            "status": "unavailable",
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "api_url": PREDICTION_API_URL,
        }


def predict_interactions(compounds: list[str], targets: list[str]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{PREDICTION_API_URL}/predict",
            json={"compounds": compounds, "targets": targets},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            headers={"Connection": "close"},
        )
    except requests.ConnectionError as exc:
        raise PredictionAPIError(
            "The predictor API connection was interrupted. The service may have restarted or exceeded "
            f"its memory allocation. URL={PREDICTION_API_URL}. Root cause: {type(exc).__name__}: {exc}"
        ) from exc
    except requests.Timeout as exc:
        raise PredictionAPIError(
            f"Prediction exceeded the {READ_TIMEOUT}-second timeout. New pairs require live Stage 1, "
            "R-GCN and HGT inference, while cached pairs should return quickly. "
            f"Root cause: {type(exc).__name__}: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise PredictionAPIError(
            f"Prediction request failed. URL={PREDICTION_API_URL}. "
            f"Root cause: {type(exc).__name__}: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise PredictionAPIError(
            f"Prediction service error ({response.status_code}): {_response_detail(response)}"
        )
    return response.json()
