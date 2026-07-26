"""Centralized, process-safe PyTorch Geometric runtime initialization.

The predictor can receive Docker health checks while a user prediction request is
starting. Importing ``torch_geometric.data`` and ``torch_geometric.loader`` from
separate request threads can expose PyG's package while it is only partially
initialized. This module serializes the import sequence, performs it once during
API startup, and shares the resolved classes with Stage 3 modules.
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PyGRuntime:
    ready: bool
    versions: dict[str, str]
    symbols: dict[str, Any]
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


_LOCK = threading.RLock()
_RUNTIME: PyGRuntime | None = None


def _purge_partially_initialized_pyg_modules() -> list[str]:
    """Remove only incomplete PyG modules before a single-threaded retry.

    A failed or interrupted import can leave submodules with
    ``__spec__._initializing`` set. They must not be reused by a later import.
    Fully initialized modules are left untouched.
    """
    removed: list[str] = []
    for name, module in list(sys.modules.items()):
        if name != "torch_geometric" and not name.startswith("torch_geometric."):
            continue
        spec = getattr(module, "__spec__", None)
        if bool(getattr(spec, "_initializing", False)):
            sys.modules.pop(name, None)
            removed.append(name)
    return removed


def _import_runtime_once() -> PyGRuntime:
    import torch

    # Keep this order aligned with the Docker build-time preflight.
    torch_geometric = importlib.import_module("torch_geometric")
    pyg_lib = importlib.import_module("pyg_lib")
    torch_scatter = importlib.import_module("torch_scatter")
    torch_sparse = importlib.import_module("torch_sparse")

    data_module = importlib.import_module("torch_geometric.data")
    loader_module = importlib.import_module("torch_geometric.loader")
    nn_module = importlib.import_module("torch_geometric.nn")

    symbols = {
        "Data": getattr(data_module, "Data"),
        "HeteroData": getattr(data_module, "HeteroData"),
        "LinkNeighborLoader": getattr(loader_module, "LinkNeighborLoader"),
        "HGTConv": getattr(nn_module, "HGTConv"),
        "RGCNConv": getattr(nn_module, "RGCNConv"),
    }

    # Instantiate a lightweight object to verify that data classes are usable.
    _ = symbols["HeteroData"]()

    versions = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "torch_geometric": str(torch_geometric.__version__),
        "pyg_lib": str(getattr(pyg_lib, "__version__", "installed")),
        "torch_scatter": str(getattr(torch_scatter, "__version__", "installed")),
        "torch_sparse": str(getattr(torch_sparse, "__version__", "installed")),
    }
    return PyGRuntime(ready=True, versions=versions, symbols=symbols)


def initialize_pyg_runtime(*, force: bool = False, retry_partial_import: bool = True) -> PyGRuntime:
    """Initialize PyG exactly once and return a structured diagnostic result."""
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is not None and not force:
            return _RUNTIME

        try:
            _RUNTIME = _import_runtime_once()
            return _RUNTIME
        except Exception as first_exc:
            first_trace = traceback.format_exc()

            # A partially initialized PyG package is usually transient and can
            # result from an interrupted/concurrent import. At API startup there
            # are no prediction threads yet, so it is safe to purge incomplete
            # PyG modules and retry exactly once.
            if retry_partial_import and "partially initialized" in str(first_exc).lower():
                _purge_partially_initialized_pyg_modules()
                try:
                    _RUNTIME = _import_runtime_once()
                    return _RUNTIME
                except Exception as retry_exc:
                    _RUNTIME = PyGRuntime(
                        ready=False,
                        versions={},
                        symbols={},
                        error_type=type(retry_exc).__name__,
                        error=str(retry_exc),
                        traceback=traceback.format_exc(),
                    )
                    return _RUNTIME

            _RUNTIME = PyGRuntime(
                ready=False,
                versions={},
                symbols={},
                error_type=type(first_exc).__name__,
                error=str(first_exc),
                traceback=first_trace,
            )
            return _RUNTIME


def require_pyg_runtime() -> PyGRuntime:
    runtime = initialize_pyg_runtime()
    if not runtime.ready:
        detail = f"{runtime.error_type}: {runtime.error}"
        raise RuntimeError(
            "PyTorch Geometric runtime initialization failed. "
            f"Root cause: {detail}. See the predictor /health response or logs "
            "for the complete traceback."
        )
    return runtime


def get_pyg_symbol(name: str) -> Any:
    runtime = require_pyg_runtime()
    try:
        return runtime.symbols[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"PyG symbol {name!r} was not initialized") from exc


def pyg_runtime_status(*, include_traceback: bool = True) -> dict[str, Any]:
    runtime = initialize_pyg_runtime()
    if runtime.ready:
        return {
            "ready": True,
            **runtime.versions,
            "data_class": runtime.symbols["Data"].__name__,
            "heterodata_class": runtime.symbols["HeteroData"].__name__,
            "link_neighbor_loader": runtime.symbols["LinkNeighborLoader"].__name__,
            "hgt_conv": runtime.symbols["HGTConv"].__name__,
            "rgcn_conv": runtime.symbols["RGCNConv"].__name__,
        }

    out: dict[str, Any] = {
        "ready": False,
        "error_type": runtime.error_type,
        "error": runtime.error,
    }
    if include_traceback:
        out["traceback"] = runtime.traceback
    return out


if __name__ == "__main__":  # pragma: no cover - operational diagnostic
    print(json.dumps(pyg_runtime_status(include_traceback=True), indent=2, default=str))
