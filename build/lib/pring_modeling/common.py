from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:  # Torch is optional for Stage 1 tabular but required for PyTorch stages.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
except Exception:  # pragma: no cover
    average_precision_score = roc_auc_score = accuracy_score = f1_score = precision_score = recall_score = balanced_accuracy_score = matthews_corrcoef = confusion_matrix = None

STAGE1 = "stage1_neo4j_gds_baselines"
STAGE2 = "stage2_kg_embedding_baselines"
STAGE3 = "stage3_heterogeneous_gnn"
STAGE4 = "stage4_explainability"
STAGE_NAMES = [STAGE1, STAGE2, STAGE3, STAGE4]


@dataclass
class ResolvedInput:
    input_path: Path
    root: Path
    temporary_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    def cleanup(self) -> None:
        if self.temporary_dir is not None:
            self.temporary_dir.cleanup()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    kwargs.setdefault("low_memory", False)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **{k: v for k, v in kwargs.items() if k not in {"low_memory"}})
    return pd.read_csv(path, **kwargs)


def read_triples(path: str | Path) -> pd.DataFrame:
    """Read a triple file with or without a header and return head/relation/tail columns."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if {"head", "relation", "tail"}.issubset(df.columns):
        return df[["head", "relation", "tail"]].astype(str)
    # No useful header detected. Re-read without a header.
    df = pd.read_csv(path, sep="\t", header=None, low_memory=False)
    if df.shape[1] < 3:
        raise ValueError(f"Triple file has fewer than three columns: {path}")
    df = df.iloc[:, :3]
    df.columns = ["head", "relation", "tail"]
    # Drop accidental header rows from manually edited files.
    df = df[~((df["head"].str.lower() == "head") & (df["relation"].str.lower() == "relation"))]
    return df.astype(str).reset_index(drop=True)


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    if required:
        raise KeyError(f"None of these columns were found: {list(candidates)}. Available: {list(df.columns)}")
    return None


def read_pairs(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    if "label" not in df.columns:
        for c in ["y", "target", "is_positive", "positive", "activity_label"]:
            if c in df.columns:
                df = df.rename(columns={c: "label"})
                break
    return df


def coerce_binary_label(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = str(value).strip().lower()
        if text in {"1", "1.0", "positive", "active", "true", "yes"}:
            return 1.0
        if text in {"0", "0.0", "negative", "inactive", "weak", "false", "no"}:
            return 0.0
        if text in {"-1", "-1.0", "unknown", "candidate", "nan", "none", ""}:
            return -1.0
        try:
            return float(text)
        except ValueError:
            return np.nan
    return series.map(convert)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out: dict[str, float | int | None] = {}
    if len(y_true) == 0:
        return {
            "roc_auc": None, "average_precision": None, "accuracy": None,
            "f1": None, "precision": None, "recall": None,
            "balanced_accuracy": None, "mcc": None, "specificity": None,
            "threshold": float(threshold),
        }
    if roc_auc_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = None
    if average_precision_score is not None:
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        out["average_precision"] = None
    y_pred = (y_score >= threshold).astype(int)
    out["threshold"] = float(threshold)
    if accuracy_score is not None:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) == 2:
            out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred)) if balanced_accuracy_score is not None else None
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if matthews_corrcoef is not None else None
            if confusion_matrix is not None:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                out["tn"] = int(tn)
                out["fp"] = int(fp)
                out["fn"] = int(fn)
                out["tp"] = int(tp)
                out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else None
                out["negative_precision"] = float(tn / (tn + fn)) if (tn + fn) else None
                out["positive_ratio"] = float(np.mean(y_true == 1))
                out["negative_ratio"] = float(np.mean(y_true == 0))
                out["youden_j"] = (float(out["recall"]) + float(out["specificity"]) - 1.0) if out.get("recall") is not None and out.get("specificity") is not None else None
        else:
            out["balanced_accuracy"] = None
            out["mcc"] = None
            out["specificity"] = None
    return out


def optimize_binary_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "mcc",
    grid_size: int = 101,
    min_specificity: float = 0.0,
    min_recall: float = 0.0,
) -> tuple[float, dict[str, float | int | bool | str | None]]:
    """Select a probability/rank-score threshold on labelled data.

    The implementation is specificity/recall constraint aware. This is useful
    for CYP450 active/inactive prediction where a fixed threshold or an
    unconstrained F1 threshold can silently predict almost every pair as active.
    If no threshold satisfies the requested constraints, the best unconstrained
    threshold is returned and ``constraint_satisfied`` is set to ``False``.
    """
    from .model_diagnostics import choose_threshold_with_constraints

    return choose_threshold_with_constraints(
        y_true,
        y_score,
        metric,
        binary_metrics,
        min_specificity=float(min_specificity or 0.0),
        min_recall=float(min_recall or 0.0),
        grid_size=grid_size,
    )


def get_device(device: str | None = None):
    """Return a valid torch.device.

    Accepts ``None``/empty/``auto`` and resolves them to CUDA when available,
    otherwise CPU. This keeps SLURM wrapper defaults compatible with legacy and
    improved implementations that previously passed ``--device auto`` directly
    to torch.device().
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed in this environment.")
    if device is None or str(device).strip() == "" or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device).strip())


def tensor_from_embedding_value(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return [float(x) for x in json.loads(text)]
        except Exception:
            text = text.strip("[]")
    text = text.replace(",", " ")
    return [float(x) for x in text.split() if x]


def _extract_zip_if_needed(path: Path) -> ResolvedInput:
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="pring_modeling_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp.name)
        return ResolvedInput(input_path=path, root=Path(tmp.name), temporary_dir=tmp)
    return ResolvedInput(input_path=path, root=path)


def resolve_input(path: str | Path) -> ResolvedInput:
    return _extract_zip_if_needed(Path(path).expanduser().resolve())


def resolve_modeling_dir(path: str | Path) -> ResolvedInput:
    """Resolve a path that may be a full PRING run, graph/ml/modeling folder, standalone stage folder, or zip."""
    resolved = resolve_input(path)
    p = resolved.root
    candidates = [
        p,
        p / "graph" / "ml" / "modeling",
        p / "ml" / "modeling",
        p / "modeling",
    ]
    for c in candidates:
        if c.exists() and any((c / s).exists() for s in STAGE_NAMES):
            resolved.root = c
            return resolved
    # Standalone stage folder: return its parent so resolve_stage_dir can find it.
    if p.name in STAGE_NAMES:
        resolved.root = p.parent
        return resolved
    # Search shallowly for modeling folder or stage folder.
    for c in p.rglob("modeling"):
        if c.is_dir() and any((c / s).exists() for s in STAGE_NAMES):
            resolved.root = c
            return resolved
    for s in STAGE_NAMES:
        hits = list(p.rglob(s))
        if hits:
            resolved.root = hits[0].parent
            return resolved
    resolved.root = p
    return resolved


def resolve_stage_dir(path: str | Path, stage_name: str) -> ResolvedInput:
    resolved = resolve_input(path)
    p = resolved.root
    candidates = [
        p if p.name == stage_name else p / stage_name,
        p / "graph" / "ml" / "modeling" / stage_name,
        p / "ml" / "modeling" / stage_name,
        p / "modeling" / stage_name,
    ]
    # Some current PRING exports put Stage 1 pair files directly in graph/ml, not graph/ml/modeling/stage1.
    if stage_name == STAGE1:
        candidates.extend([p / "graph" / "ml", p / "ml"])
    for c in candidates:
        if c.exists():
            resolved.root = c
            return resolved
    hits = list(p.rglob(stage_name))
    if hits:
        resolved.root = hits[0]
        return resolved
    raise FileNotFoundError(f"Could not resolve {stage_name} below {p}")


def find_existing(*paths: str | Path) -> Path:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    raise FileNotFoundError("None of the expected files exists: " + ", ".join(map(str, paths)))


def parse_node_ref(node_ref: Any) -> dict[str, str]:
    """Parse PRING node refs such as Compound|cid=10002960 or Protein|protein_id=P08684."""
    text = str(node_ref or "").strip()
    out: dict[str, str] = {"node_ref": text}
    if not text:
        return out
    parts = text.split("|")
    if parts:
        out["label"] = parts[0]
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    if ":" in text and len(parts) == 1:
        label, value = text.split(":", 1)
        out["label"] = label
        out.setdefault("id", value)
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_entity_reference(stage_or_modeling_dir: str | Path) -> pd.DataFrame | None:
    p = Path(stage_or_modeling_dir)
    candidates = [
        p / "entities.tsv",
        p / STAGE2 / "entities.tsv",
        p / "node_mapping_reference.csv",
        p / STAGE1 / "node_mapping_reference.csv",
        p / "graph" / "ml" / "modeling" / STAGE2 / "entities.tsv",
        p / "graph" / "ml" / "modeling" / STAGE1 / "node_mapping_reference.csv",
    ]
    for c in candidates:
        if c.exists():
            df = read_table(c)
            if "entity_id" not in df.columns and "node_ref" in df.columns:
                df = df.copy()
                df["entity_id"] = df.get("node_id", df.index).astype(str)
            return df
    return None


def attach_refs_from_entities(preds: pd.DataFrame, entities: pd.DataFrame | None) -> pd.DataFrame:
    if entities is None or preds.empty:
        return preds
    if "entity_id" not in entities.columns or "node_ref" not in entities.columns:
        return preds
    ref_map = entities.set_index(entities["entity_id"].astype(str))["node_ref"].to_dict()
    type_map = entities.set_index(entities["entity_id"].astype(str))["node_type"].to_dict() if "node_type" in entities.columns else {}
    out = preds.copy()
    if "head" in out.columns and "tail" in out.columns:
        head_type = out["head"].astype(str).map(type_map)
        tail_type = out["tail"].astype(str).map(type_map)
        # CYP450 target triples are Compound -> INTERACTS_WITH -> Protein in the attached Stage 2 export.
        compound_from_head = head_type.eq("Compound") | (~tail_type.eq("Compound") & ~head_type.eq("Protein"))
        out["compound_entity_id"] = np.where(compound_from_head, out["head"].astype(str), out["tail"].astype(str))
        out["protein_entity_id"] = np.where(compound_from_head, out["tail"].astype(str), out["head"].astype(str))
        out["compound_node_ref"] = out["compound_entity_id"].astype(str).map(ref_map)
        out["protein_node_ref"] = out["protein_entity_id"].astype(str).map(ref_map)
    return out
