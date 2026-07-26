from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def normalise_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_optional_text(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return default if not text or text.casefold() in {"nan", "none", "null"} else text


class ReferenceScoreStore:
    """Immutable validated component-score frame used for lookup and diagnostics."""

    def __init__(self, path: str | Path | None, score_columns: Iterable[str]):
        self.path = Path(path) if path else None
        self.score_columns = [str(column) for column in score_columns]
        self.frame: pd.DataFrame | None = None
        self.index: dict[tuple[str, str], dict[str, Any]] = {}
        self.error: str | None = None
        self.provenance_audit: dict[str, Any] = {"scientific_status": "unknown"}
        self.reload()

    def reload(self) -> None:
        try:
            if not self.path:
                raise RuntimeError("PRECOMPUTED_SCORE_FRAME is not configured.")
            if not self.path.exists():
                raise FileNotFoundError(f"Validated score frame does not exist: {self.path}")
            frame = pd.read_csv(self.path, low_memory=False)
            required = ["compound_key", "target_key", *self.score_columns]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"Validated score frame is missing columns: {missing}")

            # A finalized model frame is immutable. Ignore legacy production-cache
            # rows if they were appended by an older deployment.
            cache_mask = pd.Series(False, index=frame.index)
            if "record_type" in frame.columns:
                cache_mask |= frame["record_type"].astype(str).str.casefold().eq("production_prediction_cache")
            if "final_split" in frame.columns:
                cache_mask |= frame["final_split"].astype(str).str.casefold().eq("production_inference")
            frame = frame.loc[~cache_mask].copy()

            component_split_columns = [c for c in frame.columns if c.startswith("split__")]
            held_out_names = {"test", "holdout", "heldout", "held_out"}
            all_components_held_out = bool(component_split_columns) and all(
                set(frame[column].dropna().astype(str).str.casefold().str.strip().unique()).issubset(held_out_names)
                and frame[column].notna().any()
                for column in component_split_columns
            )
            marked_diagnostic = (
                "split_is_diagnostic" in frame.columns
                and frame["split_is_diagnostic"].astype(str).str.casefold().isin({"true", "1", "yes"}).any()
            )
            diagnostic = bool(all_components_held_out or marked_diagnostic)
            self.provenance_audit = {
                "scientific_status": "diagnostic_only" if diagnostic else "registered_or_unverified",
                "publishable": False if diagnostic else None,
                "component_split_columns": component_split_columns,
                "all_component_rows_held_out": all_components_held_out,
                "warning": (
                    "Component scores were already held out and then re-split. "
                    "Predictions remain available for legacy reproduction, but reported "
                    "test metrics and calibration are not publication-valid."
                    if diagnostic else None
                ),
            }

            index: dict[tuple[str, str], dict[str, Any]] = {}
            for row in frame.to_dict(orient="records"):
                key = (normalise_key(row.get("compound_key")), normalise_key(row.get("target_key")))
                if key[0] and key[1]:
                    index[key] = row
            self.frame = frame
            self.index = index
            self.error = None
        except Exception as exc:
            self.frame = None
            self.index = {}
            self.error = f"{type(exc).__name__}: {exc}"
            self.provenance_audit = {"scientific_status": "unavailable"}

    @property
    def available(self) -> bool:
        return bool(self.index)

    @property
    def row_count(self) -> int:
        return len(self.index)

    def get(self, compound_key: str, target_key: str) -> dict[str, Any] | None:
        return self.index.get((normalise_key(compound_key), normalise_key(target_key)))

    def target_frame(self, target_key: str) -> pd.DataFrame:
        if self.frame is None:
            return pd.DataFrame()
        mask = self.frame["target_key"].astype(str).map(normalise_key).eq(normalise_key(target_key))
        return self.frame.loc[mask].copy()

    def background(self, target_key: str, minimum_target_rows: int = 20, score_columns: Iterable[str] | None = None) -> dict[str, Any]:
        if self.frame is None or self.frame.empty:
            return {
                "scope": "bundle_global",
                "sample_size": 0,
                "medians": {},
            }
        columns = [str(column) for column in (score_columns or self.score_columns)]
        target = self.target_frame(target_key)
        scope = "target_specific" if len(target) >= minimum_target_rows else "global_reference"
        source = target if scope == "target_specific" else self.frame
        medians = {
            column: float(pd.to_numeric(source[column], errors="coerce").median())
            for column in columns if column in source.columns
        }
        return {"scope": scope, "sample_size": int(len(source)), "medians": medians}

    def applicability(self, target_key: str, scores: dict[str, float], minimum_target_rows: int = 20, score_columns: Iterable[str] | None = None) -> dict[str, Any]:
        if self.frame is None or self.frame.empty:
            return {
                "status": "unknown",
                "scope": "unavailable",
                "reason": "Validated score distribution is unavailable.",
                "components": [],
            }
        columns = [str(column) for column in (score_columns or self.score_columns)]
        target = self.target_frame(target_key)
        scope = "target_specific" if len(target) >= minimum_target_rows else "global_reference"
        source = target if scope == "target_specific" else self.frame
        component_rows: list[dict[str, Any]] = []
        outside = 0
        borderline = 0
        for column in columns:
            series = pd.to_numeric(source[column], errors="coerce").dropna()
            value = float(scores[column])
            if series.empty:
                component_rows.append({"component_score": column, "value": value, "status": "unknown"})
                continue
            q01, q05, q95, q99 = [float(series.quantile(q)) for q in (0.01, 0.05, 0.95, 0.99)]
            percentile = float((series <= value).mean())
            status = "in_domain"
            if value < q01 or value > q99:
                status = "outside_reference_range"
                outside += 1
            elif value < q05 or value > q95:
                status = "borderline"
                borderline += 1
            component_rows.append(
                {
                    "component_score": column,
                    "value": value,
                    "percentile": percentile,
                    "q01": q01,
                    "q05": q05,
                    "q95": q95,
                    "q99": q99,
                    "status": status,
                    "saturated": bool(value <= 0.001 or value >= 0.999),
                }
            )
        if outside:
            status = "outside_domain"
            reason = f"{outside} component score(s) fall outside the 1st-99th percentile reference range."
        elif borderline:
            status = "borderline"
            reason = f"{borderline} component score(s) fall outside the 5th-95th percentile reference range."
        else:
            status = "in_domain"
            reason = "All component scores lie within the central validated reference ranges."
        return {
            "status": status,
            "scope": scope,
            "sample_size": int(len(source)),
            "reason": reason,
            "components": component_rows,
        }

    def parity_sample(self, sample_size: int, seed: int = 42) -> pd.DataFrame:
        if self.frame is None or self.frame.empty or sample_size <= 0:
            return pd.DataFrame()
        valid = self.frame.dropna(subset=["compound_key", "target_key", *self.score_columns]).copy()
        if valid.empty:
            return valid
        # Stratify across CYP targets where possible.
        groups = list(valid.groupby("target_key", sort=True))
        selected: list[pd.DataFrame] = []
        per_group = max(1, sample_size // max(1, len(groups)))
        for offset, (_, group) in enumerate(groups):
            selected.append(group.sample(n=min(per_group, len(group)), random_state=seed + offset))
        result = pd.concat(selected, ignore_index=True).drop_duplicates(["compound_key", "target_key"])
        if len(result) < sample_size:
            remaining = valid.merge(
                result[["compound_key", "target_key"]],
                on=["compound_key", "target_key"],
                how="left",
                indicator=True,
            )
            remaining = remaining.loc[remaining["_merge"].eq("left_only"), valid.columns]
            if not remaining.empty:
                result = pd.concat(
                    [result, remaining.sample(n=min(sample_size - len(result), len(remaining)), random_state=seed + 100)],
                    ignore_index=True,
                )
        return result.head(sample_size).copy()

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.path) if self.path else None,
            "row_count": self.row_count,
            "immutable": True,
            "load_error": self.error,
            "provenance_audit": self.provenance_audit,
        }


class PredictionCacheStore:
    """Writable serving cache kept separate from the finalized modeling frame."""

    def __init__(self, path: str | Path | None, score_columns: Iterable[str], write_enabled: bool = True):
        self.path = Path(path) if path else None
        self.score_columns = [str(column) for column in score_columns]
        self.write_enabled = bool(write_enabled)
        self.frame = pd.DataFrame()
        self.index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        self.error: str | None = None
        self.last_write_error: str | None = None
        self.last_write_utc: str | None = None
        self._lock = threading.RLock()
        self.reload()

    @property
    def available(self) -> bool:
        return bool(self.index)

    @property
    def row_count(self) -> int:
        return len(self.index)

    @property
    def writable(self) -> bool:
        if not self.path or not self.write_enabled:
            return False
        target = self.path if self.path.exists() else self.path.parent
        return os.access(target, os.W_OK)

    def reload(self) -> None:
        try:
            if not self.path:
                raise RuntimeError("PREDICTION_CACHE_FRAME is not configured.")
            if not self.path.exists():
                self.frame = pd.DataFrame()
                self.index = {}
                self.error = None
                return
            frame = pd.read_csv(self.path, low_memory=False)
            required = ["compound_key", "target_key", *self.score_columns]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"Prediction cache is missing columns: {missing}")
            index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
            valid_variants = {"primary", "stage3_fallback"}
            for row in frame.to_dict(orient="records"):
                variant = clean_optional_text(row.get("model_variant"), "legacy_unvalidated")
                if variant not in valid_variants:
                    continue
                key = self._row_key(row)
                if key[0] and key[1]:
                    index[key] = row
            self.frame = frame
            self.index = index
            self.error = None
        except Exception as exc:
            self.frame = pd.DataFrame()
            self.index = {}
            self.error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            normalise_key(row.get("compound_key")),
            normalise_key(row.get("target_key")),
            clean_optional_text(row.get("model_variant"), "legacy_unvalidated"),
            clean_optional_text(row.get("model_version"), "unknown_model_version"),
            clean_optional_text(row.get("graph_version"), "unknown_graph_snapshot"),
        )

    def get(
        self,
        compound_key: str,
        target_key: str,
        *,
        model_versions: dict[str, str],
        graph_version: str,
        allow_stale: bool = False,
    ) -> dict[str, Any] | None:
        pair = (normalise_key(compound_key), normalise_key(target_key))
        requested_graph = clean_optional_text(graph_version, "unknown_graph_snapshot")
        candidates = [
            row for key, row in self.index.items()
            if key[:2] == pair and key[2] in model_versions
        ]
        if allow_stale:
            return max(candidates, key=lambda row: str(row.get("created_at_utc", "")), default=None)
        if requested_graph == "unknown_graph_snapshot":
            return None
        compatible = [
            row for row in candidates
            if clean_optional_text(row.get("model_version"), "unknown_model_version")
            == clean_optional_text(model_versions.get(clean_optional_text(row.get("model_variant"), "")), "")
            and clean_optional_text(row.get("graph_version"), "unknown_graph_snapshot") == requested_graph
        ]
        return max(compatible, key=lambda row: str(row.get("created_at_utc", "")), default=None)

    def upsert_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"status": "not_needed", "added_rows": 0, "skipped_existing_rows": 0}
        if not self.write_enabled:
            raise RuntimeError("PREDICTION_PERSIST_NEW_SCORES is disabled.")
        if not self.path:
            raise RuntimeError("PREDICTION_CACHE_FRAME is not configured.")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.reload()
            frame = self.frame.copy()
            existing = set(self.index)
            additions: list[dict[str, Any]] = []
            skipped = 0
            for row in rows:
                prepared = dict(row)
                # Cache rows are inference records, never observational labels.
                # Persist the exclusion flag as text so CSV type inference cannot
                # silently turn it into 1.0/0.0 when nullable columns are present.
                prepared["record_type"] = "production_prediction_cache"
                prepared["exclude_from_training"] = "true"
                prepared["final_split"] = "production_inference"
                prepared["observed_label"] = None
                key = self._row_key(prepared)
                if not key[0] or not key[1]:
                    raise ValueError("Cannot cache a row without canonical compound and target keys.")
                if key in existing:
                    skipped += 1
                    continue
                additions.append(prepared)
                existing.add(key)
            if not additions:
                return {
                    "status": "already_cached",
                    "added_rows": 0,
                    "skipped_existing_rows": skipped,
                    "path": str(self.path),
                }
            columns = list(frame.columns)
            for row in additions:
                for column in row:
                    if column not in columns:
                        columns.append(column)
            additions_frame = pd.DataFrame(additions).reindex(columns=columns)
            updated = pd.concat([frame.reindex(columns=columns), additions_frame], ignore_index=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                updated.to_csv(temporary, index=False)
                os.replace(temporary, self.path)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                self.last_write_error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"Could not persist production prediction cache: {self.last_write_error}") from exc
            self.last_write_error = None
            self.last_write_utc = utc_now()
            self.reload()
            return {
                "status": "written",
                "added_rows": len(additions),
                "skipped_existing_rows": skipped,
                "path": str(self.path),
                "last_write_utc": self.last_write_utc,
            }

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.path) if self.path else None,
            "row_count": self.row_count,
            "write_enabled": self.write_enabled,
            "writable": self.writable,
            "load_error": self.error,
            "last_write_error": self.last_write_error,
            "last_write_utc": self.last_write_utc,
            "cache_identity": "compound+target+model_variant+model_version+graph_version",
        }
