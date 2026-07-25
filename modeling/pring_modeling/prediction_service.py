from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional outside Docker
    GraphDatabase = None  # type: ignore

from .live_prediction import LiveComponentScorer


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _extract_cid(value: str) -> str:
    text = str(value or "").strip()
    for pattern in (r"\bCID\s*([0-9]+)\b", r"Compound\|cid=([0-9]+)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return text if text.isdigit() else text


def _extract_target(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"Protein\|protein_id=([^|\s]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    parenthesized = re.search(r"\(([^()]+)\)\s*$", text)
    if parenthesized and re.fullmatch(r"[A-Za-z0-9_.:-]+", parenthesized.group(1).strip()):
        return parenthesized.group(1).strip()
    return text.split()[0] if text else text


def _normalise_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResolvedPair:
    compound_input: str
    target_input: str
    compound_key: str
    target_key: str
    compound_element_id: str | None
    protein_element_id: str | None
    compound_name: str
    target_name: str
    cid: str
    protein_id: str


class PrecomputedScoreStore:
    """Thread-safe, append-capable cache backed by finalized_training_frame.csv.

    Existing train/validation/test rows are preserved. Newly inferred rows are
    marked as production_prediction_cache and have no observed label, so they
    must never be included in model fitting or evaluation.
    """

    def __init__(
        self,
        path: str | Path | None,
        score_columns: list[str],
        *,
        write_enabled: bool = True,
        backup_before_first_write: bool = True,
    ):
        self.path = Path(path) if path else None
        self.score_columns = list(score_columns)
        self.write_enabled = bool(write_enabled)
        self.backup_before_first_write = bool(backup_before_first_write)
        self.frame: pd.DataFrame | None = None
        self.index: dict[tuple[str, str], dict[str, Any]] = {}
        self.error: str | None = None
        self.last_write_error: str | None = None
        self.last_write_utc: str | None = None
        self._write_lock = threading.RLock()
        self._backup_created = False
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

    def _load_frame(self) -> pd.DataFrame:
        if not self.path:
            raise RuntimeError("PRECOMPUTED_SCORE_FRAME is not configured.")
        if not self.path.exists():
            raise FileNotFoundError(f"Precomputed score frame does not exist: {self.path}")
        frame = pd.read_csv(self.path, low_memory=False)
        required = ["compound_key", "target_key", *self.score_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Precomputed score frame is missing columns: {missing}")
        return frame

    def _rebuild_index(self, frame: pd.DataFrame) -> None:
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for row in frame.to_dict(orient="records"):
            key = (_normalise_key(row.get("compound_key")), _normalise_key(row.get("target_key")))
            if key[0] and key[1]:
                index[key] = row
        self.frame = frame
        self.index = index

    def reload(self) -> None:
        try:
            frame = self._load_frame()
            self._rebuild_index(frame)
            self.error = None
        except Exception as exc:
            self.frame = None
            self.index = {}
            self.error = f"{type(exc).__name__}: {exc}"

    def get(self, compound_key: str, target_key: str) -> dict[str, Any] | None:
        return self.index.get((_normalise_key(compound_key), _normalise_key(target_key)))

    def _create_backup(self) -> str | None:
        if not self.path or not self.path.exists() or self._backup_created or not self.backup_before_first_write:
            return None
        backup = self.path.with_name(
            f"{self.path.stem}.before_live_cache_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{self.path.suffix}"
        )
        shutil.copy2(self.path, backup)
        self._backup_created = True
        return str(backup)

    def upsert_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"status": "not_needed", "added_rows": 0, "skipped_existing_rows": 0}
        if not self.write_enabled:
            raise RuntimeError("PREDICTION_PERSIST_NEW_SCORES is disabled.")
        if not self.path:
            raise RuntimeError("PRECOMPUTED_SCORE_FRAME is not configured.")

        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            frame = self._load_frame()
            existing = {
                (_normalise_key(row["compound_key"]), _normalise_key(row["target_key"]))
                for row in frame[["compound_key", "target_key"]].to_dict(orient="records")
            }

            additions: list[dict[str, Any]] = []
            skipped = 0
            for row in rows:
                key = (_normalise_key(row.get("compound_key")), _normalise_key(row.get("target_key")))
                if not key[0] or not key[1]:
                    raise ValueError(f"Cannot persist a row without canonical compound/target keys: {row}")
                if key in existing:
                    skipped += 1
                    continue
                additions.append(dict(row))
                existing.add(key)

            if not additions:
                self._rebuild_index(frame)
                return {
                    "status": "already_cached",
                    "added_rows": 0,
                    "skipped_existing_rows": skipped,
                    "path": str(self.path),
                }

            backup_path = self._create_backup()
            all_columns = list(frame.columns)
            for row in additions:
                for column in row:
                    if column not in all_columns:
                        all_columns.append(column)
            additions_frame = pd.DataFrame(additions).reindex(columns=all_columns)
            updated = pd.concat([frame.reindex(columns=all_columns), additions_frame], ignore_index=True)

            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                updated.to_csv(temporary, index=False)
                os.replace(temporary, self.path)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                self.last_write_error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(
                    f"Live scores were generated but could not be written to {self.path}: {self.last_write_error}"
                ) from exc

            self.last_write_error = None
            self.last_write_utc = _utc_now()
            self._rebuild_index(updated)
            return {
                "status": "written",
                "added_rows": len(additions),
                "skipped_existing_rows": skipped,
                "path": str(self.path),
                "backup_path": backup_path,
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
        }


class PRINGPredictionService:
    """Hybrid production predictor with an append-only serving score cache."""

    def __init__(self) -> None:
        self.model_dir = Path(os.getenv("PRODUCTION_MODEL_DIR", "/models/production"))
        model_file = self.model_dir / "production_ensemble.joblib"
        manifest_file = self.model_dir / "manifest.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Production ensemble not found: {model_file}")
        if not manifest_file.exists():
            raise FileNotFoundError(f"Production manifest not found: {manifest_file}")

        self.bundle = joblib.load(model_file)
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.score_columns = list(self.bundle.get("score_columns") or self.manifest.get("score_columns") or [])
        if not self.score_columns:
            raise ValueError("Production bundle does not define score_columns.")
        for key in ("model", "calibrator", "threshold"):
            if key not in self.bundle:
                raise ValueError(f"Production bundle is missing required key: {key}")

        self.score_mode = os.getenv("PREDICTION_SCORE_MODE", "auto").strip().lower()
        if self.score_mode not in {"auto", "precomputed", "live"}:
            raise ValueError("PREDICTION_SCORE_MODE must be auto, precomputed, or live.")

        self.persist_new_scores = _env_flag("PREDICTION_PERSIST_NEW_SCORES", True)
        self.require_cache_write = _env_flag("PREDICTION_REQUIRE_CACHE_WRITE", True)
        self.precomputed = PrecomputedScoreStore(
            os.getenv("PRECOMPUTED_SCORE_FRAME", "/results/production/finalized_training_frame.csv"),
            self.score_columns,
            write_enabled=self.persist_new_scores,
            backup_before_first_write=_env_flag("PREDICTION_CACHE_BACKUP", True),
        )
        self.live = LiveComponentScorer(self.score_columns)

        self.max_pairs = int(
            os.getenv(
                "PREDICTION_MAX_PAIRS_PER_REQUEST",
                os.getenv("PREDICTION_MAX_PAIRS", "9"),
            )
        )
        self.tree_shap_enabled = _env_flag(
            "PREDICTION_TREE_SHAP_ENABLED",
            _env_flag("PREDICTION_ENABLE_TREE_SHAP", False),
        )

        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        if GraphDatabase is not None:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

        self.stage1_importance = self._read_optional_csv("stage1_feature_importance.csv")
        self.component_importance = self._read_optional_csv("component_feature_importance.csv")
        self.per_target_metrics = self._read_optional_csv("per_target_metrics.csv")
        background_file = self.model_dir / "explainability_background.csv"
        self.explainability_background = (
            pd.read_csv(background_file, low_memory=False)
            if background_file.exists()
            else None
        )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def _read_optional_csv(self, name: str) -> list[dict[str, Any]]:
        path = self.model_dir / name
        return pd.read_csv(path, low_memory=False).to_dict(orient="records") if path.exists() else []

    def status(self) -> dict[str, Any]:
        live_status = self.live.status()
        can_serve = bool(self.precomputed.available or live_status.get("ready"))
        hybrid_ready = bool(
            self.precomputed.available
            and live_status.get("ready")
            and (not self.persist_new_scores or self.precomputed.writable)
        )
        return {
            "status": "ready" if can_serve else "not_ready",
            "ready": can_serve,
            "hybrid_ready": hybrid_ready,
            "model": self.manifest,
            "serving_architecture": "precomputed_first_then_live_inference_with_persistent_score_cache",
            "score_mode": self.score_mode,
            "precomputed_score_store": self.precomputed.status(),
            "live_inference": live_status,
            "persist_new_scores": self.persist_new_scores,
            "require_cache_write": self.require_cache_write,
            "max_pairs_per_request": self.max_pairs,
            "neo4j_configured": self.driver is not None,
        }

    def _resolve_pair(self, compound_input: str, target_input: str) -> ResolvedPair:
        cid = _extract_cid(compound_input)
        target = _extract_target(target_input)
        record = None
        if self.driver is not None:
            query = """
            MATCH (c:Compound)
            WHERE toString(c.cid) = $cid
               OR toLower(coalesce(c.preferred_name, c.name, c.CompoundName, c.title, c.label, '')) = toLower($compound_text)
            WITH c LIMIT 1
            MATCH (p:Protein)
            WITH c, p, properties(p) AS pp
            WHERE toLower(coalesce(pp['protein_id'], pp['uniprot_id'], pp['accession'], pp['cyp_symbol'], pp['symbol'], pp['gene_symbol'], pp['name'], pp['label'], '')) = toLower($target)
               OR toLower(coalesce(pp['cyp_symbol'], pp['symbol'], pp['gene_symbol'], pp['name'], pp['label'], '')) = toLower($target_text)
            RETURN elementId(c) AS compound_element_id,
                   elementId(p) AS protein_element_id,
                   toString(c.cid) AS cid,
                   coalesce(c.preferred_name, c.name, c.CompoundName, c.title, c.label, toString(c.cid)) AS compound_name,
                   coalesce(pp['protein_id'], pp['uniprot_id'], pp['accession'], pp['cyp_symbol'], pp['symbol'], pp['gene_symbol'], pp['name']) AS protein_id,
                   coalesce(pp['cyp_symbol'], pp['symbol'], pp['gene_symbol'], pp['name'], pp['protein_id'], pp['uniprot_id'], pp['accession']) AS target_name
            LIMIT 1
            """
            try:
                with self.driver.session(database=self.neo4j_database) as session:
                    record = session.run(
                        query,
                        cid=str(cid),
                        compound_text=str(compound_input),
                        target=str(target),
                        target_text=str(target_input),
                    ).single()
            except Exception:
                record = None

        if record:
            cid_value = str(record["cid"])
            protein_id = str(record["protein_id"])
            return ResolvedPair(
                compound_input=str(compound_input),
                target_input=str(target_input),
                compound_key=f"Compound|cid={cid_value}",
                target_key=f"Protein|protein_id={protein_id}",
                compound_element_id=str(record["compound_element_id"]),
                protein_element_id=str(record["protein_element_id"]),
                compound_name=str(record["compound_name"]),
                target_name=str(record["target_name"]),
                cid=cid_value,
                protein_id=protein_id,
            )

        canonical_cid = re.sub(r"\D", "", str(cid))
        canonical_target = str(target).strip()
        if not canonical_cid or not canonical_target:
            raise ValueError(f"Could not resolve compound-target pair: {compound_input} / {target_input}")
        return ResolvedPair(
            compound_input=str(compound_input),
            target_input=str(target_input),
            compound_key=f"Compound|cid={canonical_cid}",
            target_key=f"Protein|protein_id={canonical_target}",
            compound_element_id=None,
            protein_element_id=None,
            compound_name=str(compound_input),
            target_name=str(target_input),
            cid=canonical_cid,
            protein_id=canonical_target,
        )

    def _node_properties(self, pair: ResolvedPair) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.driver is None:
            raise RuntimeError("Neo4j is required for live Stage 1 inference.")
        if not pair.compound_element_id or not pair.protein_element_id:
            raise RuntimeError(
                "The selected pair was not resolved to Neo4j nodes; live inference is available only for entities already represented in the graph."
            )
        query = """
        MATCH (c:Compound), (p:Protein)
        WHERE elementId(c) = $compound_element_id
          AND elementId(p) = $protein_element_id
        RETURN properties(c) AS compound_props,
               properties(p) AS protein_props
        """
        with self.driver.session(database=self.neo4j_database) as session:
            record = session.run(
                query,
                compound_element_id=pair.compound_element_id,
                protein_element_id=pair.protein_element_id,
            ).single()
        if not record:
            raise RuntimeError("Could not retrieve Neo4j node properties for live Stage 1 inference.")
        return dict(record["compound_props"] or {}), dict(record["protein_props"] or {})

    def _precomputed_scores(self, pair: ResolvedPair) -> tuple[dict[str, float], dict[str, Any]] | None:
        row = self.precomputed.get(pair.compound_key, pair.target_key)
        if row is None:
            return None
        source = str(row.get("score_source") or row.get("record_type") or "precomputed_validated_model_outputs")
        return (
            {column: float(row[column]) for column in self.score_columns},
            {
                "score_source": source,
                "final_split": row.get("final_split"),
                "observed_label": row.get("label"),
                "pair_key": row.get("pair_key"),
                "cache_created_at_utc": row.get("created_at_utc"),
            },
        )

    def _ensemble_probability(self, scores: dict[str, float]) -> tuple[float, float]:
        row = pd.DataFrame([[scores[column] for column in self.score_columns]], columns=self.score_columns)
        raw_probability = float(self.bundle["model"].predict_proba(row)[:, 1][0])
        calibrated = self.bundle["calibrator"].predict(np.asarray([raw_probability], dtype=float))
        calibrated_probability = float(np.asarray(calibrated).reshape(-1)[0])
        return raw_probability, calibrated_probability

    def _local_explanation(self, scores: dict[str, float], probability: float) -> list[dict[str, Any]]:
        medians = self.bundle.get("background_medians", {})
        components = {
            item.get("score_column"): item
            for item in self.manifest.get("production_components", [])
        }
        rows: list[dict[str, Any]] = []
        for column in self.score_columns:
            perturbed = dict(scores)
            perturbed[column] = float(medians.get(column, 0.5))
            _, changed_probability = self._ensemble_probability(perturbed)
            contribution = float(probability - changed_probability)
            rows.append(
                {
                    "component_score": column,
                    "display_name": components.get(column, {}).get("display_name", column),
                    "value": float(scores[column]),
                    "background_median": float(medians.get(column, 0.5)),
                    "probability_change_when_replaced": contribution,
                    "direction": (
                        "supports active"
                        if contribution > 0
                        else "supports inactive"
                        if contribution < 0
                        else "neutral"
                    ),
                }
            )
        denominator = sum(abs(row["probability_change_when_replaced"]) for row in rows) or 1.0
        for row in rows:
            row["local_importance_share"] = abs(row["probability_change_when_replaced"]) / denominator
        return sorted(rows, key=lambda row: abs(row["probability_change_when_replaced"]), reverse=True)

    def _tree_shap_explanation(self, scores: dict[str, float]) -> dict[str, Any]:
        if not self.tree_shap_enabled:
            return {
                "status": "disabled",
                "reason": "TreeSHAP is disabled to keep online prediction memory bounded.",
                "values": [],
            }
        try:
            import shap

            row = pd.DataFrame([[scores[column] for column in self.score_columns]], columns=self.score_columns)
            pipeline = self.bundle["model"]
            imputer = pipeline.named_steps["imputer"]
            classifier = pipeline.named_steps["classifier"]
            row_imputed = imputer.transform(row)
            background_imputed = None
            if self.explainability_background is not None:
                background = self.explainability_background[self.score_columns].head(200)
                background_imputed = imputer.transform(background)
            explainer = shap.TreeExplainer(classifier, data=background_imputed)
            explanation = explainer(row_imputed)
            values = np.asarray(explanation.values)
            base_values = np.asarray(explanation.base_values)
            if values.ndim == 3:
                values = values[0, :, 1]
                base = float(base_values[0, 1] if base_values.ndim > 1 else base_values.reshape(-1)[-1])
            elif values.ndim == 2:
                values = values[0]
                base = float(base_values.reshape(-1)[-1])
            else:
                values = values.reshape(-1)[: len(self.score_columns)]
                base = float(base_values.reshape(-1)[-1])
            components = {
                item.get("score_column"): item
                for item in self.manifest.get("production_components", [])
            }
            output = []
            for column, value, shap_value in zip(self.score_columns, row.iloc[0], values):
                output.append(
                    {
                        "component_score": column,
                        "display_name": components.get(column, {}).get("display_name", column),
                        "value": float(value),
                        "shap_value_raw_ensemble_output": float(shap_value),
                        "direction": (
                            "supports active"
                            if shap_value > 0
                            else "supports inactive"
                            if shap_value < 0
                            else "neutral"
                        ),
                    }
                )
            return {
                "status": "computed",
                "explained_output": "uncalibrated Extra Trees ensemble probability",
                "base_value": base,
                "values": sorted(output, key=lambda item: abs(item["shap_value_raw_ensemble_output"]), reverse=True),
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "values": [],
            }

    def _evidence(self, pair: ResolvedPair) -> dict[str, Any]:
        if self.driver is None:
            return {
                "status": "unavailable",
                "tier": "Tier 3",
                "tier_reason": "Neo4j driver is unavailable.",
            }
        query = """
        MATCH (c:Compound), (p:Protein)
        WITH c, p, properties(p) AS pp
        WHERE toString(c.cid) = $cid
          AND $protein_id IN [
              toString(pp['protein_id']),
              toString(pp['uniprot_id']),
              toString(pp['accession']),
              toString(pp['cyp_symbol']),
              toString(pp['symbol'])
          ]
        OPTIONAL MATCH (c)<-[:ASSERTS_CHEMICAL]-(direct:Interaction)-[:ASSERTS_TARGET]->(p)
        WITH c, p, collect(DISTINCT properties(direct))[0..10] AS direct_interactions
        OPTIONAL MATCH (c)-[sim:SIMILAR_TO]-(analog:Compound)<-[:ASSERTS_CHEMICAL]-(ai:Interaction)-[:ASSERTS_TARGET]->(p)
        WITH c, p, direct_interactions,
             collect(DISTINCT {
                cid: analog.cid,
                name: coalesce(analog.preferred_name, analog.name, analog.title, analog.label),
                similarity: coalesce(sim.similarity, sim.score, sim.tanimoto),
                interaction: properties(ai)
             })[0..10] AS analog_support
        RETURN direct_interactions, analog_support
        """
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                record = session.run(query, cid=pair.cid, protein_id=pair.protein_id).single()
        except Exception as exc:
            return {
                "status": "unavailable",
                "error": str(exc),
                "tier": "Tier 3",
                "tier_reason": "Evidence query failed.",
            }
        if not record:
            return {
                "status": "not_found",
                "tier": "Tier 3",
                "tier_reason": "No graph evidence was returned.",
            }
        direct = [item for item in (record["direct_interactions"] or []) if item]
        analogs = [
            item
            for item in (record["analog_support"] or [])
            if item and item.get("cid") is not None
        ]
        if direct:
            tier, reason = "Tier 1", "A direct PRING interaction assertion exists."
        elif analogs:
            tier, reason = "Tier 2", "Similar compounds have target-linked evidence."
        else:
            tier, reason = "Tier 3", "No direct or analogue support was found; this is a model-only hypothesis."
        return {
            "status": "available",
            "tier": tier,
            "tier_reason": reason,
            "known_direct_interaction": bool(direct),
            "direct_interactions": direct,
            "similar_compound_support": analogs,
        }

    def _target_metrics(self, pair: ResolvedPair) -> dict[str, Any] | None:
        for row in self.per_target_metrics:
            group = str(row.get("group", row.get("target_key", "")))
            if pair.protein_id in group or pair.target_key == group:
                return _json_safe(row)
        return None

    def _build_result(
        self,
        pair: ResolvedPair,
        scores: dict[str, float],
        component_details: dict[str, Any],
        cache_write: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_probability, probability = self._ensemble_probability(scores)
        threshold = float(self.bundle["threshold"])
        predicted_label = int(probability >= threshold)
        disagreement = float(np.std(list(scores.values())))
        margin = float(abs(probability - threshold))
        score_source = str(component_details.get("score_source", "unknown"))
        return _json_safe(
            {
                "pair": asdict(pair),
                "model": {
                    "name": self.bundle.get("model_name"),
                    "version": self.bundle.get("model_version"),
                    "score_source": score_source,
                    "calibration": self.manifest.get("calibration"),
                    "threshold_selection": self.manifest.get("selection_basis"),
                },
                "prediction": {
                    "raw_ensemble_probability": raw_probability,
                    "calibrated_probability": probability,
                    "threshold": threshold,
                    "predicted_label": predicted_label,
                    "predicted_class": (
                        "Active interaction"
                        if predicted_label
                        else "Inactive / unsupported interaction"
                    ),
                },
                "component_scores": scores,
                "component_details": component_details,
                "prediction_cache": cache_write or {"status": "precomputed_hit"},
                "explainability": {
                    "local_component_contributions": self._local_explanation(scores, probability),
                    "tree_shap": self._tree_shap_explanation(scores),
                    "global_component_importance": self.component_importance,
                    "stage1_structural_feature_importance": self.stage1_importance,
                    "uncertainty_metrics": {
                        "decision_margin": margin,
                        "component_disagreement_std": disagreement,
                        "predictive_entropy_bits": _entropy(probability),
                        "confidence_band": (
                            "high"
                            if margin >= 0.25 and disagreement <= 0.20
                            else "moderate"
                            if margin >= 0.10
                            else "low"
                        ),
                    },
                    "calibration_metrics_on_frozen_test": {
                        "brier_score": self.manifest.get("metrics", {}).get("brier_score"),
                        "expected_calibration_error": self.manifest.get("metrics", {}).get("expected_calibration_error"),
                    },
                },
                "evidence": self._evidence(pair),
                "validation": {
                    "global_test_metrics": self.manifest.get("metrics", {}),
                    "target_specific_metrics": self._target_metrics(pair),
                },
                "interpretation": {
                    "statement": (
                        f"The calibrated probability is {probability:.3f}; with threshold "
                        f"{threshold:.3f}, the pair is classified as "
                        f"{'active' if predicted_label else 'inactive/unsupported'}."
                    ),
                    "scope_warning": (
                        "This statistical prediction supports prioritization; it is not proof of mechanism, causality, clinical effect, or safety."
                    ),
                },
            }
        )

    def _cache_rows(
        self,
        live_items: list[tuple[ResolvedPair, dict[str, float], dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pair, scores, details in live_items:
            raw_probability, probability = self._ensemble_probability(scores)
            threshold = float(self.bundle["threshold"])
            rows.append(
                {
                    "pair_key": f"{pair.compound_key}||{pair.target_key}",
                    "compound_key": pair.compound_key,
                    "target_key": pair.target_key,
                    "label": np.nan,
                    **{column: float(scores[column]) for column in self.score_columns},
                    "final_split": "production_inference",
                    "record_type": "production_prediction_cache",
                    "score_source": "live_component_inference_cached",
                    "exclude_from_training": "true",
                    "predicted_label": int(probability >= threshold),
                    "raw_ensemble_probability": raw_probability,
                    "calibrated_probability": probability,
                    "decision_threshold": threshold,
                    "model_name": self.bundle.get("model_name"),
                    "model_version": self.bundle.get("model_version"),
                    "created_at_utc": _utc_now(),
                    "stage1_pair_features_json": json.dumps(
                        details.get("stage1_pair_features", {}),
                        sort_keys=True,
                    ),
                }
            )
        return rows

    def predict_pair(self, compound_input: str, target_input: str) -> dict[str, Any]:
        payload = self.predict_many([compound_input], [target_input])
        if payload["predictions"]:
            return payload["predictions"][0]
        error = payload["errors"][0]["error"] if payload["errors"] else "Prediction failed."
        raise RuntimeError(error)

    def predict_many(self, compounds: list[str], targets: list[str]) -> dict[str, Any]:
        if not compounds or not targets:
            raise ValueError("At least one compound and one target are required.")
        pair_count = len(compounds) * len(targets)
        if pair_count > self.max_pairs:
            raise ValueError(
                f"Request contains {pair_count} pairs; the configured limit is {self.max_pairs}."
            )

        resolved_pairs: list[ResolvedPair] = []
        resolution_errors: list[dict[str, str]] = []
        for compound in compounds:
            for target in targets:
                try:
                    resolved_pairs.append(self._resolve_pair(compound, target))
                except Exception as exc:
                    resolution_errors.append(
                        {
                            "compound": str(compound),
                            "target": str(target),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        score_records: dict[tuple[str, str], tuple[dict[str, float], dict[str, Any]]] = {}
        missing_pairs: list[ResolvedPair] = []
        for pair in resolved_pairs:
            precomputed = None if self.score_mode == "live" else self._precomputed_scores(pair)
            if precomputed is not None:
                score_records[(pair.compound_key, pair.target_key)] = precomputed
            else:
                missing_pairs.append(pair)

        if missing_pairs and self.score_mode == "precomputed":
            for pair in missing_pairs:
                resolution_errors.append(
                    {
                        "compound": pair.compound_input,
                        "target": pair.target_input,
                        "error": "Pair not found in the precomputed score cache and live inference is disabled.",
                    }
                )
            missing_pairs = []

        live_items: list[tuple[ResolvedPair, dict[str, float], dict[str, Any]]] = []
        cache_write: dict[str, Any] | None = None
        if missing_pairs:
            try:
                live_output = self.live.score_many(missing_pairs, self._node_properties)
            except Exception as exc:
                for pair in missing_pairs:
                    resolution_errors.append(
                        {
                            "compound": pair.compound_input,
                            "target": pair.target_input,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            else:
                for pair in missing_pairs:
                    item = live_output[(pair.compound_key, pair.target_key)]
                    scores = item["scores"]
                    details = dict(item.get("details", {}))
                    details["score_source"] = "live_component_inference"
                    score_records[(pair.compound_key, pair.target_key)] = (scores, details)
                    live_items.append((pair, scores, details))

                if live_items and self.persist_new_scores:
                    try:
                        cache_write = self.precomputed.upsert_rows(self._cache_rows(live_items))
                    except Exception as exc:
                        cache_write = {
                            "status": "failed",
                            "added_rows": 0,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        for _, _, details in live_items:
                            details["cache_write"] = cache_write
                        if self.require_cache_write:
                            for pair, _, _ in live_items:
                                score_records.pop((pair.compound_key, pair.target_key), None)
                                resolution_errors.append(
                                    {
                                        "compound": pair.compound_input,
                                        "target": pair.target_input,
                                        "error": (
                                            "Live scores were generated, but persistence to the production "
                                            f"score cache failed: {type(exc).__name__}: {exc}"
                                        ),
                                    }
                                )
                    else:
                        for _, _, details in live_items:
                            details["cache_write"] = cache_write
                elif live_items:
                    cache_write = {"status": "disabled", "added_rows": 0}
                    for _, _, details in live_items:
                        details["cache_write"] = cache_write

        results: list[dict[str, Any]] = []
        for pair in resolved_pairs:
            record = score_records.get((pair.compound_key, pair.target_key))
            if record is None:
                continue
            scores, details = record
            pair_cache = details.get("cache_write") if details.get("score_source") == "live_component_inference" else None
            results.append(self._build_result(pair, scores, details, pair_cache))

        return {
            "model_status": self.status(),
            "predictions": results,
            "errors": resolution_errors,
            "requested_pairs": pair_count,
            "successful_pairs": len(results),
            "precomputed_hits": sum(
                1
                for result in results
                if str(result.get("model", {}).get("score_source", "")).startswith("precomputed")
                or "cache" in str(result.get("model", {}).get("score_source", ""))
            ),
            "live_predictions_generated": len(live_items),
            "cache_write": cache_write,
        }
