from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore

from .live_prediction import LiveComponentScorer
from .prediction_store import (
    PredictionCacheStore,
    ReferenceScoreStore,
    clean_optional_text,
    utc_now,
)


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


def _component_mapping(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("score_column")): dict(item)
        for item in manifest.get("production_components", [])
        if item.get("score_column")
    }


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


class PRINGPredictionService:
    """Validated-cache-first predictor with guarded live inference.

    Lookup order:
      1. Immutable finalized modeling frame.
      2. Separate production prediction cache.
      3. Live Stage 1/R-GCN/HGT inference after a parity validation.
    """

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

        self.fallback_bundle = None
        self.fallback_manifest: dict[str, Any] = {}
        fallback_file = Path(os.getenv(
            "STAGE3_FALLBACK_MODEL_FILE",
            str(self.model_dir / "production_stage3_fallback.joblib"),
        ))
        fallback_manifest_file = Path(os.getenv(
            "STAGE3_FALLBACK_MANIFEST_FILE",
            str(self.model_dir / "stage3_fallback_manifest.json"),
        ))
        if fallback_file.exists():
            candidate = joblib.load(fallback_file)
            for key in ("model", "calibrator", "threshold", "score_columns"):
                if key not in candidate:
                    raise ValueError(f"Stage 3 fallback bundle is missing required key: {key}")
            self.fallback_bundle = candidate
        if fallback_manifest_file.exists():
            self.fallback_manifest = json.loads(fallback_manifest_file.read_text(encoding="utf-8"))

        self.component_metadata = _component_mapping(self.manifest)
        self.score_mode = os.getenv("PREDICTION_SCORE_MODE", "auto").strip().lower()
        if self.score_mode not in {"auto", "precomputed", "live"}:
            raise ValueError("PREDICTION_SCORE_MODE must be auto, precomputed, or live.")

        self.persist_new_scores = _env_flag("PREDICTION_PERSIST_NEW_SCORES", True)
        self.require_cache_write = _env_flag("PREDICTION_REQUIRE_CACHE_WRITE", True)
        self.reference = ReferenceScoreStore(
            os.getenv("PRECOMPUTED_SCORE_FRAME", "/results/production/finalized_training_frame.csv"),
            self.score_columns,
        )
        self.cache = PredictionCacheStore(
            os.getenv("PREDICTION_CACHE_FRAME", "/results/production/production_prediction_cache.csv"),
            self.score_columns,
            write_enabled=self.persist_new_scores,
        )
        self.live = LiveComponentScorer(self.score_columns)

        self.max_pairs = int(os.getenv("PREDICTION_MAX_PAIRS_PER_REQUEST", os.getenv("PREDICTION_MAX_PAIRS", "9")))
        self.tree_shap_enabled = _env_flag("PREDICTION_TREE_SHAP_ENABLED", _env_flag("PREDICTION_ENABLE_TREE_SHAP", False))
        self.minimum_target_background_rows = int(os.getenv("PREDICTION_TARGET_BACKGROUND_MIN_ROWS", "20"))
        self.local_importance_min_effect = float(os.getenv("PREDICTION_LOCAL_IMPORTANCE_MIN_EFFECT", "0.01"))
        self.graph_version = os.getenv("PRING_GRAPH_VERSION", os.getenv("PRING_RUN_DIR", "unknown_graph_snapshot"))

        self.parity_required = _env_flag("PREDICTION_PARITY_REQUIRED", True)
        self.allow_stage3_fallback = _env_flag(
            "PREDICTION_ALLOW_STAGE3_FALLBACK",
            _env_flag("PREDICTION_ALLOW_COMPONENT_FALLBACK", True),
        )
        self.parity_sample_size = int(os.getenv("PREDICTION_PARITY_SAMPLE_SIZE", "5"))
        self.parity_mae_max = float(os.getenv("PREDICTION_PARITY_MAE_MAX", "0.05"))
        self.parity_max_abs_error = float(os.getenv("PREDICTION_PARITY_MAX_ABS_ERROR", "0.15"))
        self.parity_spearman_min = float(os.getenv("PREDICTION_PARITY_SPEARMAN_MIN", "0.90"))
        self.parity_decision_agreement_min = float(os.getenv("PREDICTION_PARITY_DECISION_AGREEMENT_MIN", "0.95"))
        self._parity_lock = threading.RLock()
        self._parity_status: dict[str, Any] = {
            "status": "not_run",
            "required": self.parity_required,
            "sample_size_requested": self.parity_sample_size,
        }

        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        if GraphDatabase is not None:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

        self.stage1_importance = self._read_optional_csv("stage1_feature_importance.csv")
        self.component_importance = self._read_optional_csv("component_feature_importance.csv")
        self.per_target_metrics = self._read_optional_csv("per_target_metrics.csv")
        background_file = self.model_dir / "explainability_background.csv"
        self.explainability_background = pd.read_csv(background_file, low_memory=False) if background_file.exists() else None

    def _variant_context(self, variant: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        if variant == "stage3_fallback":
            if self.fallback_bundle is None:
                raise RuntimeError("Stage 3 fallback bundle is unavailable.")
            columns = [str(column) for column in self.fallback_bundle["score_columns"]]
            return self.fallback_bundle, self.fallback_manifest, columns
        return self.bundle, self.manifest, list(self.score_columns)

    def _variant_available(self, variant: str) -> bool:
        return variant == "primary" or (variant == "stage3_fallback" and self.fallback_bundle is not None)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def _read_optional_csv(self, name: str) -> list[dict[str, Any]]:
        path = self.model_dir / name
        return pd.read_csv(path, low_memory=False).to_dict(orient="records") if path.exists() else []

    def status(self) -> dict[str, Any]:
        live_status = self.live.status()
        can_serve = bool(
            self.reference.available
            or self.cache.available
            or live_status.get("primary_ready")
            or (self.allow_stage3_fallback and live_status.get("stage3_fallback_ready") and self.fallback_bundle is not None)
        )
        hybrid_ready = bool(
            (live_status.get("primary_ready") or live_status.get("stage3_fallback_ready"))
            and (not self.persist_new_scores or self.cache.writable)
            and (not self.parity_required or self._parity_status.get("status") in {"not_run", "passed", "passed_with_stage3_fallback"})
        )
        return {
            "status": "ready" if can_serve else "not_ready",
            "ready": can_serve,
            "hybrid_ready": hybrid_ready,
            "model": self.manifest,
            "serving_architecture": "validated_reference_then_separate_prediction_cache_then_parity_guarded_live_inference",
            "score_mode": self.score_mode,
            "validated_reference_store": self.reference.status(),
            "production_prediction_cache": self.cache.status(),
            "live_inference": live_status,
            "live_inference_parity": self._parity_status,
            "live_model_variants": {
                "primary": {
                    "available": True,
                    "model_name": self.bundle.get("model_name"),
                    "model_version": self.bundle.get("model_version"),
                    "score_columns": self.score_columns,
                },
                "stage3_fallback": {
                    "available": self.fallback_bundle is not None,
                    "model_name": self.fallback_bundle.get("model_name") if self.fallback_bundle else None,
                    "model_version": self.fallback_bundle.get("model_version") if self.fallback_bundle else None,
                    "score_columns": self.fallback_bundle.get("score_columns") if self.fallback_bundle else [],
                },
            },
            "persist_new_scores": self.persist_new_scores,
            "allow_stage3_fallback": self.allow_stage3_fallback,
            "require_cache_write": self.require_cache_write,
            "max_pairs_per_request": self.max_pairs,
            "neo4j_configured": self.driver is not None,
            "graph_version": self.graph_version,
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
            raise RuntimeError("Live inference is available only for entities already represented in Neo4j.")
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

    def _scores_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        return {column: float(row[column]) for column in self.score_columns}

    def _lookup_scores(self, pair: ResolvedPair) -> tuple[dict[str, float], dict[str, Any]] | None:
        reference_row = self.reference.get(pair.compound_key, pair.target_key)
        if reference_row is not None:
            return self._scores_from_row(reference_row), {
                "score_source": "validated_precomputed_record",
                "model_variant": "primary",
                "final_split": reference_row.get("final_split"),
                "observed_label": reference_row.get("label"),
                "pair_key": reference_row.get("pair_key"),
            }
        cache_row = self.cache.get(pair.compound_key, pair.target_key)
        if cache_row is not None:
            variant = clean_optional_text(cache_row.get("model_variant"), "legacy_unvalidated")
            return self._scores_from_row(cache_row), {
                "score_source": "production_prediction_cache",
                "model_variant": variant,
                "final_split": cache_row.get("final_split"),
                "observed_label": None,
                "pair_key": cache_row.get("pair_key"),
                "cache_created_at_utc": cache_row.get("created_at_utc"),
                "model_version_at_cache": cache_row.get("model_version"),
                "graph_version_at_cache": cache_row.get("graph_version"),
            }
        return None

    def _ensemble_probability(self, scores: dict[str, float], variant: str = "primary") -> tuple[float, float]:
        bundle, _, columns = self._variant_context(variant)
        row = pd.DataFrame([[scores[column] for column in columns]], columns=columns)
        raw_probability = float(bundle["model"].predict_proba(row)[:, 1][0])
        calibrated = bundle["calibrator"].predict(np.asarray([raw_probability], dtype=float))
        return raw_probability, float(np.asarray(calibrated).reshape(-1)[0])

    def _background_for_pair(self, pair: ResolvedPair, variant: str) -> dict[str, Any]:
        bundle, _, columns = self._variant_context(variant)
        background = self.reference.background(
            pair.target_key, self.minimum_target_background_rows, score_columns=columns
        )
        if not background.get("medians"):
            background = {
                "scope": "bundle_global",
                "sample_size": 0,
                "medians": dict(bundle.get("background_medians", {})),
            }
        return background

    def _local_explanation(
        self, pair: ResolvedPair, scores: dict[str, float], probability: float, variant: str
    ) -> dict[str, Any]:
        bundle, _, columns = self._variant_context(variant)
        background = self._background_for_pair(pair, variant)
        medians = background["medians"]
        rows: list[dict[str, Any]] = []
        for column in columns:
            perturbed = dict(scores)
            perturbed[column] = float(
                medians.get(column, bundle.get("background_medians", {}).get(column, 0.5))
            )
            _, changed_probability = self._ensemble_probability(perturbed, variant)
            effect = float(probability - changed_probability)
            rows.append(
                {
                    "component_score": column,
                    "display_name": self.component_metadata.get(column, {}).get("display_name", column),
                    "value": float(scores[column]),
                    "background_median": float(perturbed[column]),
                    "background_scope": background["scope"],
                    "background_sample_size": background["sample_size"],
                    "probability_change_when_replaced": effect,
                    "probability_change_percentage_points": effect * 100.0,
                    "effect_relative_to_background": (
                        "increases predicted interaction probability" if effect > 0
                        else "decreases predicted interaction probability" if effect < 0
                        else "neutral"
                    ),
                }
            )
        total_effect = float(sum(abs(row["probability_change_when_replaced"]) for row in rows))
        reliable_share = total_effect >= self.local_importance_min_effect
        for row in rows:
            row["local_importance_share"] = (
                abs(row["probability_change_when_replaced"]) / total_effect
                if reliable_share and total_effect else None
            )
        return {
            "background_scope": background["scope"],
            "background_sample_size": background["sample_size"],
            "total_absolute_probability_effect": total_effect,
            "relative_importance_reliable": reliable_share,
            "relative_importance_note": (
                None if reliable_share
                else "The total absolute effect is below the configured minimum; relative percentage rankings are suppressed."
            ),
            "components": sorted(rows, key=lambda row: abs(row["probability_change_when_replaced"]), reverse=True),
        }

    def _tree_shap_explanation(self, scores: dict[str, float], variant: str) -> dict[str, Any]:
        if not self.tree_shap_enabled:
            return {"status": "disabled", "reason": "TreeSHAP is disabled for bounded online memory.", "values": []}
        try:
            import shap
            bundle, _, columns = self._variant_context(variant)
            row = pd.DataFrame([[scores[column] for column in columns]], columns=columns)
            pipeline = bundle["model"]
            imputer = pipeline.named_steps["imputer"]
            classifier = pipeline.named_steps["classifier"]
            row_imputed = imputer.transform(row)
            background_imputed = None
            if self.explainability_background is not None and all(
                column in self.explainability_background.columns for column in columns
            ):
                background = self.explainability_background[columns].head(200)
                background_imputed = imputer.transform(background)
            explanation = shap.TreeExplainer(classifier, data=background_imputed)(row_imputed)
            values = np.asarray(explanation.values)
            base_values = np.asarray(explanation.base_values)
            if values.ndim == 3:
                values = values[0, :, 1]
                base = float(base_values[0, 1] if base_values.ndim > 1 else base_values.reshape(-1)[-1])
            elif values.ndim == 2:
                values = values[0]
                base = float(base_values.reshape(-1)[-1])
            else:
                values = values.reshape(-1)[: len(columns)]
                base = float(base_values.reshape(-1)[-1])
            output = []
            for column, value, shap_value in zip(columns, row.iloc[0], values):
                output.append(
                    {
                        "component_score": column,
                        "display_name": self.component_metadata.get(column, {}).get("display_name", column),
                        "value": float(value),
                        "shap_value_raw_ensemble_output": float(shap_value),
                        "effect": "increases raw ensemble output" if shap_value > 0 else "decreases raw ensemble output" if shap_value < 0 else "neutral",
                    }
                )
            return {
                "status": "computed",
                "explained_output": "uncalibrated Extra Trees ensemble probability",
                "base_value": base,
                "values": sorted(output, key=lambda item: abs(item["shap_value_raw_ensemble_output"]), reverse=True),
            }
        except Exception as exc:
            return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "values": []}

    def _evidence(self, pair: ResolvedPair) -> dict[str, Any]:
        if self.driver is None:
            return {
                "status": "unavailable",
                "tier": "Tier 3",
                "tier_reason": "Neo4j driver is unavailable.",
                "evidence_support": "low",
                "known_direct_interaction": False,
            }
        query = """
        MATCH (c:Compound), (p:Protein)
        WITH c, p, properties(p) AS pp
        WHERE toString(c.cid) = $cid
          AND $protein_id IN [
              toString(pp['protein_id']), toString(pp['uniprot_id']),
              toString(pp['accession']), toString(pp['cyp_symbol']),
              toString(pp['symbol'])
          ]
        CALL (c, p) {
          OPTIONAL MATCH (c)<-[:ASSERTS_CHEMICAL]-(direct:Interaction)-[:ASSERTS_TARGET]->(p)
          OPTIONAL MATCH (direct)-[:SUPPORTED_BY_ENDPOINT]->(endpoint:Endpoint)
          OPTIONAL MATCH (measure_group:MeasureGrp)-[:HAS_ENDPOINT]->(endpoint)
          OPTIONAL MATCH (bioassay:BioAssay)-[:HAS_ENDPOINT]->(endpoint)
          WITH direct,
               collect(DISTINCT endpoint) AS endpoints,
               collect(DISTINCT measure_group) AS measure_groups,
               collect(DISTINCT bioassay) AS bioassays
          WITH collect(CASE WHEN direct IS NULL THEN null ELSE {
                 interaction_id: elementId(direct),
                 interaction: properties(direct),
                 endpoint_count: size(endpoints),
                 endpoint_ids: [item IN endpoints | elementId(item)],
                 measure_group_count: size(measure_groups),
                 bioassay_count: size(bioassays)
               } END) AS rows,
               sum(size(endpoints)) AS endpoint_path_count,
               sum(size(measure_groups)) AS measure_group_count,
               sum(size(bioassays)) AS bioassay_count
          RETURN [row IN rows WHERE row IS NOT NULL][0..10] AS direct_interactions,
                 endpoint_path_count, measure_group_count, bioassay_count
        }
        CALL (c, p) {
          OPTIONAL MATCH (c)-[similarity_edge:SIMILAR_TO]-(analog:Compound)
                         <-[:ASSERTS_CHEMICAL]-(analog_interaction:Interaction)
                         -[:ASSERTS_TARGET]->(p)
          OPTIONAL MATCH (analog_interaction)-[:SUPPORTED_BY_ENDPOINT]->(analog_endpoint:Endpoint)
          WITH analog, similarity_edge, analog_interaction,
               collect(DISTINCT analog_endpoint) AS endpoints
          WITH collect(CASE WHEN analog IS NULL THEN null ELSE {
                 cid: analog.cid,
                 name: coalesce(analog.preferred_name, analog.name, analog.title, analog.label),
                 similarity: coalesce(similarity_edge.similarity, similarity_edge.score, similarity_edge.tanimoto),
                 same_target: true,
                 interaction_id: elementId(analog_interaction),
                 interaction: properties(analog_interaction),
                 endpoint_count: size(endpoints),
                 endpoint_ids: [item IN endpoints | elementId(item)]
               } END) AS rows
          RETURN [row IN rows WHERE row IS NOT NULL][0..10] AS analog_support
        }
        RETURN direct_interactions, analog_support,
               endpoint_path_count, measure_group_count, bioassay_count
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
                "evidence_support": "low",
                "known_direct_interaction": False,
            }
        if not record:
            return {
                "status": "not_found",
                "tier": "Tier 3",
                "tier_reason": "No graph evidence was returned.",
                "evidence_support": "low",
                "known_direct_interaction": False,
            }
        direct = [item for item in (record["direct_interactions"] or []) if item]
        analogs = [item for item in (record["analog_support"] or []) if item and item.get("cid") is not None]
        endpoint_count = int(record["endpoint_path_count"] or 0)
        measure_group_count = int(record["measure_group_count"] or 0)
        bioassay_count = int(record["bioassay_count"] or 0)
        if direct:
            tier = "Tier 1"
            reason = "A direct PRING interaction assertion exists for this exact compound-target pair."
            evidence_support = "high"
            completeness = "complete" if endpoint_count or measure_group_count or bioassay_count else "partial"
        elif analogs:
            tier = "Tier 2"
            reason = "Structurally similar compounds have interaction evidence for the same CYP target."
            evidence_support = "moderate"
            completeness = "analogue_supported"
        else:
            tier = "Tier 3"
            reason = "No direct or same-target analogue support was found; this is a model-only hypothesis."
            evidence_support = "low"
            completeness = "model_only"
        return {
            "status": "available",
            "tier": tier,
            "tier_reason": reason,
            "evidence_support": evidence_support,
            "provenance_completeness": completeness,
            "known_direct_interaction": bool(direct),
            "direct_interactions": direct,
            "similar_compound_support": analogs,
            "endpoint_path_count": endpoint_count,
            "measure_group_count": measure_group_count,
            "bioassay_count": bioassay_count,
        }

    def _target_metrics(self, pair: ResolvedPair) -> dict[str, Any] | None:
        for row in self.per_target_metrics:
            group = str(row.get("group", row.get("target_key", "")))
            if pair.protein_id in group or pair.target_key == group:
                return _json_safe(row)
        return None

    def _model_certainty(self, probability: float, threshold: float, disagreement: float, applicability: dict[str, Any]) -> dict[str, Any]:
        margin = abs(probability - threshold)
        entropy = _entropy(probability)
        if applicability.get("status") == "outside_domain":
            band = "low"
            reason = "At least one component score lies outside the validated reference range."
        elif margin >= 0.25 and disagreement <= 0.20 and entropy <= 0.50:
            band = "high"
            reason = "The probability is far from the decision threshold with low component disagreement."
        elif margin >= 0.10 and disagreement <= 0.40:
            band = "moderate"
            reason = "The decision has a useful margin but some model disagreement remains."
        else:
            band = "low"
            reason = "The prediction is close to the threshold or component disagreement is high."
        return {
            "band": band,
            "reason": reason,
            "decision_margin": float(margin),
            "component_disagreement_std": float(disagreement),
            "predictive_entropy_bits": float(entropy),
        }

    @staticmethod
    def _result_classification(predicted_label: int, evidence: dict[str, Any]) -> tuple[str, str]:
        direct = bool(evidence.get("known_direct_interaction"))
        if direct and predicted_label:
            return "known_interaction_rediscovered", "Known interaction / model rediscovery"
        if direct and not predicted_label:
            return "known_interaction_not_rediscovered", "Known interaction not recovered at the selected threshold"
        if predicted_label:
            return "novel_predicted_interaction", "Interaction predicted at the selected threshold"
        return "interaction_not_predicted", "Interaction not predicted at the selected threshold"

    @staticmethod
    def _recommended_action(result_status: str, evidence: dict[str, Any], model_certainty: dict[str, Any]) -> str:
        if result_status == "known_interaction_rediscovered":
            return "Review the direct assertion and provenance; treat this as validation/rediscovery, not a novel prediction."
        if result_status == "known_interaction_not_rediscovered":
            return "Investigate model-evidence disagreement and verify the label, graph mapping, and assay context."
        if result_status == "novel_predicted_interaction" and evidence.get("tier") == "Tier 2":
            return "Prioritize for expert review using the same-target analogue evidence, followed by independent validation."
        if result_status == "novel_predicted_interaction":
            return "Prioritize for independent database review or experimental validation; current support is model-only."
        if model_certainty.get("band") == "low":
            return "Do not conclude biological inactivity. Retain as unconfirmed and consider additional evidence or model review."
        return "Lower-priority candidate at the current threshold; retain as unconfirmed rather than biologically inactive."

    def _build_result(
        self, pair: ResolvedPair, scores: dict[str, float], component_details: dict[str, Any],
        cache_write: dict[str, Any] | None, variant: str = "primary"
    ) -> dict[str, Any]:
        bundle, variant_manifest, active_columns = self._variant_context(variant)
        raw_probability, probability = self._ensemble_probability(scores, variant)
        threshold = float(bundle["threshold"])
        predicted_label = int(probability >= threshold)
        active_score_values = [float(scores[column]) for column in active_columns]
        disagreement = float(np.std(active_score_values))
        evidence = self._evidence(pair)
        applicability = self.reference.applicability(
            pair.target_key, scores, self.minimum_target_background_rows, score_columns=active_columns
        )
        certainty = self._model_certainty(probability, threshold, disagreement, applicability)
        result_status, predicted_class = self._result_classification(predicted_label, evidence)
        local = self._local_explanation(pair, scores, probability, variant)
        score_source = clean_optional_text(component_details.get("score_source"), "unknown")
        target_metrics = self._target_metrics(pair) if variant == "primary" else None
        if variant == "primary":
            global_component_importance = self.component_importance
        else:
            classifier = bundle.get("model").named_steps.get("classifier")
            importance_values = getattr(classifier, "feature_importances_", np.zeros(len(active_columns)))
            global_component_importance = [
                {"component_score": column, "importance": float(value)}
                for column, value in zip(active_columns, importance_values)
            ]
        component_scores = [
            {
                "component_score": column,
                "display_name": self.component_metadata.get(column, {}).get("display_name", column),
                "artifact_role": self.component_metadata.get(column, {}).get("artifact_role"),
                "score": float(scores[column]),
            }
            for column in active_columns
        ]
        statement = (
            f"The calibrated interaction probability is {probability:.3f}. At the locked global threshold "
            f"of {threshold:.3f}, the result is: {predicted_class}."
        )
        return _json_safe(
            {
                "pair": asdict(pair),
                "model": {
                    "name": bundle.get("model_name"),
                    "version": bundle.get("model_version"),
                    "variant": variant,
                    "score_source": score_source,
                    "calibration": variant_manifest.get("calibration", self.manifest.get("calibration")),
                    "threshold_selection": variant_manifest.get("selection_basis", self.manifest.get("selection_basis")),
                    "graph_version": self.graph_version,
                },
                "prediction": {
                    "task_definition": "Aggregated PRING compound-CYP450 interaction-label prediction",
                    "raw_ensemble_probability": raw_probability,
                    "calibrated_probability": probability,
                    "threshold": threshold,
                    "target_context_threshold": target_metrics.get("selected_threshold") if target_metrics else None,
                    "predicted_label": predicted_label,
                    "predicted_class": predicted_class,
                    "result_status": result_status,
                    "knowledge_graph_support": evidence.get("tier"),
                },
                "component_scores": {column: float(scores[column]) for column in active_columns},
                "diagnostic_component_scores": {
                    column: (None if not np.isfinite(float(value)) else float(value))
                    for column, value in scores.items()
                },
                "component_score_details": component_scores,
                "component_details": component_details,
                "prediction_cache": cache_write or {"status": "lookup_hit"},
                "explainability": {
                    "local_component_explanation": local,
                    "local_component_contributions": local["components"],
                    "tree_shap": self._tree_shap_explanation(scores, variant),
                    "global_component_importance": global_component_importance,
                    "stage1_structural_feature_importance": self.stage1_importance if variant == "primary" else [],
                    "applicability_domain": applicability,
                    "uncertainty_metrics": {
                        "decision_margin": certainty["decision_margin"],
                        "component_disagreement_std": certainty["component_disagreement_std"],
                        "predictive_entropy_bits": certainty["predictive_entropy_bits"],
                        "model_certainty": certainty["band"],
                        "model_certainty_reason": certainty["reason"],
                    },
                    "calibration_metrics_on_frozen_test": {
                        "brier_score": variant_manifest.get("metrics", {}).get("brier_score"),
                        "expected_calibration_error": variant_manifest.get("metrics", {}).get("expected_calibration_error"),
                    },
                },
                "evidence": evidence,
                "validation": {
                    "global_test_metrics": variant_manifest.get("metrics", self.manifest.get("metrics", {})),
                    "target_specific_metrics": target_metrics,
                },
                "interpretation": {
                    "statement": statement,
                    "model_certainty": certainty["band"],
                    "evidence_support": evidence.get("evidence_support", "low"),
                    "recommended_action": self._recommended_action(result_status, evidence, certainty),
                    "scope_warning": (
                        "This output predicts the aggregated PRING interaction label. It does not by itself distinguish "
                        "substrate, inhibitor, inducer, binding, or metabolic mechanism and is not proof of causality, "
                        "clinical effect, biological inactivity, or safety."
                    ),
                },
            }
        )

    def _cache_rows(
        self, live_items: list[tuple[ResolvedPair, dict[str, float], dict[str, Any], str]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pair, scores, details, variant in live_items:
            bundle, _, active_columns = self._variant_context(variant)
            raw_probability, probability = self._ensemble_probability(scores, variant)
            threshold = float(bundle["threshold"])
            rows.append(
                {
                    "pair_key": f"{pair.compound_key}||{pair.target_key}",
                    "compound_key": pair.compound_key,
                    "target_key": pair.target_key,
                    **{column: float(scores.get(column, np.nan)) for column in self.score_columns},
                    "final_split": "production_inference",
                    "record_type": "production_prediction_cache",
                    "score_source": "production_prediction_cache",
                    "model_variant": variant,
                    "active_score_columns_json": json.dumps(active_columns),
                    "exclude_from_training": True,
                    "observed_label": np.nan,
                    "predicted_label": int(probability >= threshold),
                    "raw_ensemble_probability": raw_probability,
                    "calibrated_probability": probability,
                    "decision_threshold": threshold,
                    "model_name": bundle.get("model_name"),
                    "model_version": bundle.get("model_version"),
                    "graph_version": self.graph_version,
                    "created_at_utc": utc_now(),
                    "stage1_feature_source": details.get("stage1_feature_source"),
                    "stage1_pair_features_json": json.dumps(details.get("stage1_pair_features", {}), sort_keys=True),
                }
            )
        return rows

    def validate_live_parity(self, force: bool = False) -> dict[str, Any]:
        with self._parity_lock:
            if not force and self._parity_status.get("status") in {
                "passed", "passed_with_stage3_fallback", "failed"
            }:
                return self._parity_status
            started = datetime.now(timezone.utc)
            sample = self.reference.parity_sample(
                self.parity_sample_size, seed=int(self.bundle.get("seed", 42))
            )
            if len(sample) < 2:
                self._parity_status = {
                    "status": "failed",
                    "required": self.parity_required,
                    "error": "At least two validated reference pairs are required for parity validation.",
                    "sample_size": int(len(sample)),
                    "checked_at_utc": utc_now(),
                }
                return self._parity_status

            pairs: list[ResolvedPair] = []
            reference_rows: dict[tuple[str, str], dict[str, Any]] = {}
            resolution_errors: list[str] = []
            for row in sample.to_dict(orient="records"):
                try:
                    pair = self._resolve_pair(str(row["compound_key"]), str(row["target_key"]))
                    if not pair.compound_element_id or not pair.protein_element_id:
                        raise RuntimeError("Pair could not be mapped to Neo4j element IDs.")
                    pairs.append(pair)
                    reference_rows[(pair.compound_key, pair.target_key)] = row
                except Exception as exc:
                    resolution_errors.append(
                        f"{row.get('compound_key')} / {row.get('target_key')}: {exc}"
                    )
            if len(pairs) < 2:
                self._parity_status = {
                    "status": "failed",
                    "required": self.parity_required,
                    "error": "Fewer than two parity pairs could be resolved to the live graph.",
                    "resolution_errors": resolution_errors,
                    "checked_at_utc": utc_now(),
                }
                return self._parity_status

            try:
                live_output = self.live.score_many(
                    pairs, self._node_properties, include_stage1=True
                )
            except Exception as exc:
                self._parity_status = {
                    "status": "failed",
                    "required": self.parity_required,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "checked_at_utc": utc_now(),
                }
                return self._parity_status

            component_results: dict[str, Any] = {}
            for column in self.score_columns:
                expected = pd.Series([
                    float(reference_rows[(pair.compound_key, pair.target_key)][column])
                    for pair in pairs
                ])
                actual = pd.Series([
                    float(live_output[(pair.compound_key, pair.target_key)]["scores"][column])
                    for pair in pairs
                ])
                absolute = (actual - expected).abs()
                spearman = (
                    float(expected.corr(actual, method="spearman"))
                    if expected.nunique() > 1 and actual.nunique() > 1
                    else (1.0 if np.allclose(expected, actual) else 0.0)
                )
                result = {
                    "display_name": self.component_metadata.get(column, {}).get("display_name", column),
                    "sample_size": int(len(expected)),
                    "mae": float(absolute.mean()),
                    "max_absolute_error": float(absolute.max()),
                    "spearman": spearman,
                }
                result["passed"] = bool(
                    result["mae"] <= self.parity_mae_max
                    and result["max_absolute_error"] <= self.parity_max_abs_error
                    and result["spearman"] >= self.parity_spearman_min
                )
                component_results[column] = result

            primary_reference_decisions: list[int] = []
            primary_live_decisions: list[int] = []
            fallback_reference_decisions: list[int] = []
            fallback_live_decisions: list[int] = []
            comparisons: list[dict[str, Any]] = []
            primary_threshold = float(self.bundle["threshold"])
            fallback_threshold = (
                float(self.fallback_bundle["threshold"]) if self.fallback_bundle is not None else None
            )
            for pair in pairs:
                key = (pair.compound_key, pair.target_key)
                expected_scores = {
                    column: float(reference_rows[key][column]) for column in self.score_columns
                }
                actual_scores = live_output[key]["scores"]
                _, expected_primary = self._ensemble_probability(expected_scores, "primary")
                _, actual_primary = self._ensemble_probability(actual_scores, "primary")
                primary_reference_decisions.append(int(expected_primary >= primary_threshold))
                primary_live_decisions.append(int(actual_primary >= primary_threshold))
                row = {
                    "compound_key": pair.compound_key,
                    "target_key": pair.target_key,
                    "reference_probability_primary": expected_primary,
                    "live_probability_primary": actual_primary,
                    "stage1_feature_source": live_output[key].get("details", {}).get("stage1_feature_source"),
                }
                if self.fallback_bundle is not None:
                    _, expected_fallback = self._ensemble_probability(expected_scores, "stage3_fallback")
                    _, actual_fallback = self._ensemble_probability(actual_scores, "stage3_fallback")
                    fallback_reference_decisions.append(int(expected_fallback >= fallback_threshold))
                    fallback_live_decisions.append(int(actual_fallback >= fallback_threshold))
                    row["reference_probability_stage3_fallback"] = expected_fallback
                    row["live_probability_stage3_fallback"] = actual_fallback
                comparisons.append(row)

            primary_agreement = float(np.mean(
                np.asarray(primary_reference_decisions) == np.asarray(primary_live_decisions)
            ))
            primary_component_pass = all(
                component_results[column]["passed"] for column in self.score_columns
            )
            primary_pass = bool(
                primary_component_pass
                and primary_agreement >= self.parity_decision_agreement_min
            )

            stage3_columns = [
                LiveComponentScorer.RGCN_COLUMN, LiveComponentScorer.HGT_COLUMN
            ]
            stage3_component_pass = all(
                component_results.get(column, {}).get("passed", False) for column in stage3_columns
            )
            fallback_agreement = (
                float(np.mean(
                    np.asarray(fallback_reference_decisions) == np.asarray(fallback_live_decisions)
                ))
                if fallback_reference_decisions else None
            )
            fallback_pass = bool(
                self.allow_stage3_fallback
                and self.fallback_bundle is not None
                and stage3_component_pass
                and fallback_agreement is not None
                and fallback_agreement >= self.parity_decision_agreement_min
            )

            if primary_pass:
                status = "passed"
                active_variant = "primary"
            elif fallback_pass:
                status = "passed_with_stage3_fallback"
                active_variant = "stage3_fallback"
            else:
                status = "failed"
                active_variant = None

            stage1_result = component_results.get(LiveComponentScorer.STAGE1_COLUMN, {})
            stage1_sources = sorted({
                str(live_output[(pair.compound_key, pair.target_key)].get("details", {}).get("stage1_feature_source"))
                for pair in pairs
            })
            stage1_diagnosis = {
                "passed": bool(stage1_result.get("passed")),
                "feature_sources": stage1_sources,
                "interpretation": (
                    "The exact training-time pair-feature export reproduced Stage 1."
                    if stage1_result.get("passed") and "training_time_pair_feature_export" in stage1_sources
                    else "The current Neo4j FastRP-derived features do not reproduce the training-time Stage 1 component. "
                         "FastRP coordinates are graph-projection specific. Mount the original Stage 1 pair-feature exports "
                         "or use the validated Stage 3 fallback ensemble."
                    if not stage1_result.get("passed")
                    else "Stage 1 parity passed."
                ),
            }

            self._parity_status = {
                "status": status,
                "required": self.parity_required,
                "active_live_variant": active_variant,
                "sample_size": len(pairs),
                "sample_size_requested": self.parity_sample_size,
                "component_results": component_results,
                "primary_decision_agreement": primary_agreement,
                "stage3_fallback_decision_agreement": fallback_agreement,
                "primary_passed": primary_pass,
                "stage3_fallback_passed": fallback_pass,
                "stage1_diagnosis": stage1_diagnosis,
                "thresholds": {
                    "mae_max": self.parity_mae_max,
                    "max_absolute_error": self.parity_max_abs_error,
                    "spearman_min": self.parity_spearman_min,
                    "decision_agreement_min": self.parity_decision_agreement_min,
                },
                "resolution_errors": resolution_errors,
                "comparisons": comparisons,
                "checked_at_utc": utc_now(),
                "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            }
            return self._parity_status

    def _ensure_live_parity(self) -> str:
        if not self.parity_required:
            return "primary"
        status = self.validate_live_parity(force=False)
        variant = status.get("active_live_variant")
        if status.get("status") not in {"passed", "passed_with_stage3_fallback"} or not variant:
            raise RuntimeError(
                "Live inference parity validation failed. New scores were not generated or cached. "
                f"Details: {status}"
            )
        return str(variant)

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
            raise ValueError(f"Request contains {pair_count} pairs; the configured limit is {self.max_pairs}.")

        resolved_pairs: list[ResolvedPair] = []
        errors: list[dict[str, str]] = []
        for compound in compounds:
            for target in targets:
                try:
                    resolved_pairs.append(self._resolve_pair(compound, target))
                except Exception as exc:
                    errors.append({"compound": str(compound), "target": str(target), "error": f"{type(exc).__name__}: {exc}"})

        score_records: dict[tuple[str, str], tuple[dict[str, float], dict[str, Any], str]] = {}
        missing_pairs: list[ResolvedPair] = []
        for pair in resolved_pairs:
            lookup = None if self.score_mode == "live" else self._lookup_scores(pair)
            if lookup is not None:
                scores, details = lookup
                variant = clean_optional_text(details.get("model_variant"), "primary")
                if not self._variant_available(variant):
                    # Ignore legacy cache rows created before model-variant and parity
                    # provenance were introduced. They will be recomputed and a new
                    # validated cache row will be appended.
                    missing_pairs.append(pair)
                else:
                    score_records[(pair.compound_key, pair.target_key)] = (scores, details, variant)
            else:
                missing_pairs.append(pair)

        if missing_pairs and self.score_mode == "precomputed":
            for pair in missing_pairs:
                errors.append({
                    "compound": pair.compound_input,
                    "target": pair.target_input,
                    "error": "Pair was absent from both the validated reference frame and the production prediction cache; live inference is disabled.",
                })
            missing_pairs = []

        live_items: list[tuple[ResolvedPair, dict[str, float], dict[str, Any], str]] = []
        cache_write: dict[str, Any] | None = None
        if missing_pairs:
            try:
                live_variant = self._ensure_live_parity()
                live_output = self.live.score_many(
                    missing_pairs,
                    self._node_properties if live_variant == "primary" else None,
                    include_stage1=(live_variant == "primary"),
                )
            except Exception as exc:
                for pair in missing_pairs:
                    errors.append({"compound": pair.compound_input, "target": pair.target_input, "error": f"{type(exc).__name__}: {exc}"})
            else:
                for pair in missing_pairs:
                    item = live_output[(pair.compound_key, pair.target_key)]
                    scores = item["scores"]
                    details = dict(item.get("details", {}))
                    details["score_source"] = "live_component_inference"
                    details["model_variant"] = live_variant
                    if live_variant == "stage3_fallback":
                        details["stage1_parity_warning"] = self._parity_status.get("stage1_diagnosis")
                    score_records[(pair.compound_key, pair.target_key)] = (scores, details, live_variant)
                    live_items.append((pair, scores, details, live_variant))

                if live_items and self.persist_new_scores:
                    try:
                        cache_write = self.cache.upsert_rows(self._cache_rows(live_items))
                    except Exception as exc:
                        cache_write = {"status": "failed", "added_rows": 0, "error_type": type(exc).__name__, "error": str(exc)}
                        if self.require_cache_write:
                            for pair, _, _, _ in live_items:
                                score_records.pop((pair.compound_key, pair.target_key), None)
                                errors.append({
                                    "compound": pair.compound_input,
                                    "target": pair.target_input,
                                    "error": f"Live scores were generated but cache persistence failed: {type(exc).__name__}: {exc}",
                                })
                    for _, _, details, _ in live_items:
                        details["cache_write"] = cache_write
                elif live_items:
                    cache_write = {"status": "disabled", "added_rows": 0}

        results: list[dict[str, Any]] = []
        for pair in resolved_pairs:
            record = score_records.get((pair.compound_key, pair.target_key))
            if record is None:
                continue
            scores, details, variant = record
            pair_cache = details.get("cache_write") if details.get("score_source") == "live_component_inference" else None
            results.append(self._build_result(pair, scores, details, pair_cache, variant))

        source_counts: dict[str, int] = {}
        for result in results:
            source = clean_optional_text(result.get("model", {}).get("score_source"), "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        status_counts: dict[str, int] = {}
        for result in results:
            status = str(result.get("prediction", {}).get("result_status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

        return _json_safe({
            "model_status": self.status(),
            "report_context": {
                "model_provenance": {
                    "model_name": self.bundle.get("model_name"),
                    "model_version": self.bundle.get("model_version"),
                    "graph_version": self.graph_version,
                    "calibration": self.manifest.get("calibration"),
                    "threshold_selection": self.manifest.get("selection_basis"),
                    "score_columns": self.score_columns,
                    "available_live_variants": {
                        "primary": {
                            "model_name": self.bundle.get("model_name"),
                            "model_version": self.bundle.get("model_version"),
                            "metrics": self.manifest.get("metrics", {}),
                        },
                        "stage3_fallback": {
                            "available": self.fallback_bundle is not None,
                            "model_name": self.fallback_bundle.get("model_name") if self.fallback_bundle else None,
                            "model_version": self.fallback_bundle.get("model_version") if self.fallback_bundle else None,
                            "metrics": self.fallback_manifest.get("metrics", {}),
                        },
                    },
                },
                "global_validation_metrics": self.manifest.get("metrics", {}),
                "validation_rows": self.manifest.get("test_rows"),
                "live_inference_parity": self._parity_status,
                "score_source_counts": source_counts,
                "result_status_counts": status_counts,
            },
            "predictions": results,
            "errors": errors,
            "requested_pairs": pair_count,
            "successful_pairs": len(results),
            "validated_precomputed_hits": source_counts.get("validated_precomputed_record", 0),
            "production_cache_hits": source_counts.get("production_prediction_cache", 0),
            "live_predictions_generated": len(live_items),
            "cache_write": cache_write,
        })
