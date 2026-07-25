from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
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
    paren = re.search(r"\(([^()]+)\)\s*$", text)
    if paren and re.fullmatch(r"[A-Za-z0-9_.-]+", paren.group(1).strip()):
        return paren.group(1).strip()
    return text.split()[0] if text else text


def _normalise_key(value: Any) -> str:
    return str(value or "").strip().casefold()


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
    def __init__(self, path: str | Path | None, score_columns: list[str]):
        self.path = Path(path) if path else None
        self.score_columns = list(score_columns)
        self.frame: pd.DataFrame | None = None
        self.index: dict[tuple[str, str], dict[str, Any]] = {}
        self.error: str | None = None
        if not self.path:
            self.error = "PRECOMPUTED_SCORE_FRAME is not configured."
            return
        if not self.path.exists():
            self.error = f"Precomputed score frame does not exist: {self.path}"
            return
        try:
            frame = pd.read_csv(self.path, low_memory=False)
        except Exception as exc:
            self.error = f"Could not read precomputed score frame {self.path}: {exc}"
            return
        required = ["compound_key", "target_key", *self.score_columns]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            self.error = f"Precomputed score frame is missing columns: {missing}"
            return
        self.frame = frame
        for row in frame.to_dict(orient="records"):
            key = (_normalise_key(row["compound_key"]), _normalise_key(row["target_key"]))
            self.index[key] = row

    @property
    def available(self) -> bool:
        return bool(self.index)

    @property
    def row_count(self) -> int:
        return len(self.index)

    def get(self, compound_key: str, target_key: str) -> dict[str, Any] | None:
        return self.index.get((_normalise_key(compound_key), _normalise_key(target_key)))


class PRINGPredictionService:
    """Production predictor using precomputed component scores only.

    Full Stage-3 graph scoring must be performed offline. The API resolves the
    requested pair, retrieves frozen component scores, applies the locked
    ensemble and calibration, and returns graph evidence when Neo4j is available.
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

        self.precomputed = PrecomputedScoreStore(
            os.getenv("PRECOMPUTED_SCORE_FRAME", "/results/production/finalized_training_frame.csv"),
            self.score_columns,
        )
        self.max_pairs = int(os.getenv("PREDICTION_MAX_PAIRS_PER_REQUEST", os.getenv("PREDICTION_MAX_PAIRS", "25")))
        self.tree_shap_enabled = os.getenv("PREDICTION_TREE_SHAP_ENABLED", os.getenv("PREDICTION_ENABLE_TREE_SHAP", "false")).lower() in {"1", "true", "yes", "on"}

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

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def _read_optional_csv(self, name: str) -> list[dict[str, Any]]:
        path = self.model_dir / name
        return pd.read_csv(path).to_dict(orient="records") if path.exists() else []

    def status(self) -> dict[str, Any]:
        ready = self.precomputed.available
        return {
            "status": "ready" if ready else "not_ready",
            "model": self.manifest,
            "serving_architecture": "precomputed_component_scores_plus_calibrated_ensemble",
            "score_mode": "precomputed",
            "precomputed_score_lookup": ready,
            "precomputed_score_rows": self.precomputed.row_count,
            "precomputed_score_frame": str(self.precomputed.path) if self.precomputed.path else None,
            "precomputed_score_error": self.precomputed.error,
            "allow_live_stage3": False,
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
            raise ValueError(f"Could not resolve pair: {compound_input} / {target_input}")
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

    def _component_scores(self, pair: ResolvedPair) -> tuple[dict[str, float], dict[str, Any]]:
        row = self.precomputed.get(pair.compound_key, pair.target_key)
        if row is None:
            raise ValueError(
                "The selected pair is not present in the production score store. "
                "Score candidate pairs offline and refresh finalized_training_frame.csv. "
                f"Resolved keys: {pair.compound_key!r}, {pair.target_key!r}."
            )
        scores: dict[str, float] = {}
        for column in self.score_columns:
            value = row.get(column)
            if value is None or pd.isna(value):
                raise ValueError(f"Precomputed score {column!r} is missing for the selected pair.")
            scores[column] = float(value)
        details = {
            "score_source": "precomputed_exact_model_outputs",
            "final_split": row.get("final_split"),
            "observed_label": row.get("label"),
            "pair_key": row.get("pair_key"),
        }
        return scores, details

    def _ensemble_probability(self, scores: dict[str, float]) -> tuple[float, float]:
        row = pd.DataFrame([[scores[c] for c in self.score_columns]], columns=self.score_columns)
        raw = float(self.bundle["model"].predict_proba(row)[:, 1][0])
        calibrated = float(np.asarray(self.bundle["calibrator"].predict(np.asarray([raw], dtype=float))).reshape(-1)[0])
        return raw, calibrated

    def _local_explanation(self, scores: dict[str, float], probability: float) -> list[dict[str, Any]]:
        medians = self.bundle.get("background_medians", {})
        components = {x.get("score_column"): x for x in self.manifest.get("production_components", [])}
        rows: list[dict[str, Any]] = []
        for col in self.score_columns:
            perturbed = dict(scores)
            perturbed[col] = float(medians.get(col, 0.5))
            _, changed = self._ensemble_probability(perturbed)
            contribution = float(probability - changed)
            rows.append({
                "component_score": col,
                "display_name": components.get(col, {}).get("display_name", col),
                "value": float(scores[col]),
                "background_median": float(medians.get(col, 0.5)),
                "probability_change_when_replaced": contribution,
                "direction": "supports active" if contribution > 0 else "supports inactive" if contribution < 0 else "neutral",
            })
        denominator = sum(abs(r["probability_change_when_replaced"]) for r in rows) or 1.0
        for row in rows:
            row["local_importance_share"] = abs(row["probability_change_when_replaced"]) / denominator
        return sorted(rows, key=lambda r: abs(r["probability_change_when_replaced"]), reverse=True)

    def _evidence(self, pair: ResolvedPair) -> dict[str, Any]:
        if self.driver is None:
            return {"status": "unavailable", "tier": "Tier 3", "tier_reason": "Neo4j driver is unavailable."}
        query = """
        MATCH (c:Compound), (p:Protein)
        WITH c, p, properties(p) AS pp
        WHERE toString(c.cid) = $cid
          AND $protein_id IN [toString(pp['protein_id']), toString(pp['uniprot_id']), toString(pp['accession']), toString(pp['cyp_symbol']), toString(pp['symbol'])]
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
                rec = session.run(query, cid=pair.cid, protein_id=pair.protein_id).single()
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc), "tier": "Tier 3", "tier_reason": "Evidence query failed."}
        if not rec:
            return {"status": "not_found", "tier": "Tier 3", "tier_reason": "No graph evidence was returned."}
        direct = [x for x in (rec["direct_interactions"] or []) if x]
        analogs = [x for x in (rec["analog_support"] or []) if x and x.get("cid") is not None]
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

    def predict_pair(self, compound_input: str, target_input: str) -> dict[str, Any]:
        pair = self._resolve_pair(compound_input, target_input)
        scores, details = self._component_scores(pair)
        raw_probability, probability = self._ensemble_probability(scores)
        threshold = float(self.bundle["threshold"])
        predicted_label = int(probability >= threshold)
        disagreement = float(np.std(list(scores.values())))
        margin = float(abs(probability - threshold))
        result = {
            "pair": asdict(pair),
            "model": {
                "name": self.bundle.get("model_name"),
                "version": self.bundle.get("model_version"),
                "score_source": details["score_source"],
                "calibration": self.manifest.get("calibration"),
                "threshold_selection": self.manifest.get("selection_basis"),
            },
            "prediction": {
                "raw_ensemble_probability": raw_probability,
                "calibrated_probability": probability,
                "threshold": threshold,
                "predicted_label": predicted_label,
                "predicted_class": "Active interaction" if predicted_label else "Inactive / unsupported interaction",
            },
            "component_scores": scores,
            "component_details": details,
            "explainability": {
                "local_component_contributions": self._local_explanation(scores, probability),
                "tree_shap": {"status": "disabled", "reason": "Disabled in production serving."},
                "global_component_importance": self.component_importance,
                "stage1_structural_feature_importance": self.stage1_importance,
                "uncertainty_metrics": {
                    "decision_margin": margin,
                    "component_disagreement_std": disagreement,
                    "predictive_entropy_bits": _entropy(probability),
                    "confidence_band": "high" if margin >= 0.25 and disagreement <= 0.20 else "moderate" if margin >= 0.10 else "low",
                },
            },
            "evidence": self._evidence(pair),
            "validation": {
                "global_test_metrics": self.manifest.get("metrics", {}),
                "target_specific_metrics": self._target_metrics(pair),
            },
            "interpretation": {
                "statement": f"The calibrated probability is {probability:.3f}; with threshold {threshold:.3f}, the pair is classified as {'active' if predicted_label else 'inactive/unsupported'}.",
                "scope_warning": "This statistical prediction supports prioritization; it is not proof of mechanism, causality, clinical effect, or safety.",
            },
        }
        return _json_safe(result)

    def predict_many(self, compounds: list[str], targets: list[str]) -> dict[str, Any]:
        if not compounds or not targets:
            raise ValueError("At least one compound and one target are required.")
        pair_count = len(compounds) * len(targets)
        if pair_count > self.max_pairs:
            raise ValueError(f"Request contains {pair_count} pairs; production limit is {self.max_pairs}.")
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for compound in compounds:
            for target in targets:
                try:
                    results.append(self.predict_pair(compound, target))
                except Exception as exc:
                    errors.append({"compound": str(compound), "target": str(target), "error": str(exc)})
        return {
            "model_status": self.status(),
            "predictions": results,
            "errors": errors,
            "requested_pairs": pair_count,
            "successful_pairs": len(results),
        }
