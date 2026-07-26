from __future__ import annotations

import argparse
import gc
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd


def _normalise_pair_key(value: Any) -> str:
    return str(value or "").strip().casefold()


class Stage1PairFeatureStore:
    """Replay the exact Stage 1 pair features exported during modeling.

    FastRP coordinates are tied to the exact graph projection and random seed.
    Recomputing them from a rematerialized Neo4j graph may not reproduce the
    training-time Stage 1 score. This store therefore prefers the original
    exported pair-feature rows whenever they are available.
    """

    DEFAULT_FILES = (
        "compound_target_training_pairs_gds_features.csv",
        "candidate_pairs_gds_features.csv",
        "compound_target_training_pairs_gds_features.parquet",
        "candidate_pairs_gds_features.parquet",
    )
    COMPOUND_COLUMNS = ("compound_key", "compound_node_ref", "compound_ref", "source_ref")
    TARGET_COLUMNS = ("target_key", "protein_node_ref", "protein_ref", "target_ref")

    def __init__(self, feature_columns: list[str]):
        self.feature_columns = list(feature_columns)
        self.index: dict[tuple[str, str], dict[str, float]] = {}
        self.loaded_files: list[str] = []
        self.error: str | None = None
        self._load()

    def _configured_paths(self) -> list[Path]:
        paths: list[Path] = []
        explicit = os.getenv("STAGE1_PAIR_FEATURE_FRAME", "").strip()
        if explicit:
            for token in explicit.split(os.pathsep):
                token = token.strip()
                if token:
                    paths.append(Path(token))
        directory = Path(os.getenv("STAGE1_PAIR_FEATURE_DIR", "/modeling_prepared/stage1_neo4j_gds_baselines"))
        if directory.exists():
            for name in self.DEFAULT_FILES:
                candidate = directory / name
                if candidate.exists():
                    paths.append(candidate)
        # Preserve order while removing duplicates.
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(path)
            if resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)

    def _load(self) -> None:
        errors: list[str] = []
        for path in self._configured_paths():
            try:
                if not path.exists():
                    continue
                frame = self._read(path)
                compound_column = next((c for c in self.COMPOUND_COLUMNS if c in frame.columns), None)
                target_column = next((c for c in self.TARGET_COLUMNS if c in frame.columns), None)
                missing_features = [c for c in self.feature_columns if c not in frame.columns]
                if not compound_column or not target_column or missing_features:
                    errors.append(
                        f"{path}: compound_column={compound_column}, target_column={target_column}, "
                        f"missing_features={missing_features}"
                    )
                    continue
                subset = frame[[compound_column, target_column, *self.feature_columns]].dropna(
                    subset=[compound_column, target_column, *self.feature_columns]
                )
                for row in subset.to_dict(orient="records"):
                    key = (_normalise_pair_key(row[compound_column]), _normalise_pair_key(row[target_column]))
                    if key[0] and key[1]:
                        self.index[key] = {column: float(row[column]) for column in self.feature_columns}
                self.loaded_files.append(str(path))
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        self.error = "; ".join(errors) if errors else None

    @property
    def available(self) -> bool:
        return bool(self.index)

    def get(self, compound_key: str, target_key: str) -> dict[str, float] | None:
        return self.index.get((_normalise_pair_key(compound_key), _normalise_pair_key(target_key)))

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "row_count": len(self.index),
            "loaded_files": self.loaded_files,
            "load_warning": self.error,
        }


class Stage1LiveScorer:
    """Reproduce Stage 1 using exact exported features before Neo4j fallback."""

    def __init__(self, model_dir: str | Path | None, embedding_property: str = "pringFastRP"):
        self.model_dir = Path(model_dir) if model_dir else None
        self.embedding_property = str(embedding_property)
        self.model: Any | None = None
        self.feature_columns: list[str] = []
        self.feature_store: Stage1PairFeatureStore | None = None
        self.load_error: str | None = None

        if not self.model_dir:
            self.load_error = "STAGE1_MODEL_DIR is not configured."
            return
        model_file = self.model_dir / "stage1_tabular_extra_trees.joblib"
        columns_file = self.model_dir / "feature_columns.json"
        if not model_file.exists() or not columns_file.exists():
            self.load_error = (
                f"Stage 1 artifacts are incomplete under {self.model_dir}; expected "
                "stage1_tabular_extra_trees.joblib and feature_columns.json."
            )
            return
        try:
            self.model = joblib.load(model_file)
            loaded_columns = json.loads(columns_file.read_text(encoding="utf-8"))
            if isinstance(loaded_columns, dict):
                loaded_columns = loaded_columns.get("feature_columns", loaded_columns.get("columns", []))
            self.feature_columns = [str(value) for value in loaded_columns]
            if not self.feature_columns:
                raise ValueError("feature_columns.json does not define any feature columns.")
            self.feature_store = Stage1PairFeatureStore(self.feature_columns)
        except Exception as exc:
            self.model = None
            self.feature_columns = []
            self.load_error = f"Could not load Stage 1 artifacts: {type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.feature_columns)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "embedding_property": self.embedding_property,
            "feature_columns": self.feature_columns,
            "exact_pair_feature_store": self.feature_store.status() if self.feature_store else None,
            "error": self.load_error,
        }

    def _predict(self, values: dict[str, float]) -> float:
        frame = pd.DataFrame([[values[column] for column in self.feature_columns]], columns=self.feature_columns)
        return float(self.model.predict_proba(frame)[:, 1][0])

    def _from_neo4j(
        self, compound_properties: dict[str, Any], protein_properties: dict[str, Any]
    ) -> dict[str, float]:
        compound_vector = compound_properties.get(self.embedding_property)
        protein_vector = protein_properties.get(self.embedding_property)
        if not isinstance(compound_vector, (list, tuple)) or not isinstance(protein_vector, (list, tuple)):
            raise RuntimeError(
                f"Neo4j property {self.embedding_property!r} must exist on both Compound and Protein nodes."
            )
        compound = np.asarray(compound_vector, dtype=float)
        protein = np.asarray(protein_vector, dtype=float)
        if compound.ndim != 1 or protein.ndim != 1 or compound.size == 0 or compound.shape != protein.shape:
            raise RuntimeError("Compound and Protein FastRP vectors must be non-empty vectors of equal size.")
        safe_name = self.embedding_property.lower().replace(" ", "_").replace("-", "_")
        dot = float(np.dot(compound, protein))
        denominator = float(np.linalg.norm(compound) * np.linalg.norm(protein))
        difference = compound - protein
        values = {
            f"gds_{safe_name}_dot": dot,
            f"gds_{safe_name}_cosine": float(dot / denominator) if denominator else 0.0,
            f"gds_{safe_name}_l2": float(np.linalg.norm(difference)),
            f"gds_{safe_name}_absdiff_mean": float(np.mean(np.abs(difference))),
            f"gds_{safe_name}_absdiff_max": float(np.max(np.abs(difference))),
        }
        missing = [column for column in self.feature_columns if column not in values]
        if missing:
            raise RuntimeError(
                "The live Stage 1 scorer cannot reproduce the saved feature schema. "
                f"Missing features: {missing}."
            )
        return {column: float(values[column]) for column in self.feature_columns}

    def score(
        self,
        pair: Any,
        node_property_provider: Callable[[Any], tuple[dict[str, Any], dict[str, Any]]] | None,
    ) -> tuple[float, dict[str, float], str]:
        if not self.available:
            raise RuntimeError(self.load_error or "Stage 1 live artifacts are unavailable.")
        exact = self.feature_store.get(pair.compound_key, pair.target_key) if self.feature_store else None
        if exact is not None:
            return self._predict(exact), exact, "training_time_pair_feature_export"
        if node_property_provider is None:
            raise RuntimeError(
                "No exact Stage 1 pair-feature row was found and Neo4j feature recomputation is unavailable."
            )
        compound_properties, protein_properties = node_property_provider(pair)
        values = self._from_neo4j(compound_properties, protein_properties)
        return self._predict(values), values, "neo4j_embedding_recomputed"


class SharedStage3Graph:
    """Load one immutable HeteroData graph shared by R-GCN and HGT."""

    # Must exactly match the synthetic supervision edge used when the sampled
    # R-GCN/HGT checkpoints were trained. HGT relation parameters are keyed by
    # this canonical edge-type string inside the serialized state_dict.
    TARGET_EDGE_TYPE = ("Compound", "stage3_supervision", "Protein")

    def __init__(self, prepared_dir: str | Path | None):
        self.prepared_dir = Path(prepared_dir) if prepared_dir else None
        self._load_lock = threading.RLock()
        self.inference_lock = threading.RLock()
        self.data: Any | None = None
        self.node_maps: dict[str, dict[str, int]] | None = None
        self.stage_dir: Path | None = None
        self.node_counts: dict[str, int] = {}
        self.feature_dims: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return bool(self.prepared_dir and self.prepared_dir.exists())

    @property
    def loaded(self) -> bool:
        return self.data is not None and self.node_maps is not None

    @staticmethod
    def _store_value(store: Any, key: str) -> Any:
        try:
            return store.get(key, None)
        except Exception:
            return getattr(store, key, None)

    def _infer_node_count(self, node_type: str, expected_counts: dict[str, Any]) -> int:
        store = self.data[node_type]
        explicit = self._store_value(store, "num_nodes")
        if explicit is not None:
            return int(explicit)

        features = self._store_value(store, "x")
        if features is not None:
            return int(features.size(0))

        node_map = (self.node_maps or {}).get(node_type, {})
        if node_map:
            return int(max(node_map.values())) + 1

        maximum = -1
        for edge_type in self.data.edge_types:
            source_type, _, target_type = edge_type
            edge_index = self._store_value(self.data[edge_type], "edge_index")
            if edge_index is None or edge_index.numel() == 0:
                continue
            if source_type == node_type:
                maximum = max(maximum, int(edge_index[0].max()))
            if target_type == node_type:
                maximum = max(maximum, int(edge_index[1].max()))
        if maximum >= 0:
            return maximum + 1

        if node_type in expected_counts:
            return int(expected_counts[node_type])
        return 0

    def _validate_expected_counts(self, expected_counts: dict[str, Any]) -> None:
        mismatches: list[str] = []
        for node_type, expected in expected_counts.items():
            actual = self.node_counts.get(str(node_type))
            if actual is not None and int(actual) != int(expected):
                mismatches.append(f"{node_type}: prepared={actual}, checkpoint={int(expected)}")
        if mismatches:
            raise RuntimeError(
                "Stage 3 prepared graph node counts do not match checkpoint metadata: "
                + "; ".join(mismatches[:10])
            )

    def load(self, expected_counts: dict[str, Any] | None = None) -> "SharedStage3Graph":
        expected_counts = {str(key): int(value) for key, value in dict(expected_counts or {}).items()}
        if self.loaded:
            self._validate_expected_counts(expected_counts)
            return self

        with self._load_lock:
            if self.loaded:
                self._validate_expected_counts(expected_counts)
                return self
            if not self.available:
                raise RuntimeError(f"Stage 3 prepared graph directory is unavailable: {self.prepared_dir}")

            from .pyg_runtime import require_pyg_runtime
            require_pyg_runtime()

            import torch
            from .stage3_common import load_heterodata, load_node_maps, resolve_stage3_dir

            resolved = resolve_stage3_dir(self.prepared_dir)
            self.stage_dir = resolved.root
            self.data = load_heterodata(self.stage_dir)
            self.node_maps = load_node_maps(self.stage_dir)

            if self.TARGET_EDGE_TYPE not in self.data.edge_types:
                self.data[self.TARGET_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)

            for node_type, count in expected_counts.items():
                if node_type not in self.data.node_types:
                    self.data[node_type].num_nodes = int(count)

            self.node_counts = {}
            self.feature_dims = {}
            for node_type in self.data.node_types:
                count = self._infer_node_count(node_type, expected_counts)
                self.data[node_type].num_nodes = int(count)
                self.node_counts[node_type] = int(count)
                features = self._store_value(self.data[node_type], "x")
                self.feature_dims[node_type] = (
                    int(features.size(-1))
                    if features is not None and getattr(features, "ndim", 0) >= 2
                    else 0
                )

            self._validate_expected_counts(expected_counts)
            return self

    def unload(self) -> None:
        with self._load_lock:
            self.data = None
            self.node_maps = None
            self.stage_dir = None
            self.node_counts = {}
            self.feature_dims = {}
            gc.collect()

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "loaded": self.loaded,
            "prepared_dir": str(self.prepared_dir) if self.prepared_dir else None,
            "resolved_stage_dir": str(self.stage_dir) if self.stage_dir else None,
            "node_type_count": len(self.node_counts) if self.loaded else None,
            "edge_type_count": len(self.data.edge_types) if self.loaded else None,
            "target_edge_type": list(self.TARGET_EDGE_TYPE),
            "shared_by": ["stage3_rgcn", "stage3_hgt"],
        }


class Stage3LiveScorer:
    """Lazy, batch-capable scorer for one sampled Stage 3 checkpoint."""

    def __init__(
        self,
        kind: str,
        model_dir: str | Path | None,
        shared_graph: SharedStage3Graph,
        device: str = "cpu",
    ):
        self.kind = str(kind).lower()
        if self.kind not in {"rgcn", "hgt"}:
            raise ValueError(f"Unsupported Stage 3 scorer kind: {kind}")
        self.model_dir = Path(model_dir) if model_dir else None
        self.shared_graph = shared_graph
        self.device_name = str(device)
        self._load_lock = threading.RLock()
        self.model: Any | None = None
        self.args: argparse.Namespace | None = None
        self.device: Any | None = None
        self._score_rows: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.load_error: str | None = None

    @property
    def metadata_path(self) -> Path | None:
        if not self.model_dir:
            return None
        filename = "rgcn_sampled_metadata.json" if self.kind == "rgcn" else "hgt_sampled_metadata.json"
        return self.model_dir / filename

    @property
    def checkpoint_path(self) -> Path | None:
        return self.model_dir / "best_model.pt" if self.model_dir else None

    @property
    def artifact_available(self) -> bool:
        return bool(
            self.shared_graph.available
            and self.metadata_path
            and self.metadata_path.exists()
            and self.checkpoint_path
            and self.checkpoint_path.exists()
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None and self._score_rows is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.artifact_available,
            "loaded": self.loaded,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": self.device_name,
            "error": self.load_error,
        }

    def _load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if not self.artifact_available:
                raise RuntimeError(f"Stage 3 {self.kind.upper()} artifacts are unavailable: {self.status()}")

            from .pyg_runtime import require_pyg_runtime
            require_pyg_runtime()

            import torch
            from .common import get_device

            if self.kind == "rgcn":
                from .stage3_rgcn import SampledRGCNLinkPredictor, _score_rows
            else:
                from .stage3_hgt import SampledHGTLinkPredictor, _score_rows

            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            checkpoint_target_edge = tuple(
                self.metadata.get("target_edge_type", SharedStage3Graph.TARGET_EDGE_TYPE)
            )
            if checkpoint_target_edge != SharedStage3Graph.TARGET_EDGE_TYPE:
                raise RuntimeError(
                    "Stage 3 checkpoint supervision edge does not match the runtime loader: "
                    f"checkpoint={checkpoint_target_edge}, "
                    f"runtime={SharedStage3Graph.TARGET_EDGE_TYPE}. "
                    "Use the same target_edge_type that was used during training."
                )
            expected_counts = dict(self.metadata.get("node_counts", {}))
            shared = self.shared_graph.load(expected_counts)
            self.device = get_device(self.device_name)

            saved_args = dict(self.metadata.get("args", {}))
            saved_args.update(
                {
                    "device": str(self.device),
                    "batch_size": int(os.getenv("PREDICTION_STAGE3_BATCH_SIZE", "1")),
                    "score_batch_size": int(os.getenv("PREDICTION_STAGE3_BATCH_SIZE", "1")),
                    "num_workers": 0,
                    "pin_memory": False,
                    "amp": False,
                }
            )
            saved_args.setdefault("num_layers", 2)
            saved_args.setdefault("num_neighbors", "10,5")
            saved_args.setdefault("featureless_mode", "type")
            self.args = argparse.Namespace(**saved_args)

            if self.kind == "rgcn":
                relation_to_id = {
                    tuple(key.split("|", 2)): int(value)
                    for key, value in self.metadata["relation_to_id"].items()
                }
                self.model = SampledRGCNLinkPredictor(
                    metadata=shared.data.metadata(),
                    node_counts=shared.node_counts,
                    feature_dims=shared.feature_dims,
                    relation_to_id=relation_to_id,
                    hidden_dim=int(saved_args.get("hidden_dim", 128)),
                    num_layers=int(saved_args.get("num_layers", 2)),
                    num_bases=int(saved_args.get("num_bases", 16)),
                    dropout=float(saved_args.get("dropout", 0.2)),
                    featureless_mode=str(saved_args.get("featureless_mode", "type")),
                ).to(self.device)
            else:
                self.model = SampledHGTLinkPredictor(
                    metadata=shared.data.metadata(),
                    node_counts=shared.node_counts,
                    feature_dims=shared.feature_dims,
                    hidden_dim=int(saved_args.get("hidden_dim", 64)),
                    num_layers=int(saved_args.get("num_layers", 2)),
                    heads=int(saved_args.get("heads", 2)),
                    dropout=float(saved_args.get("dropout", 0.2)),
                    featureless_mode=str(saved_args.get("featureless_mode", "type")),
                ).to(self.device)

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
            self._score_rows = _score_rows
            del state
            del checkpoint
            gc.collect()
            if getattr(self.device, "type", str(self.device)) == "cuda":
                torch.cuda.empty_cache()

    def score_many(self, pairs: list[Any]) -> list[float]:
        if not pairs:
            return []
        with self.shared_graph.inference_lock:
            self._load()
            rows = pd.DataFrame(
                [
                    {
                        "request_index": index,
                        "compound_node_ref": pair.compound_key,
                        "protein_node_ref": pair.target_key,
                        "label": -1,
                    }
                    for index, pair in enumerate(pairs)
                ]
            )
            kept, score_tensor = self._score_rows(
                self.model,
                self.shared_graph.data,
                rows,
                self.shared_graph.node_maps,
                self.args,
                self.device,
            )
            scores_by_index = {
                int(row["request_index"]): float(score)
                for (_, row), score in zip(kept.iterrows(), score_tensor.tolist())
            }
            missing = [index for index in range(len(pairs)) if index not in scores_by_index]
            if missing:
                unresolved = [
                    f"{pairs[index].compound_key} / {pairs[index].target_key}"
                    for index in missing
                ]
                raise RuntimeError(
                    f"Stage 3 {self.kind.upper()} could not map these pairs to node_mapping.csv: {unresolved}"
                )
            return [scores_by_index[index] for index in range(len(pairs))]

    def unload(self) -> None:
        with self._load_lock:
            model = self.model
            self.model = None
            self.args = None
            self.device = None
            self._score_rows = None
            self.metadata = None
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class LiveComponentScorer:
    """Generate production component scores with optional Stage 1 execution."""

    STAGE1_COLUMN = "score__stage1_tabular_extra_trees"
    RGCN_COLUMN = "score__stage3_rgcn_sampled"
    HGT_COLUMN = "score__stage3_hgt_sampled"

    def __init__(self, score_columns: Iterable[str]):
        self.score_columns = [str(column) for column in score_columns]
        expected = {self.STAGE1_COLUMN, self.RGCN_COLUMN, self.HGT_COLUMN}
        if set(self.score_columns) != expected:
            raise ValueError(
                "The hybrid predictor requires the three deployable production scores: "
                f"{sorted(expected)}; received {self.score_columns}."
            )
        self.stage1 = Stage1LiveScorer(
            os.getenv("STAGE1_MODEL_DIR", "/models/improved_v2/stage1_gds_extra_trees"),
            os.getenv("STAGE1_EMBEDDING_PROPERTY", "pringFastRP"),
        )
        self.shared_graph = SharedStage3Graph(os.getenv("STAGE3_PREPARED_DIR", "/modeling_prepared"))
        device = os.getenv("PREDICTION_DEVICE", "cpu")
        self.rgcn = Stage3LiveScorer(
            "rgcn", os.getenv("RGCN_MODEL_DIR", "/models/improved_v2/stage3_rgcn_sampled"), self.shared_graph, device
        )
        self.hgt = Stage3LiveScorer(
            "hgt", os.getenv("HGT_MODEL_DIR", "/models/improved_v2/stage3_hgt_sampled"), self.shared_graph, device
        )
        self.unload_models_after_score = os.getenv(
            "PREDICTION_UNLOAD_STAGE3_AFTER_SCORE", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.unload_graph_after_score = os.getenv(
            "PREDICTION_UNLOAD_SHARED_GRAPH_AFTER_SCORE", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

    def status(self) -> dict[str, Any]:
        try:
            from .pyg_runtime import pyg_runtime_status
            pyg_status = pyg_runtime_status(include_traceback=False)
        except Exception as exc:
            pyg_status = {"ready": False, "error_type": type(exc).__name__, "error": str(exc)}
        stage3_ready = bool(
            self.rgcn.artifact_available and self.hgt.artifact_available and pyg_status.get("ready")
        )
        return {
            "ready": bool(stage3_ready and self.stage1.available),
            "primary_ready": bool(stage3_ready and self.stage1.available),
            "stage3_fallback_ready": stage3_ready,
            "components": {
                "stage1": self.stage1.status(),
                "stage3_rgcn": self.rgcn.status(),
                "stage3_hgt": self.hgt.status(),
            },
            "pytorch_geometric": pyg_status,
            "shared_graph": self.shared_graph.status(),
            "unload_models_after_score": self.unload_models_after_score,
            "unload_graph_after_score": self.unload_graph_after_score,
        }

    def score_many(
        self,
        pairs: list[Any],
        node_property_provider: Callable[[Any], tuple[dict[str, Any], dict[str, Any]]] | None,
        *,
        include_stage1: bool = True,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not pairs:
            return {}
        status = self.status()
        required_ready = status["primary_ready"] if include_stage1 else status["stage3_fallback_ready"]
        if not required_ready:
            raise RuntimeError(f"Live component inference is not ready: {status}")

        stage1_scores = [float("nan")] * len(pairs)
        stage1_features: list[dict[str, float]] = [{} for _ in pairs]
        stage1_sources = ["not_used_by_stage3_fallback"] * len(pairs)
        if include_stage1:
            for index, pair in enumerate(pairs):
                score, features, source = self.stage1.score(pair, node_property_provider)
                stage1_scores[index] = score
                stage1_features[index] = features
                stage1_sources[index] = source

        try:
            rgcn_scores = self.rgcn.score_many(pairs)
        finally:
            if self.unload_models_after_score:
                self.rgcn.unload()
        try:
            hgt_scores = self.hgt.score_many(pairs)
        finally:
            if self.unload_models_after_score:
                self.hgt.unload()

        output: dict[tuple[str, str], dict[str, Any]] = {}
        for index, pair in enumerate(pairs):
            output[(pair.compound_key, pair.target_key)] = {
                "scores": {
                    self.STAGE1_COLUMN: float(stage1_scores[index]),
                    self.RGCN_COLUMN: float(rgcn_scores[index]),
                    self.HGT_COLUMN: float(hgt_scores[index]),
                },
                "details": {
                    "score_source": "live_component_inference",
                    "stage1_pair_features": stage1_features[index],
                    "stage1_feature_source": stage1_sources[index],
                },
            }
        if self.unload_graph_after_score:
            self.shared_graph.unload()
        return output
