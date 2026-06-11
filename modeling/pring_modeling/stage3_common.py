from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.data import HeteroData
except Exception:  # pragma: no cover
    HeteroData = None  # type: ignore

from .common import STAGE3, pick_col, read_pairs, read_table, resolve_stage_dir


def resolve_stage3_dir(modeling_dir: str | Path):
    return resolve_stage_dir(modeling_dir, STAGE3)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _edge_type_from_key(key: Any) -> tuple[str, str, str] | None:
    """Normalize edge-type keys from tuple/list/string payload formats."""
    if isinstance(key, tuple) and len(key) == 3:
        return str(key[0]), str(key[1]), str(key[2])
    if isinstance(key, list) and len(key) == 3:
        return str(key[0]), str(key[1]), str(key[2])
    text = str(key)
    if text.startswith("(") and text.endswith(")"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple) and len(parsed) == 3:
                return str(parsed[0]), str(parsed[1]), str(parsed[2])
        except Exception:
            pass
    for sep in ["__", "|", ":"]:
        parts = text.split(sep)
        if len(parts) == 3:
            return str(parts[0]), str(parts[1]), str(parts[2])
    return None


def _node_feature_file(stage_dir: Path, node_type: str) -> Path | None:
    low = node_type.lower()
    candidates = [
        stage_dir / f"node_features_{low}_tensor.csv",
        stage_dir / f"node_features_{low}_model_matrix.csv",
        stage_dir / f"node_features_{node_type}_tensor.csv",
        stage_dir / f"node_features_{node_type}_model_matrix.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _set_node_store_from_mapping(data: Any, stage_dir: Path) -> None:
    mapping_path = stage_dir / "node_mapping.csv"
    if not mapping_path.exists():
        mapping_path = stage_dir / "pyg_export" / "node_mapping.csv"
    if not mapping_path.exists():
        return
    df = read_table(mapping_path)
    type_col = pick_col(df, ["node_type", "type", "label"], required=False)
    idx_col = pick_col(df, ["local_id", "local_index", "idx", "node_index", "mapped_id"], required=False)
    if not type_col:
        return
    for node_type, sub in df.groupby(type_col):
        node_type = str(node_type)
        if idx_col:
            data[node_type].num_nodes = int(pd.to_numeric(sub[idx_col], errors="coerce").max()) + 1
        else:
            data[node_type].num_nodes = int(len(sub))
        fpath = _node_feature_file(stage_dir, node_type)
        if fpath is not None:
            try:
                feat = read_table(fpath)
                # Keep only numeric feature columns. Exclude IDs/references and all-empty columns.
                numeric = feat.select_dtypes(include=["number", "bool"]).copy()
                for c in list(numeric.columns):
                    if c.lower() in {"node_id", "local_id", "local_index", "idx", "node_index", "mapped_id"}:
                        numeric = numeric.drop(columns=[c])
                if numeric.shape[1] > 0 and numeric.shape[0] == data[node_type].num_nodes:
                    data[node_type].x = torch.tensor(numeric.to_numpy(), dtype=torch.float32)
            except Exception:
                # Features are optional. If they cannot be loaded, Stage 3 falls back to learned node embeddings.
                pass


def _relation_map(stage_dir: Path) -> dict[str, tuple[str, str, str]]:
    path = stage_dir / "relation_mapping.csv"
    if not path.exists():
        path = stage_dir / "pyg_export" / "relation_mapping.csv"
    if not path.exists():
        return {}
    df = read_table(path)
    id_col = pick_col(df, ["relation_id", "rel_id", "edge_type_id", "id", "idx", "mapped_id"], required=False)
    src_col = pick_col(df, ["source_type", "src_type", "head_type", "source_node_type", "start_node_type"], required=False)
    rel_col = pick_col(df, ["relation_type", "rel_type", "relation", "edge_type", "type", "relationship_type"], required=False)
    dst_col = pick_col(df, ["target_type", "dst_type", "tail_type", "target_node_type", "end_node_type"], required=False)
    out: dict[str, tuple[str, str, str]] = {}
    if id_col and src_col and rel_col and dst_col:
        for _, row in df.iterrows():
            out[str(row[id_col])] = (str(row[src_col]), str(row[rel_col]), str(row[dst_col]))
    return out


def _add_edges_from_csv(data: Any, stage_dir: Path) -> bool:
    path = stage_dir / "edge_index_train_only.csv"
    if not path.exists():
        path = stage_dir / "edge_index.csv"
    if not path.exists():
        return False
    df = read_table(path)

    src_col = pick_col(df, ["source_local_id", "src_local_id", "source_idx", "src_idx", "head_idx", "source_index", "src", "source"], required=False)
    dst_col = pick_col(df, ["target_local_id", "dst_local_id", "target_idx", "dst_idx", "tail_idx", "target_index", "dst", "target", "destination"], required=False)
    rel_id_col = pick_col(df, ["relation_id", "rel_id", "edge_type_id", "mapped_relation_id"], required=False)
    src_type_col = pick_col(df, ["source_type", "src_type", "head_type", "source_node_type", "start_node_type"], required=False)
    dst_type_col = pick_col(df, ["target_type", "dst_type", "tail_type", "target_node_type", "end_node_type"], required=False)
    rel_col = pick_col(df, ["relation_type", "rel_type", "relation", "edge_type", "relationship_type"], required=False)

    if not src_col or not dst_col:
        # Some exports store edge_index as columns 0/1/2 without friendly names.
        if df.shape[1] >= 2:
            src_col, dst_col = df.columns[0], df.columns[1]
            if df.shape[1] >= 3 and rel_id_col is None and rel_col is None:
                rel_id_col = df.columns[2]
        else:
            return False

    rel_map = _relation_map(stage_dir)
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for _, row in df.iterrows():
        if src_type_col and dst_type_col and rel_col:
            etype = (str(row[src_type_col]), str(row[rel_col]), str(row[dst_type_col]))
        elif rel_id_col and str(row[rel_id_col]) in rel_map:
            etype = rel_map[str(row[rel_id_col])]
        elif rel_col:
            parsed = _edge_type_from_key(row[rel_col])
            etype = parsed if parsed else ("Node", str(row[rel_col]), "Node")
        elif rel_id_col:
            etype = ("Node", str(row[rel_id_col]), "Node")
        else:
            etype = ("Node", "edge", "Node")
        try:
            grouped.setdefault(etype, []).append((int(row[src_col]), int(row[dst_col])))
        except Exception:
            continue

    for etype, pairs in grouped.items():
        if not pairs:
            continue
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[etype].edge_index = edge_index
        # If node_mapping was unavailable and generic Node type is used, set num_nodes.
        src_t, _, dst_t = etype
        max_src = int(edge_index[0].max()) + 1 if edge_index.numel() else 0
        max_dst = int(edge_index[1].max()) + 1 if edge_index.numel() else 0
        data[src_t].num_nodes = max(getattr(data[src_t], "num_nodes", 0) or 0, max_src)
        data[dst_t].num_nodes = max(getattr(data[dst_t], "num_nodes", 0) or 0, max_dst)
    return bool(grouped)


def _dict_to_heterodata(obj: dict[str, Any], stage_dir: Path | None = None) -> Any | None:
    if HeteroData is None:
        return None

    # Common wrapper keys.
    for key in ["data", "heterodata", "hetero_data", "pyg_data"]:
        if key in obj:
            val = obj[key]
            if isinstance(val, HeteroData):
                return val
            if isinstance(val, dict):
                converted = _dict_to_heterodata(val, stage_dir=stage_dir)
                if converted is not None:
                    return converted

    # Any nested HeteroData value.
    for val in obj.values():
        if isinstance(val, HeteroData):
            return val

    data = HeteroData()

    # Payload style: x_dict / edge_index_dict.
    x_dict = obj.get("x_dict") or obj.get("node_features") or obj.get("features")
    if isinstance(x_dict, dict):
        for node_type, x in x_dict.items():
            data[str(node_type)].x = _as_tensor(x, dtype=torch.float32)
            data[str(node_type)].num_nodes = int(data[str(node_type)].x.size(0))

    num_nodes = obj.get("num_nodes_dict") or obj.get("num_nodes") or obj.get("node_counts")
    if isinstance(num_nodes, dict):
        for node_type, n in num_nodes.items():
            data[str(node_type)].num_nodes = int(n)

    edge_dict = obj.get("edge_index_dict") or obj.get("edge_indices") or obj.get("edges")
    if isinstance(edge_dict, dict):
        for key, edge_index in edge_dict.items():
            etype = _edge_type_from_key(key)
            if etype is None:
                continue
            ei = _as_tensor(edge_index, dtype=torch.long)
            if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
                ei = ei.t().contiguous()
            data[etype].edge_index = ei

    # PyG HeteroData.to_dict() style: {node_type: {x, num_nodes}, (src, rel, dst): {edge_index}}
    for key, value in obj.items():
        if not isinstance(value, dict):
            continue
        etype = _edge_type_from_key(key)
        if etype is not None and "edge_index" in value:
            data[etype].edge_index = _as_tensor(value["edge_index"], dtype=torch.long)
            continue
        if isinstance(key, str) and ("x" in value or "num_nodes" in value):
            if "x" in value and value["x"] is not None:
                data[key].x = _as_tensor(value["x"], dtype=torch.float32)
                data[key].num_nodes = int(data[key].x.size(0))
            if "num_nodes" in value and value["num_nodes"] is not None:
                data[key].num_nodes = int(value["num_nodes"])

    if stage_dir is not None:
        # Use CSV metadata to complete missing node counts/features and train-only edges.
        _set_node_store_from_mapping(data, stage_dir)
        if len(data.edge_types) == 0:
            _add_edges_from_csv(data, stage_dir)

    if len(data.node_types) > 0 and len(data.edge_types) > 0:
        return data
    return None


def _load_heterodata_from_csv(stage_dir: Path) -> Any | None:
    if HeteroData is None:
        return None
    data = HeteroData()
    _set_node_store_from_mapping(data, stage_dir)
    ok = _add_edges_from_csv(data, stage_dir)
    if ok and len(data.node_types) > 0 and len(data.edge_types) > 0:
        return data
    return None


def load_heterodata(stage_dir: Path):
    """Load Stage 3 HeteroData from multiple PRING export variants.

    Some PRING runs save ``heterodata.pt`` as a raw ``HeteroData`` object, while
    others save a dictionary payload or a tensor-only payload. This loader accepts
    both. If the PyG object cannot be reconstructed directly from the .pt file, it
    falls back to the CSV exports, preferring ``edge_index_train_only.csv`` to keep
    validation/test interaction evidence leakage-safe.
    """
    if HeteroData is None:
        raise RuntimeError("PyTorch Geometric is required for Stage 3. Install torch-geometric matching your PyTorch build.")

    candidates = [
        stage_dir / "pyg_export" / "heterodata.pt",
        stage_dir / "pyg_export" / "heterodata_payload.pt",
        stage_dir / "heterodata.pt",
    ]
    errors: list[str] = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            obj = _torch_load(p)
            if isinstance(obj, HeteroData):
                return obj
            if isinstance(obj, dict):
                converted = _dict_to_heterodata(obj, stage_dir=stage_dir)
                if converted is not None:
                    return converted
                errors.append(f"{p}: dict keys={list(obj.keys())[:25]}")
            else:
                errors.append(f"{p}: unsupported type={type(obj)}")
        except Exception as exc:
            errors.append(f"{p}: {type(exc).__name__}: {exc}")

    csv_data = _load_heterodata_from_csv(stage_dir)
    if csv_data is not None:
        return csv_data

    detail = "\n".join(errors) if errors else "No candidate .pt files were found."
    raise TypeError(
        "Could not load/reconstruct Stage 3 HeteroData. Tried .pt payloads and CSV fallback.\n"
        f"Stage directory: {stage_dir}\n{detail}"
    )


def load_node_maps(stage_dir: Path) -> dict[str, dict[str, int]]:
    path = stage_dir / "node_mapping.csv"
    if not path.exists():
        path = stage_dir / "pyg_export" / "node_mapping.csv"
    df = read_table(path)
    type_col = pick_col(df, ["node_type", "type", "label"])
    id_col = pick_col(df, ["node_id", "node_key", "element_id", "eid", "id", "node_ref"])
    idx_col = pick_col(df, ["local_id", "local_index", "idx", "node_index", "mapped_id"])
    maps: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        maps.setdefault(str(row[type_col]), {})[str(row[id_col])] = int(row[idx_col])
    return maps


def load_pairs(stage_dir: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_candidates = [
        stage_dir / "compound_target_link_prediction_pairs.csv",
        stage_dir / "pyg_export" / "compound_target_link_prediction_pairs.csv",
    ]
    split_path = next((p for p in split_candidates if p.exists()), None)
    if split_path:
        df = read_pairs(split_path)
        split_col = pick_col(df, ["split", "set", "partition"], required=False)
        if split_col:
            split = df[split_col].astype(str).str.lower()
            train = df[split.isin(["train", "training"])].copy()
            valid = df[split.isin(["valid", "validation", "val"])].copy()
            test = df[split.eq("test")].copy()
            if len(train) and len(valid) and len(test):
                return train, valid, test
        y_col = pick_col(df, ["label", "y", "is_positive", "positive"])
        stratify = df[y_col] if df[y_col].nunique() == 2 else None
        train, tmp = train_test_split(df, test_size=0.30, stratify=stratify, random_state=seed)
        valid, test = train_test_split(tmp, test_size=0.50, stratify=tmp[y_col] if tmp[y_col].nunique() == 2 else None, random_state=seed)
        return train.copy(), valid.copy(), test.copy()

    pos_path = stage_dir / "positive_compound_target_pairs.csv"
    neg_path = stage_dir / "negative_compound_target_pairs.csv"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"No Stage 3 pair file found under {stage_dir}")
    pos = read_pairs(pos_path); neg = read_pairs(neg_path)
    pos["label"] = 1; neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    valid, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=seed)
    return train.copy(), valid.copy(), test.copy()


def load_candidate_pairs(stage_dir: Path) -> pd.DataFrame | None:
    for name in [
        "candidate_missing_pairs_observed_compounds_only.csv",
        "candidate_missing_pairs_all_materialized_compounds.csv",
        "candidate_missing_compound_target_pairs.csv",
        "candidate_compound_target_pairs.csv",
        "compound_target_candidate_pairs.csv",
        "candidate_pairs.csv",
    ]:
        p = stage_dir / name
        if p.exists():
            return read_pairs(p)
    return None


def encode_pairs(df: pd.DataFrame, node_maps: dict[str, dict[str, int]]) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, pd.DataFrame]:
    c_col = pick_col(df, ["compound_node_id", "compound_eid", "compound_id", "compound_node_ref", "source", "head"])
    p_col = pick_col(df, ["protein_node_id", "protein_eid", "target_eid", "protein_id", "protein_node_ref", "target_id", "destination", "tail"])
    y_col = pick_col(df, ["label", "y", "is_positive", "positive"], required=False)
    compound_map = node_maps["Compound"]
    protein_map = node_maps["Protein"]
    c_idx, p_idx, y, kept = [], [], [], []
    for idx, row in df.iterrows():
        c_candidates = [str(row.get(c_col, "")), str(row.get("compound_node_ref", "")), str(row.get("compound_node_id", ""))]
        p_candidates = [str(row.get(p_col, "")), str(row.get("protein_node_ref", "")), str(row.get("protein_node_id", ""))]
        c_key = next((v for v in c_candidates if v in compound_map), None)
        p_key = next((v for v in p_candidates if v in protein_map), None)
        if c_key is None or p_key is None:
            continue
        c_idx.append(compound_map[c_key]); p_idx.append(protein_map[p_key])
        y.append(float(row[y_col]) if y_col else -1.0)
        kept.append(idx)
    if not kept:
        raise ValueError("No pairs could be mapped to local Compound/Protein indices.")
    return torch.tensor(c_idx, dtype=torch.long), torch.tensor(p_idx, dtype=torch.long), torch.tensor(y, dtype=torch.float32), df.loc[kept].copy().reset_index(drop=True)
