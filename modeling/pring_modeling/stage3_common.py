from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.data import HeteroData
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch Geometric is required for Stage 3. Install torch-geometric matching your PyTorch build.") from exc

from .common import STAGE3, pick_col, read_pairs, read_table, resolve_stage_dir


def resolve_stage3_dir(modeling_dir: str | Path):
    return resolve_stage_dir(modeling_dir, STAGE3)


def load_heterodata(stage_dir: Path) -> HeteroData:
    candidates = [
        stage_dir / "pyg_export" / "heterodata.pt",
        stage_dir / "pyg_export" / "heterodata_payload.pt",
        stage_dir / "heterodata.pt",
    ]
    for p in candidates:
        if p.exists():
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(p, map_location="cpu")
            if isinstance(obj, HeteroData):
                return obj
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], HeteroData):
                return obj["data"]
            raise TypeError(f"Unsupported HeteroData payload in {p}: {type(obj)}")
    raise FileNotFoundError(f"No HeteroData export found under {stage_dir}. Expected pyg_export/heterodata.pt")


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
    for name in ["candidate_compound_target_pairs.csv", "compound_target_candidate_pairs.csv", "candidate_pairs.csv"]:
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
