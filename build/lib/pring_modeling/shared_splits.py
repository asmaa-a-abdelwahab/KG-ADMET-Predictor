"""Shared split preparation utilities for PRING modeling comparisons.

This module creates one canonical train/valid/test split manifest and materializes
that split into the supervised pair CSV files used by Stage 1, Stage 2 and Stage 3.
It is intentionally implementation-neutral so legacy, improved and improved_v2 can
be evaluated on exactly the same rows.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin1", low_memory=False)


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower.get(c.lower())
        if hit is not None:
            return hit
    return None


def _label_series(df: pd.DataFrame) -> pd.Series | None:
    c = _first_existing(df, ["label", "_label", "y", "is_positive", "positive", "activity_label"])
    if c is None:
        return None
    raw = df[c]
    if pd.api.types.is_numeric_dtype(raw):
        y = pd.to_numeric(raw, errors="coerce")
    else:
        s = raw.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1, "1.0": 1, "true": 1, "yes": 1, "positive": 1, "pos": 1,
            "active": 1, "inhibitor": 1, "inhibited": 1,
            "0": 0, "0.0": 0, "false": 0, "no": 0, "negative": 0, "neg": 0,
            "inactive": 0, "nonactive": 0, "non-active": 0, "not active": 0,
        }
        y = s.map(mapping)
        y = pd.to_numeric(y, errors="coerce")
    return y.where(y.isin([0, 1]))


def _compound_key(df: pd.DataFrame) -> pd.Series | None:
    c = _first_existing(df, [
        "compound_node_id", "compound_id", "compound_cid", "cid", "pubchem_cid",
        "compound_node_ref", "compound_entity_id", "compound", "head", "source",
        "source_id", "src", "start_id",
    ])
    if c is None:
        return None
    s = df[c].astype(str).str.strip()
    if c.endswith("node_id") or c in {"compound_id", "compound_cid", "cid", "pubchem_cid"}:
        s = s.str.replace(r"^n", "", regex=True)
    return s


def _target_key(df: pd.DataFrame) -> pd.Series | None:
    c = _first_existing(df, [
        "protein_node_id", "target_node_id", "gene_node_id", "protein_id", "target_id", "gene_id",
        "protein_node_ref", "target_node_ref", "gene_node_ref", "protein_entity_id", "target_entity_id",
        "protein", "target", "gene", "tail", "destination", "target_name", "end_id",
    ])
    if c is None:
        return None
    s = df[c].astype(str).str.strip()
    if c.endswith("node_id") or c in {"protein_id", "target_id", "gene_id"}:
        s = s.str.replace(r"^n", "", regex=True)
    return s


def _scaffold_or_group_key(df: pd.DataFrame, compound: pd.Series, strategy: str) -> pd.Series:
    strategy = strategy.lower()
    if strategy == "pair":
        target = _target_key(df)
        if target is not None:
            return compound.astype(str) + "__" + target.astype(str)
        return compound.astype(str)
    if strategy in {"scaffold", "murcko", "bemis_murcko"}:
        scaf_col = _first_existing(df, ["scaffold", "murcko_scaffold", "bemis_murcko_scaffold", "scaffold_key"])
        if scaf_col is not None and df[scaf_col].notna().any():
            return df[scaf_col].astype(str).fillna(compound.astype(str))
        # The robust fallback keeps whole compounds together. RDKit scaffold
        # generation can be added upstream when SMILES are available.
        return compound.astype(str)
    source_col = _first_existing(df, ["source_group", "source", "assay_id", "bioassay_id", "reference_id", "pmid", "split_group"])
    if strategy in {"source", "assay", "reference"} and source_col is not None:
        return df[source_col].astype(str).fillna(compound.astype(str))
    return compound.astype(str)


def _candidate_pair_files(modeling_dir: Path) -> list[Path]:
    names = [
        "compound_target_training_pairs_gds_features.csv",
        "compound_target_training_pairs_for_gds.csv",
        "compound_target_training_pairs.csv",
        "positive_negative_compound_target_pairs.csv",
        "compound_target_link_prediction_pairs.csv",
        "training_pairs.csv",
        "link_prediction_pairs.csv",
    ]
    roots = [
        modeling_dir,
        modeling_dir / "stage1_neo4j_gds_baselines",
        modeling_dir / "stage3_heterogeneous_gnn",
        modeling_dir / "stage3_heterogeneous_gnn" / "pyg_export",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            p = root / name
            if p.exists() and p.is_file():
                files.append(p)
        for p in sorted(root.glob("*pair*.csv")):
            if p.is_file():
                files.append(p)
    out: list[Path] = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def _find_base_pair_file(modeling_dir: Path) -> Path:
    preferred = [
        modeling_dir / "stage1_neo4j_gds_baselines" / "compound_target_training_pairs_gds_features.csv",
        modeling_dir / "stage3_heterogeneous_gnn" / "compound_target_link_prediction_pairs.csv",
        modeling_dir / "stage3_heterogeneous_gnn" / "pyg_export" / "compound_target_link_prediction_pairs.csv",
        modeling_dir / "stage1_neo4j_gds_baselines" / "compound_target_training_pairs_for_gds.csv",
        modeling_dir / "compound_target_training_pairs.csv",
    ]
    for p in preferred:
        if p.exists():
            df = _read_csv(p, nrows=100)
            if _label_series(df) is not None and _compound_key(df) is not None and _target_key(df) is not None:
                return p
    for p in _candidate_pair_files(modeling_dir):
        df = _read_csv(p, nrows=100)
        if _label_series(df) is not None and _compound_key(df) is not None and _target_key(df) is not None:
            return p
    raise FileNotFoundError(f"Could not find a labelled compound-target pair CSV under {modeling_dir}")


def _make_split(labels: pd.Series, groups: pd.Series, *, seed: int, test_size: float, valid_size: float) -> pd.Series:
    labels = labels.astype(int).reset_index(drop=True)
    groups = groups.astype(str).reset_index(drop=True)
    n = len(labels)
    idx = np.arange(n)
    split = pd.Series("train", index=np.arange(n), dtype=object)

    use_group = groups.nunique() > 1 and groups.nunique() < n
    try:
        if use_group:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_valid_idx, test_idx = next(gss.split(idx, labels, groups=groups))
            split.iloc[test_idx] = "test"
            remaining_groups = groups.iloc[train_valid_idx]
            relative_valid = valid_size / max(1e-9, 1.0 - test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_valid, random_state=seed + 1)
            train_idx_local, valid_idx_local = next(gss2.split(train_valid_idx, labels.iloc[train_valid_idx], groups=remaining_groups))
            valid_idx = train_valid_idx[valid_idx_local]
            split.iloc[valid_idx] = "valid"
        else:
            stratify = labels if labels.value_counts().min() >= 2 else None
            train_valid_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
            split.iloc[test_idx] = "test"
            rem_labels = labels.iloc[train_valid_idx]
            rem_stratify = rem_labels if rem_labels.value_counts().min() >= 2 else None
            relative_valid = valid_size / max(1e-9, 1.0 - test_size)
            train_idx, valid_idx = train_test_split(train_valid_idx, test_size=relative_valid, random_state=seed + 1, stratify=rem_stratify)
            split.iloc[valid_idx] = "valid"
    except Exception:
        # Safe deterministic fallback if stratified/group splitting is impossible.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        n_test = max(1, int(round(n * test_size)))
        n_valid = max(1, int(round(n * valid_size)))
        split.iloc[perm[:n_test]] = "test"
        split.iloc[perm[n_test:n_test + n_valid]] = "valid"
    return split


def build_manifest(modeling_dir: Path, *, strategy: str, seed: int, test_size: float, valid_size: float) -> tuple[pd.DataFrame, Path]:
    base = _find_base_pair_file(modeling_dir)
    df = _read_csv(base)
    y = _label_series(df)
    compound = _compound_key(df)
    target = _target_key(df)
    if y is None or compound is None or target is None:
        raise ValueError(f"Base pair file lacks label/compound/target columns: {base}")
    keep = y.isin([0, 1]) & compound.notna() & target.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).reset_index(drop=True)
    compound = compound.loc[keep].reset_index(drop=True)
    target = target.loc[keep].reset_index(drop=True)
    group = _scaffold_or_group_key(df.reset_index(drop=True), compound, strategy)
    split = _make_split(y, group, seed=seed, test_size=test_size, valid_size=valid_size)
    manifest = pd.DataFrame({
        "compound_key": compound.astype(str),
        "target_key": target.astype(str),
        "label": y.astype(int),
        "split": split.astype(str),
        "split_group": group.astype(str),
    }).drop_duplicates(["compound_key", "target_key", "label"], keep="first")
    return manifest, base


def _copy_symlink_tree(source: Path, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source):
        root_p = Path(root)
        rel = root_p.relative_to(source)
        dest_root = dest / rel
        dest_root.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (dest_root / d).mkdir(exist_ok=True)
        for f in files:
            src = root_p / f
            dst = dest_root / f
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)


def _apply_manifest_to_file(src_file: Path, dest_file: Path, manifest: pd.DataFrame) -> dict:
    df = _read_csv(src_file)
    y = _label_series(df)
    compound = _compound_key(df)
    target = _target_key(df)
    result = {"file": str(dest_file), "rows": int(len(df)), "matched_rows": 0, "applied": False}
    if y is None or compound is None or target is None:
        return result
    left = pd.DataFrame({
        "__row": np.arange(len(df)),
        "compound_key": compound.astype(str),
        "target_key": target.astype(str),
        "label": y,
    })
    left = left[left["label"].isin([0, 1])].copy()
    left["label"] = left["label"].astype(int)
    merged = left.merge(manifest, on=["compound_key", "target_key", "label"], how="left", suffixes=("", "__shared"))
    matched = merged["split"].notna()
    if matched.sum() == 0:
        return result
    result["matched_rows"] = int(matched.sum())
    split_map = pd.Series(merged.loc[matched, "split"].values, index=merged.loc[matched, "__row"].astype(int))
    group_map = pd.Series(merged.loc[matched, "split_group"].values, index=merged.loc[matched, "__row"].astype(int))
    if "split" in df.columns:
        df["original_split"] = df["split"]
    df.loc[split_map.index, "split"] = split_map
    df.loc[split_map.index, "split_group"] = group_map
    df.loc[split_map.index, "stage_use"] = split_map
    if dest_file.exists() or dest_file.is_symlink():
        dest_file.unlink()
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_file, index=False)
    result["applied"] = True
    return result


def prepare_shared_splits(source_modeling_dir: Path, output_dir: Path, prepared_modeling_dir: Path, *, strategy: str, seed: int, test_size: float, valid_size: float, force: bool = False) -> dict:
    source_modeling_dir = source_modeling_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, base = build_manifest(source_modeling_dir, strategy=strategy, seed=seed, test_size=test_size, valid_size=valid_size)
    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    _copy_symlink_tree(source_modeling_dir, prepared_modeling_dir, force=force)
    apply_results = []
    for src in _candidate_pair_files(source_modeling_dir):
        dest = prepared_modeling_dir / src.relative_to(source_modeling_dir)
        try:
            apply_results.append(_apply_manifest_to_file(src, dest, manifest))
        except Exception as exc:
            apply_results.append({"file": str(dest), "error": str(exc), "applied": False})

    summary = {
        "source_modeling_dir": str(source_modeling_dir),
        "prepared_modeling_dir": str(prepared_modeling_dir),
        "base_pair_file": str(base),
        "manifest_file": str(manifest_path),
        "strategy": strategy,
        "seed": int(seed),
        "test_size": float(test_size),
        "valid_size": float(valid_size),
        "rows": int(len(manifest)),
        "label_counts": manifest["label"].value_counts(dropna=False).sort_index().astype(int).to_dict(),
        "split_counts": manifest["split"].value_counts(dropna=False).to_dict(),
        "split_label_counts": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).astype(int).to_dict(),
        "materialized_files": apply_results,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    return summary


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# PRING shared split manifest", "",
        f"- Source modeling dir: `{summary['source_modeling_dir']}`",
        f"- Prepared modeling dir: `{summary['prepared_modeling_dir']}`",
        f"- Base pair file: `{summary['base_pair_file']}`",
        f"- Manifest file: `{summary['manifest_file']}`",
        f"- Strategy: `{summary['strategy']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Rows: `{summary['rows']}`", "",
        "## Split counts", "",
    ]
    for k, v in summary.get("split_counts", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(["", "## Materialized files", ""])
    for item in summary.get("materialized_files", []):
        if item.get("applied"):
            lines.append(f"- applied `{item['file']}` matched_rows={item.get('matched_rows')}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create/apply shared data splits for PRING model comparisons.")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="Create manifest and materialized modeling directory.")
    prep.add_argument("--source-modeling-dir", required=True, type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    prep.add_argument("--prepared-modeling-dir", required=True, type=Path)
    prep.add_argument("--strategy", default="compound", choices=["compound", "pair", "scaffold", "source", "assay", "reference"])
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--test-size", type=float, default=0.15)
    prep.add_argument("--valid-size", type=float, default=0.15)
    prep.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare":
        summary = prepare_shared_splits(
            args.source_modeling_dir,
            args.output_dir,
            args.prepared_modeling_dir,
            strategy=args.strategy,
            seed=args.seed,
            test_size=args.test_size,
            valid_size=args.valid_size,
            force=args.force,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
