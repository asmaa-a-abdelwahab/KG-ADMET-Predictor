from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_smiles_column(df: pd.DataFrame) -> str | None:
    for col in ["smiles", "canonical_smiles", "isomeric_smiles", "compound_smiles", "pubchem_smiles"]:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = col.lower()
        if "smiles" in lower:
            return col
    return None


def add_rdkit_features(
    df: pd.DataFrame,
    *,
    smiles_column: str | None = None,
    prefix: str = "rdkit_",
    fingerprint_bits: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add optional RDKit molecular descriptors/fingerprints when SMILES exist.

    The function is safe when RDKit is not installed; it returns the original
    frame and metadata explaining that descriptor generation was skipped.
    """
    smiles_column = None if smiles_column in {None, "", "auto"} else str(smiles_column)
    smiles_column = smiles_column or infer_smiles_column(df)
    if not smiles_column or smiles_column not in df.columns:
        return df, {"enabled": False, "reason": "no_smiles_column_found", "smiles_column": smiles_column}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception as exc:  # pragma: no cover - optional dependency
        return df, {"enabled": False, "reason": "rdkit_not_available", "error": str(exc), "smiles_column": smiles_column}

    unique_smiles = pd.Series(df[smiles_column].dropna().astype(str).unique())
    records: dict[str, dict[str, float]] = {}
    failed = 0
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        rec: dict[str, float] = {
            f"{prefix}mol_wt": float(Descriptors.MolWt(mol)),
            f"{prefix}logp": float(Crippen.MolLogP(mol)),
            f"{prefix}tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            f"{prefix}hbd": float(Lipinski.NumHDonors(mol)),
            f"{prefix}hba": float(Lipinski.NumHAcceptors(mol)),
            f"{prefix}rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            f"{prefix}rings": float(rdMolDescriptors.CalcNumRings(mol)),
            f"{prefix}aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            f"{prefix}heavy_atoms": float(mol.GetNumHeavyAtoms()),
            f"{prefix}fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }
        if fingerprint_bits and fingerprint_bits > 0:
            bits = int(fingerprint_bits)
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
            arr = np.zeros((bits,), dtype=np.int8)
            # ConvertToNumpyArray is not always imported directly in minimal builds.
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(bv, arr)
            for i, bit in enumerate(arr):
                rec[f"{prefix}morgan2_{bits}_{i}"] = float(bit)
        records[smi] = rec

    if not records:
        return df, {"enabled": False, "reason": "no_valid_smiles", "smiles_column": smiles_column, "failed_smiles": int(failed)}
    feat = pd.DataFrame.from_dict(records, orient="index")
    out = df.copy()
    joined = out[smiles_column].astype(str).map(records)
    feat_rows = pd.DataFrame(list(joined), index=out.index)
    out = pd.concat([out, feat_rows], axis=1)
    return out, {
        "enabled": True,
        "smiles_column": smiles_column,
        "unique_smiles": int(len(unique_smiles)),
        "valid_smiles": int(len(records)),
        "failed_smiles": int(failed),
        "feature_count": int(feat.shape[1]),
        "fingerprint_bits": int(fingerprint_bits or 0),
    }
