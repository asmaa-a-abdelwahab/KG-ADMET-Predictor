from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import STAGE1, binary_metrics, ensure_dir, get_device, pick_col, read_pairs, read_table, resolve_stage_dir, save_json, set_seed, tensor_from_embedding_value
from .neo4j_export import export_predictions_dataframe


class PairMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def forward(self, c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, p, torch.abs(c - p), c * p], dim=-1)
        return self.net(x).squeeze(-1)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    df = read_table(path)
    id_col = pick_col(df, ["node_id", "nodeId", "element_id", "eid", "id", "node", "entity_id", "node_ref"])
    emb_col = pick_col(df, ["embedding", "fastrp", "graphsage", "features", "vector"])
    out: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        vec = np.asarray(tensor_from_embedding_value(row[emb_col]), dtype=np.float32)
        if vec.size:
            out[str(row[id_col])] = vec
    if not out:
        raise ValueError(f"No embeddings parsed from {path}")
    return out


def build_dataset(pairs: pd.DataFrame, emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    c_col = pick_col(pairs, ["compound_node_id", "compound_eid", "compound_id", "compound_node_ref", "source", "head"])
    p_col = pick_col(pairs, ["protein_node_id", "protein_eid", "target_eid", "protein_id", "protein_node_ref", "target_id", "destination", "tail"])
    y_col = pick_col(pairs, ["label", "y", "is_positive", "positive"])
    c_vecs, p_vecs, ys, kept_rows = [], [], [], []
    for idx, row in pairs.iterrows():
        c_id = str(row[c_col])
        p_id = str(row[p_col])
        # Try exact ID, node_ref, and numeric-string-normalized alternatives.
        c_candidates = [c_id, str(row.get("compound_node_ref", "")), str(row.get("compound_node_id", ""))]
        p_candidates = [p_id, str(row.get("protein_node_ref", "")), str(row.get("protein_node_id", ""))]
        c_key = next((x for x in c_candidates if x in emb), None)
        p_key = next((x for x in p_candidates if x in emb), None)
        if c_key is None or p_key is None:
            continue
        c_vecs.append(emb[c_key])
        p_vecs.append(emb[p_key])
        ys.append(float(row[y_col]))
        kept_rows.append(idx)
    if not ys:
        raise ValueError("No pair rows could be matched to embeddings. Check embedding node IDs and pair IDs.")
    return np.stack(c_vecs), np.stack(p_vecs), np.asarray(ys, dtype=np.float32), pairs.loc[kept_rows].copy()


def train_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for c, p, y in loader:
        c, p, y = c.to(device), p.to(device), y.to(device)
        optim.zero_grad()
        loss = loss_fn(model(c, p), y)
        loss.backward()
        optim.step()
        total += float(loss.item()) * y.numel()
    return total / max(1, len(loader.dataset))


def predict(model, c, p, device, batch_size=4096):
    model.eval()
    scores = []
    ds = TensorDataset(torch.tensor(c, dtype=torch.float32), torch.tensor(p, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for cb, pb in loader:
            logits = model(cb.to(device), pb.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    resolved = resolve_stage_dir(args.modeling_dir, STAGE1)
    out_dir = ensure_dir(args.output_dir)
    try:
        stage_dir = resolved.root
        pairs_path = stage_dir / "compound_target_training_pairs_for_gds.csv"
        if not pairs_path.exists():
            hits = sorted(stage_dir.glob("*training*pair*.csv"))
            if not hits:
                raise FileNotFoundError(f"Could not find Stage 1 training pairs in {stage_dir}")
            pairs_path = hits[0]
        pairs = read_pairs(pairs_path)
        emb = load_embeddings(args.embedding_csv)
        c, p, y, kept_pairs = build_dataset(pairs, emb)

        stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 2 else None
        idx_train, idx_tmp = train_test_split(np.arange(len(y)), test_size=0.30, stratify=stratify, random_state=args.seed)
        idx_valid, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp] if stratify is not None and min(np.bincount(y[idx_tmp].astype(int))) >= 2 else None, random_state=args.seed)

        train_ds = TensorDataset(torch.tensor(c[idx_train], dtype=torch.float32), torch.tensor(p[idx_train], dtype=torch.float32), torch.tensor(y[idx_train], dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        device = get_device(args.device)
        model = PairMLP(c.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_ap = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optim, device)
            val_score = predict(model, c[idx_valid], p[idx_valid], device, args.batch_size)
            val_metrics = binary_metrics(y[idx_valid], val_score, threshold=args.threshold)
            history.append({"epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in val_metrics.items()}})
            ap = val_metrics.get("average_precision") or -1.0
            if ap > best_ap:
                best_ap = float(ap)
                torch.save({"model_state": model.state_dict(), "args": vars(args), "embedding_dim": int(c.shape[1])}, out_dir / "best_model.pt")
            print(f"epoch={epoch:03d} loss={loss:.4f} valid={val_metrics}")

        ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_score = predict(model, c[idx_test], p[idx_test], device, args.batch_size)
        metrics = binary_metrics(y[idx_test], test_score, threshold=args.threshold)
        preds = kept_pairs.iloc[idx_test].copy().reset_index(drop=True)
        preds["score"] = test_score
        preds["predicted_label"] = (test_score >= args.threshold).astype(int)
        preds["model"] = "stage1_gds_embedding_mlp"
        preds["stage"] = "Stage 1 — Neo4j GDS embedding MLP"
        preds.to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        summary = {"stage": "Stage 1 — Neo4j GDS embedding MLP", "model": "stage1_gds_embedding_mlp", "metrics": metrics, "pairs_used": int(len(kept_pairs)), "predictions_file": str(out_dir / "predictions.csv")}
        if args.export_neo4j:
            summary["neo4j_export"] = export_predictions_dataframe(preds, model_name="stage1_gds_embedding_mlp", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP baseline on Neo4j GDS embeddings.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--embedding-csv", required=True, help="CSV exported from GDS with node_id/node_ref and embedding columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
