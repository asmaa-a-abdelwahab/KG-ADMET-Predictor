from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import STAGE2, attach_refs_from_entities, binary_metrics, ensure_dir, get_device, read_table, read_triples, resolve_stage_dir, save_json, set_seed, sigmoid_np
from .neo4j_export import export_predictions_dataframe


class KGDataset:
    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None, test_df: pd.DataFrame | None = None):
        self.train_df = train_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True)
        self.valid_df = valid_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True) if valid_df is not None else None
        self.test_df = test_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True) if test_df is not None else None
        all_df = pd.concat([x for x in [self.train_df, self.valid_df, self.test_df] if x is not None], ignore_index=True)
        entities = sorted(set(all_df["head"]).union(set(all_df["tail"])))
        relations = sorted(set(all_df["relation"]))
        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.relation_to_id = {r: i for i, r in enumerate(relations)}

    def encode(self, df: pd.DataFrame) -> torch.LongTensor:
        enc = df[["head", "relation", "tail"]].astype(str).copy()
        enc = enc[enc["head"].isin(self.entity_to_id) & enc["tail"].isin(self.entity_to_id) & enc["relation"].isin(self.relation_to_id)]
        arr = np.column_stack([
            enc["head"].map(self.entity_to_id).to_numpy(),
            enc["relation"].map(self.relation_to_id).to_numpy(),
            enc["tail"].map(self.entity_to_id).to_numpy(),
        ]).astype(np.int64)
        return torch.from_numpy(arr)


class DistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, h, r, t):
        return (self.entity(h) * self.relation(r) * self.entity(t)).sum(dim=-1)


class ComplEx(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, dim)
        self.entity_im = nn.Embedding(num_entities, dim)
        self.relation_re = nn.Embedding(num_relations, dim)
        self.relation_im = nn.Embedding(num_relations, dim)
        for emb in [self.entity_re, self.entity_im, self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h, r, t):
        hr, hi = self.entity_re(h), self.entity_im(h)
        rr, ri = self.relation_re(r), self.relation_im(r)
        tr, ti = self.entity_re(t), self.entity_im(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)


class RotatE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, margin: float = 6.0):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, dim)
        self.entity_im = nn.Embedding(num_entities, dim)
        self.relation_phase = nn.Embedding(num_relations, dim)
        self.margin = margin
        for emb in [self.entity_re, self.entity_im, self.relation_phase]:
            nn.init.uniform_(emb.weight, -0.01, 0.01)

    def score(self, h, r, t):
        hr, hi = self.entity_re(h), self.entity_im(h)
        tr, ti = self.entity_re(t), self.entity_im(t)
        phase = self.relation_phase(r)
        rr, ri = torch.cos(phase), torch.sin(phase)
        rot_re = hr * rr - hi * ri
        rot_im = hr * ri + hi * rr
        dist = torch.sqrt((rot_re - tr).pow(2) + (rot_im - ti).pow(2) + 1e-9).sum(dim=-1)
        return self.margin - dist


def corrupt_triples(pos: torch.LongTensor, num_entities: int) -> torch.LongTensor:
    neg = pos.clone()
    mask = torch.rand(len(pos), device=pos.device) < 0.5
    random_entities = torch.randint(0, num_entities, (len(pos),), device=pos.device)
    neg[mask, 0] = random_entities[mask]
    neg[~mask, 2] = random_entities[~mask]
    return neg


def train_epoch(model, loader, optim, num_entities: int, device):
    model.train()
    total = 0.0
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    for (pos,) in loader:
        pos = pos.to(device)
        neg = corrupt_triples(pos, num_entities)
        optim.zero_grad()
        pos_score = model.score(pos[:, 0], pos[:, 1], pos[:, 2])
        neg_score = model.score(neg[:, 0], neg[:, 1], neg[:, 2])
        loss = loss_fn(pos_score, neg_score, torch.ones_like(pos_score))
        loss.backward()
        optim.step()
        total += float(loss.item()) * len(pos)
    return total / max(1, len(loader.dataset))


def evaluate_auc_like(model, triples: torch.LongTensor, num_entities: int, device, negatives_per_positive: int = 1) -> dict[str, float | None]:
    if len(triples) == 0:
        return {"roc_auc": None, "average_precision": None}
    model.eval()
    triples = triples.to(device)
    scores, labels = [], []
    with torch.no_grad():
        pos_score = model.score(triples[:, 0], triples[:, 1], triples[:, 2])
        scores.append(pos_score.cpu())
        labels.append(torch.ones_like(pos_score).cpu())
        for _ in range(negatives_per_positive):
            neg = corrupt_triples(triples, num_entities)
            neg_score = model.score(neg[:, 0], neg[:, 1], neg[:, 2])
            scores.append(neg_score.cpu())
            labels.append(torch.zeros_like(neg_score).cpu())
    return binary_metrics(torch.cat(labels).numpy(), torch.cat(scores).numpy())


def _read_triples_limited(path: Path, nrows: int = 0) -> pd.DataFrame:
    if nrows and nrows > 0:
        # Header-aware read.
        head = pd.read_csv(path, sep="\t", nrows=0)
        if {"head", "relation", "tail"}.issubset(head.columns):
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)[["head", "relation", "tail"]].astype(str)
        return pd.read_csv(path, sep="\t", header=None, nrows=nrows, names=["head", "relation", "tail"], usecols=[0, 1, 2], low_memory=False).astype(str)
    return read_triples(path)


def find_stage2_files(stage_dir: Path) -> dict[str, Path | None]:
    pykeen = stage_dir / "pykeen"
    return {
        "graph_train": (pykeen / "train.tsv") if (pykeen / "train.tsv").exists() else (stage_dir / "train_graph_triples_leakage_safe.tsv" if (stage_dir / "train_graph_triples_leakage_safe.tsv").exists() else stage_dir / "all_graph_triples.tsv"),
        "target_train": (stage_dir / "target_relation_train.tsv") if (stage_dir / "target_relation_train.tsv").exists() else None,
        "valid": (pykeen / "valid.tsv") if (pykeen / "valid.tsv").exists() else (stage_dir / "target_relation_valid.tsv" if (stage_dir / "target_relation_valid.tsv").exists() else None),
        "test": (pykeen / "test.tsv") if (pykeen / "test.tsv").exists() else (stage_dir / "target_relation_test.tsv" if (stage_dir / "target_relation_test.tsv").exists() else None),
        "candidate": (pykeen / "candidates_to_score.tsv") if (pykeen / "candidates_to_score.tsv").exists() else (stage_dir / "candidate_target_triples_to_score.tsv" if (stage_dir / "candidate_target_triples_to_score.tsv").exists() else None),
        "entities": stage_dir / "entities.tsv" if (stage_dir / "entities.tsv").exists() else None,
    }



def read_entities_subset(path: Path, entity_ids: set[str]) -> pd.DataFrame | None:
    if not path or not path.exists() or not entity_ids:
        return None
    rows = []
    for chunk in pd.read_csv(path, sep="\t", chunksize=200000, low_memory=False):
        if "entity_id" not in chunk.columns:
            return read_table(path)
        mask = chunk["entity_id"].astype(str).isin(entity_ids)
        if mask.any():
            rows.append(chunk.loc[mask].copy())
    if rows:
        return pd.concat(rows, ignore_index=True)
    return None

def score_triples_dataframe(model, dataset: KGDataset, triples_df: pd.DataFrame, device, batch_size: int = 65536) -> pd.DataFrame:
    rows, encoded = [], []
    for _, row in triples_df.iterrows():
        h, r, t = str(row["head"]), str(row["relation"]), str(row["tail"])
        if h not in dataset.entity_to_id or t not in dataset.entity_to_id or r not in dataset.relation_to_id:
            continue
        rows.append({"head": h, "relation": r, "tail": t})
        encoded.append([dataset.entity_to_id[h], dataset.relation_to_id[r], dataset.entity_to_id[t]])
    if not encoded:
        return pd.DataFrame(columns=["head", "relation", "tail", "raw_score", "score"])
    arr = torch.tensor(encoded, dtype=torch.long)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            batch = arr[start:start + batch_size].to(device)
            scores.append(model.score(batch[:, 0], batch[:, 1], batch[:, 2]).cpu())
    raw = torch.cat(scores).numpy()
    out = pd.DataFrame(rows)
    out["raw_score"] = raw
    out["score"] = sigmoid_np(raw)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    resolved = resolve_stage_dir(args.modeling_dir, STAGE2)
    out_dir = ensure_dir(args.output_dir)
    try:
        stage_dir = resolved.root
        files = find_stage2_files(stage_dir)
        if not files["graph_train"] or not Path(files["graph_train"]).exists():
            raise FileNotFoundError(f"No Stage 2 train triples found under {stage_dir}")
        graph_train = _read_triples_limited(Path(files["graph_train"]), args.max_graph_train_triples)
        train_parts = [graph_train]
        if files["target_train"]:
            train_parts.append(read_triples(Path(files["target_train"])))
        train_df = pd.concat(train_parts, ignore_index=True).drop_duplicates()
        valid_df = read_triples(Path(files["valid"])) if files["valid"] else None
        test_df = read_triples(Path(files["test"])) if files["test"] else None
        dataset = KGDataset(train_df, valid_df, test_df)
        train_triples = dataset.encode(dataset.train_df)
        valid_triples = dataset.encode(dataset.valid_df) if dataset.valid_df is not None else None
        test_triples = dataset.encode(dataset.test_df) if dataset.test_df is not None else None
        if len(train_triples) == 0:
            raise ValueError("No train triples could be encoded.")

        if args.model == "distmult":
            model = DistMult(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim)
        elif args.model == "complex":
            model = ComplEx(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim)
        else:
            model = RotatE(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim, margin=args.margin)
        device = get_device(args.device)
        model = model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        loader = DataLoader(TensorDataset(train_triples), batch_size=args.batch_size, shuffle=True)

        best_ap = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, optim, len(dataset.entity_to_id), device)
            metrics = evaluate_auc_like(model, valid_triples, len(dataset.entity_to_id), device) if valid_triples is not None and len(valid_triples) else {}
            history.append({"epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in metrics.items()}})
            ap = metrics.get("average_precision", -1.0) if metrics else -1.0
            if ap is not None and float(ap) > best_ap:
                best_ap = float(ap)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "best_model.pt")
            print(f"epoch={epoch:03d} loss={loss:.4f} valid={metrics}")
        if (out_dir / "best_model.pt").exists():
            model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device)["model_state"])
        else:
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "best_model.pt")

        metrics = evaluate_auc_like(model, test_triples, len(dataset.entity_to_id), device) if test_triples is not None and len(test_triples) else {}
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        save_json(dataset.entity_to_id, out_dir / "entity_to_id.json")
        save_json(dataset.relation_to_id, out_dir / "relation_to_id.json")

        pred_df = pd.DataFrame()
        if files["candidate"]:
            candidates_df = _read_triples_limited(Path(files["candidate"]), args.max_candidate_triples)
            pred_df = score_triples_dataframe(model, dataset, candidates_df, device, args.score_batch_size)
        elif dataset.test_df is not None:
            pred_df = score_triples_dataframe(model, dataset, dataset.test_df, device, args.score_batch_size)
        entity_ids = set()
        if not pred_df.empty and {"head", "tail"}.issubset(pred_df.columns):
            entity_ids = set(pred_df["head"].astype(str)).union(set(pred_df["tail"].astype(str)))
        entities = read_entities_subset(Path(files["entities"]), entity_ids) if files["entities"] else None
        pred_df = attach_refs_from_entities(pred_df, entities)
        pred_df["model"] = f"stage2_{args.model}"
        pred_df["stage"] = "Stage 2 — KG embedding baseline"
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

        summary = {
            "stage": "Stage 2 — KG embedding baseline",
            "model": f"stage2_{args.model}",
            "status": "trained",
            "stage_dir": str(stage_dir),
            "graph_train_file": str(files["graph_train"]),
            "target_train_file": str(files["target_train"]) if files["target_train"] else None,
            "num_entities": len(dataset.entity_to_id),
            "num_relations": len(dataset.relation_to_id),
            "train_triples_used": int(len(train_triples)),
            "prediction_rows_written": int(len(pred_df)),
            "score_note": "KGE scores are rank scores; the exported score column is sigmoid(raw_score), not a calibrated probability.",
            "metrics": metrics,
            **{k: v for k, v in metrics.items()},
        }
        if args.export_neo4j and not pred_df.empty:
            summary["neo4j_export"] = export_predictions_dataframe(pred_df, model_name=f"stage2_{args.model}", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PyTorch KG embedding baselines: DistMult, ComplEx, RotatE.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--model", choices=["distmult", "complex", "rotate"], default="rotate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-graph-train-triples", type=int, default=500000, help="Cap graph-context training triples for Docker safety. Use 0 to read all.")
    parser.add_argument("--max-candidate-triples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
