from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    STAGE2,
    attach_refs_from_entities,
    binary_metrics,
    ensure_dir,
    get_device,
    read_table,
    read_triples,
    resolve_stage_dir,
    save_json,
    set_seed,
    sigmoid_np,
)
from .neo4j_export import export_predictions_dataframe


# -----------------------------------------------------------------------------
# Dataset and model definitions
# -----------------------------------------------------------------------------

class KGDataset:
    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None, test_df: pd.DataFrame | None = None):
        self.train_df = train_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True)
        self.valid_df = valid_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True) if valid_df is not None else None
        self.test_df = test_df[["head", "relation", "tail"]].astype(str).reset_index(drop=True) if test_df is not None else None

        # Include validation/test entities to avoid dropping held-out CYP450 target triples.
        all_df = pd.concat([x for x in [self.train_df, self.valid_df, self.test_df] if x is not None], ignore_index=True)
        entities = pd.unique(pd.concat([all_df["head"], all_df["tail"]], ignore_index=True).astype(str))
        relations = pd.unique(all_df["relation"].astype(str))
        # Preserve first-seen order. This is faster than sorting millions of entity IDs and is reproducible
        # because the input files are deterministic.
        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.relation_to_id = {r: i for i, r in enumerate(relations)}

    def encode(self, df: pd.DataFrame) -> torch.LongTensor:
        enc = df[["head", "relation", "tail"]].astype(str).copy()
        h = enc["head"].map(self.entity_to_id)
        r = enc["relation"].map(self.relation_to_id)
        t = enc["tail"].map(self.entity_to_id)
        mask = h.notna() & r.notna() & t.notna()
        arr = np.column_stack([
            h[mask].to_numpy(dtype=np.int64),
            r[mask].to_numpy(dtype=np.int64),
            t[mask].to_numpy(dtype=np.int64),
        ])
        return torch.from_numpy(arr.astype(np.int64, copy=False))


class DistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, sparse: bool = True):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation = nn.Embedding(num_relations, dim, sparse=sparse)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, h, r, t):
        return (self.entity(h) * self.relation(r) * self.entity(t)).sum(dim=-1)


class ComplEx(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, sparse: bool = True):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, dim, sparse=sparse)
        self.entity_im = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_re = nn.Embedding(num_relations, dim, sparse=sparse)
        self.relation_im = nn.Embedding(num_relations, dim, sparse=sparse)
        for emb in [self.entity_re, self.entity_im, self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h, r, t):
        hr, hi = self.entity_re(h), self.entity_im(h)
        rr, ri = self.relation_re(r), self.relation_im(r)
        tr, ti = self.entity_re(t), self.entity_im(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)


class RotatE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, margin: float = 6.0, sparse: bool = True):
        super().__init__()
        self.entity_re = nn.Embedding(num_entities, dim, sparse=sparse)
        self.entity_im = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_phase = nn.Embedding(num_relations, dim, sparse=sparse)
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


# -----------------------------------------------------------------------------
# Fast training/evaluation utilities
# -----------------------------------------------------------------------------

def corrupt_triples(pos: torch.LongTensor, num_entities: int) -> torch.LongTensor:
    neg = pos.clone()
    mask = torch.rand(len(pos), device=pos.device) < 0.5
    random_entities = torch.randint(0, num_entities, (len(pos),), device=pos.device)
    neg[mask, 0] = random_entities[mask]
    neg[~mask, 2] = random_entities[~mask]
    return neg


def _loss_from_scores(pos_score: torch.Tensor, neg_score: torch.Tensor, loss_name: str, margin: float) -> torch.Tensor:
    loss_name = loss_name.lower()
    if loss_name == "margin":
        return F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=margin)
    if loss_name == "bce":
        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        return F.binary_cross_entropy_with_logits(scores, labels)
    # Softplus negative-sampling objective: stable and usually faster/better than margin ranking.
    return F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()


def _make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    opt = args.optimizer.lower()
    if opt == "auto":
        opt = "sparse_adam" if args.sparse_embeddings else "adamw"
    if opt == "sparse_adam":
        if not args.sparse_embeddings:
            raise ValueError("--optimizer sparse_adam requires --sparse-embeddings")
        # SparseAdam only updates embeddings touched in a mini-batch and avoids dense optimizer states
        # for millions of entity vectors.
        return torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_epoch(model, loader, optim, num_entities: int, device, args: argparse.Namespace) -> float:
    model.train()
    total = 0.0
    count = 0
    use_amp = bool(args.amp and str(device).startswith("cuda") and not args.sparse_embeddings)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for (pos_cpu,) in loader:
        pos = pos_cpu.to(device, non_blocking=True)
        if args.negatives_per_positive > 1:
            neg_base = pos.repeat_interleave(args.negatives_per_positive, dim=0)
        else:
            neg_base = pos
        neg = corrupt_triples(neg_base, num_entities)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pos_score = model.score(pos[:, 0], pos[:, 1], pos[:, 2])
            if args.negatives_per_positive > 1:
                pos_score_for_loss = pos_score.repeat_interleave(args.negatives_per_positive, dim=0)
            else:
                pos_score_for_loss = pos_score
            neg_score = model.score(neg[:, 0], neg[:, 1], neg[:, 2])
            loss = _loss_from_scores(pos_score_for_loss, neg_score, args.loss, args.ranking_margin)
        scaler.scale(loss).backward()
        if args.max_grad_norm and args.max_grad_norm > 0 and not args.sparse_embeddings:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optim)
        scaler.update()
        bsz = int(len(pos))
        total += float(loss.item()) * bsz
        count += bsz
    return total / max(1, count)


def _score_encoded(model, encoded: torch.LongTensor, device, batch_size: int = 65536) -> np.ndarray:
    if len(encoded) == 0:
        return np.array([], dtype=np.float32)
    scores: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size].to(device, non_blocking=True)
            s = model.score(batch[:, 0], batch[:, 1], batch[:, 2]).detach().cpu()
            scores.append(s)
    return torch.cat(scores).numpy()


def evaluate_auc_like(
    model,
    triples: torch.LongTensor,
    num_entities: int,
    device,
    negatives_per_positive: int = 1,
    max_eval_triples: int = 0,
    batch_size: int = 65536,
) -> dict[str, float | int | None]:
    if triples is None or len(triples) == 0:
        return {"roc_auc": None, "average_precision": None}
    if max_eval_triples and max_eval_triples > 0 and len(triples) > max_eval_triples:
        idx = torch.randperm(len(triples))[:max_eval_triples]
        triples = triples[idx]
    scores, labels = [], []
    pos_score = _score_encoded(model, triples, device, batch_size)
    scores.append(pos_score)
    labels.append(np.ones_like(pos_score))
    # Generate negatives on CPU then score in batches. This avoids putting the full validation tensor
    # and all random negatives on GPU at once.
    for _ in range(negatives_per_positive):
        neg = corrupt_triples(triples.clone(), num_entities)
        neg_score = _score_encoded(model, neg, device, batch_size)
        scores.append(neg_score)
        labels.append(np.zeros_like(neg_score))
    return binary_metrics(np.concatenate(labels), np.concatenate(scores))


# -----------------------------------------------------------------------------
# Input and output helpers
# -----------------------------------------------------------------------------

def _read_triples_limited(path: Path, nrows: int = 0) -> pd.DataFrame:
    if nrows and nrows > 0:
        head = pd.read_csv(path, sep="\t", nrows=0)
        if {"head", "relation", "tail"}.issubset(head.columns):
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)[["head", "relation", "tail"]].astype(str)
        return pd.read_csv(path, sep="\t", header=None, nrows=nrows, names=["head", "relation", "tail"], usecols=[0, 1, 2], low_memory=False).astype(str)
    return read_triples(path)


def _read_triples_header_aware(path: Path, chunksize: int | None = None, nrows: int | None = None):
    head = pd.read_csv(path, sep="\t", nrows=0)
    if {"head", "relation", "tail"}.issubset(head.columns):
        return pd.read_csv(path, sep="\t", usecols=["head", "relation", "tail"], chunksize=chunksize, nrows=nrows, low_memory=False)
    return pd.read_csv(path, sep="\t", header=None, names=["head", "relation", "tail"], usecols=[0, 1, 2], chunksize=chunksize, nrows=nrows, low_memory=False)


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


def _encode_triples_frame(dataset: KGDataset, triples_df: pd.DataFrame) -> tuple[pd.DataFrame, torch.LongTensor]:
    triples = triples_df[["head", "relation", "tail"]].astype(str).copy()
    h = triples["head"].map(dataset.entity_to_id)
    r = triples["relation"].map(dataset.relation_to_id)
    t = triples["tail"].map(dataset.entity_to_id)
    mask = h.notna() & r.notna() & t.notna()
    if not mask.any():
        return triples.iloc[0:0].copy(), torch.empty((0, 3), dtype=torch.long)
    out_rows = triples.loc[mask].reset_index(drop=True)
    encoded = np.column_stack([
        h[mask].to_numpy(dtype=np.int64),
        r[mask].to_numpy(dtype=np.int64),
        t[mask].to_numpy(dtype=np.int64),
    ]).astype(np.int64, copy=False)
    return out_rows, torch.from_numpy(encoded)


def score_triples_dataframe(model, dataset: KGDataset, triples_df: pd.DataFrame, device, batch_size: int = 65536, sort_predictions: bool = True, top_k: int = 0) -> pd.DataFrame:
    rows, encoded = _encode_triples_frame(dataset, triples_df)
    if len(encoded) == 0:
        return pd.DataFrame(columns=["head", "relation", "tail", "raw_score", "score"])
    raw = _score_encoded(model, encoded, device, batch_size)
    out = rows.copy()
    out["raw_score"] = raw
    out["score"] = sigmoid_np(raw)
    if sort_predictions:
        out = out.sort_values("score", ascending=False)
    if top_k and top_k > 0:
        out = out.head(top_k)
    return out.reset_index(drop=True)


def score_triples_file_fast(
    model,
    dataset: KGDataset,
    path: Path,
    device,
    batch_size: int = 65536,
    max_rows: int = 0,
    chunk_size: int = 500000,
    sort_predictions: bool = True,
    top_k: int = 0,
) -> pd.DataFrame:
    chunks = []
    remaining = max_rows if max_rows and max_rows > 0 else None
    reader = _read_triples_header_aware(path, chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        scored = score_triples_dataframe(model, dataset, chunk, device, batch_size=batch_size, sort_predictions=False)
        if len(scored):
            chunks.append(scored)
        print(f"scoring_chunk={i} input_rows={len(chunk)} scored_rows={len(scored)}")
    if not chunks:
        return pd.DataFrame(columns=["head", "relation", "tail", "raw_score", "score"])
    out = pd.concat(chunks, ignore_index=True)
    if sort_predictions:
        out = out.sort_values("score", ascending=False)
    if top_k and top_k > 0:
        out = out.head(top_k)
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def _prepare_train_df(graph_train: pd.DataFrame, target_train: pd.DataFrame | None, target_repeat: int) -> pd.DataFrame:
    graph_train = graph_train[["head", "relation", "tail"]].astype(str)
    if target_train is None or len(target_train) == 0:
        return graph_train.drop_duplicates().reset_index(drop=True)
    target_train = target_train[["head", "relation", "tail"]].astype(str).drop_duplicates()
    base = pd.concat([graph_train, target_train], ignore_index=True).drop_duplicates()
    if target_repeat and target_repeat > 1:
        # Repeat CYP450 target-relation training triples after de-duplication so the supervised relation
        # gets enough updates without having to train on the complete KG for many epochs.
        repeated = [target_train] * (target_repeat - 1)
        base = pd.concat([base] + repeated, ignore_index=True)
    return base.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    resolved = resolve_stage_dir(args.modeling_dir, STAGE2)
    out_dir = ensure_dir(args.output_dir)
    t0 = time.time()
    try:
        stage_dir = resolved.root
        files = find_stage2_files(stage_dir)
        if not files["graph_train"] or not Path(files["graph_train"]).exists():
            raise FileNotFoundError(f"No Stage 2 train triples found under {stage_dir}")

        print("Reading Stage 2 train files...")
        graph_train = _read_triples_limited(Path(files["graph_train"]), args.max_graph_train_triples)
        target_train = read_triples(Path(files["target_train"])) if files["target_train"] else None
        train_df = _prepare_train_df(graph_train, target_train, args.target_train_repeat)
        valid_df = read_triples(Path(files["valid"])) if files["valid"] else None
        test_df = read_triples(Path(files["test"])) if files["test"] else None

        print(f"train_rows={len(train_df)} graph_rows={len(graph_train)} target_rows={0 if target_train is None else len(target_train)}")
        dataset = KGDataset(train_df, valid_df, test_df)
        train_triples = dataset.encode(dataset.train_df)
        valid_triples = dataset.encode(dataset.valid_df) if dataset.valid_df is not None else None
        test_triples = dataset.encode(dataset.test_df) if dataset.test_df is not None else None
        if len(train_triples) == 0:
            raise ValueError("No train triples could be encoded.")

        if args.model == "distmult":
            model = DistMult(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim, sparse=args.sparse_embeddings)
        elif args.model == "complex":
            model = ComplEx(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim, sparse=args.sparse_embeddings)
        else:
            model = RotatE(len(dataset.entity_to_id), len(dataset.relation_to_id), args.dim, margin=args.margin, sparse=args.sparse_embeddings)

        device = get_device(args.device)
        model = model.to(device)
        optim = _make_optimizer(model, args)
        pin_memory = bool(args.pin_memory and str(device).startswith("cuda"))
        loader = DataLoader(
            TensorDataset(train_triples),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(args.num_workers > 0),
            drop_last=False,
        )

        print("============================================================")
        print("Stage 2 efficient KGE configuration")
        print("============================================================")
        print(f"model={args.model} entities={len(dataset.entity_to_id):,} relations={len(dataset.relation_to_id):,}")
        print(f"train_triples={len(train_triples):,} dim={args.dim} device={device}")
        print(f"sparse_embeddings={args.sparse_embeddings} optimizer={args.optimizer} loss={args.loss}")
        print(f"batch_size={args.batch_size} negatives_per_positive={args.negatives_per_positive}")
        print(f"eval_every={args.eval_every} patience={args.patience} score_candidates={args.score_candidates}")
        print("============================================================")

        best_value = -float("inf")
        best_epoch = 0
        stale_epochs = 0
        history = []
        checkpoint_metric = args.checkpoint_metric

        if args.score_only:
            if not args.load_model:
                raise ValueError("--score-only requires --load-model pointing to a Stage 2 best_model.pt checkpoint")
            print(f"Score-only mode: loading checkpoint from {args.load_model}")
            ckpt = torch.load(args.load_model, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            # Keep a local copy of the checkpoint path in metrics for traceability but do not retrain.
            best_epoch = int(ckpt.get("args", {}).get("best_epoch", 0) or 0)
        else:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.time()
                loss = train_epoch(model, loader, optim, len(dataset.entity_to_id), device, args)
                metrics = {}
                should_eval = bool(args.eval_every > 0 and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs))
                if should_eval and valid_triples is not None and len(valid_triples):
                    metrics = evaluate_auc_like(
                        model,
                        valid_triples,
                        len(dataset.entity_to_id),
                        device,
                        negatives_per_positive=args.eval_negatives_per_positive,
                        max_eval_triples=args.max_eval_triples,
                        batch_size=args.score_batch_size,
                    )
                metric_value = metrics.get(checkpoint_metric)
                if metric_value is None and metrics:
                    metric_value = metrics.get("average_precision") or metrics.get("roc_auc")
                if metric_value is not None:
                    metric_value = float(metric_value)
                    if metric_value > best_value + args.min_delta:
                        best_value = metric_value
                        best_epoch = epoch
                        stale_epochs = 0
                        torch.save({"model_state": model.state_dict(), "args": vars(args), "best_epoch": best_epoch}, out_dir / "best_model.pt")
                    else:
                        stale_epochs += 1
                elif epoch == args.epochs:
                    torch.save({"model_state": model.state_dict(), "args": vars(args), "best_epoch": best_epoch}, out_dir / "best_model.pt")

                elapsed = time.time() - epoch_start
                row = {"epoch": epoch, "loss": loss, "seconds": elapsed, **{f"valid_{k}": v for k, v in metrics.items()}}
                history.append(row)
                print(f"epoch={epoch:03d} loss={loss:.4f} seconds={elapsed:.1f} valid={metrics} best_{checkpoint_metric}={best_value:.6f} best_epoch={best_epoch}")
                if args.patience and args.patience > 0 and best_epoch > 0 and stale_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}; no {checkpoint_metric} improvement for {stale_epochs} evaluated epochs.")
                    break

            if (out_dir / "best_model.pt").exists():
                model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device)["model_state"])
            else:
                torch.save({"model_state": model.state_dict(), "args": vars(args), "best_epoch": best_epoch}, out_dir / "best_model.pt")

        metrics = evaluate_auc_like(
            model,
            test_triples,
            len(dataset.entity_to_id),
            device,
            negatives_per_positive=args.eval_negatives_per_positive,
            max_eval_triples=args.max_eval_triples,
            batch_size=args.score_batch_size,
        ) if test_triples is not None and len(test_triples) else {}
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        if args.save_mappings:
            save_json(dataset.entity_to_id, out_dir / "entity_to_id.json")
            save_json(dataset.relation_to_id, out_dir / "relation_to_id.json")
        else:
            save_json({
                "num_entities": len(dataset.entity_to_id),
                "num_relations": len(dataset.relation_to_id),
                "note": "Full entity_to_id/relation_to_id JSON export skipped. Use --save-mappings if needed.",
            }, out_dir / "mapping_metadata.json")

        pred_df = pd.DataFrame()
        if args.score_candidates and files["candidate"]:
            pred_df = score_triples_file_fast(
                model,
                dataset,
                Path(files["candidate"]),
                device,
                batch_size=args.score_batch_size,
                max_rows=args.max_candidate_triples,
                chunk_size=args.scoring_chunk_size,
                sort_predictions=args.sort_predictions,
                top_k=args.prediction_top_k,
            )
        elif args.score_candidates and dataset.test_df is not None:
            pred_df = score_triples_dataframe(model, dataset, dataset.test_df, device, args.score_batch_size, sort_predictions=args.sort_predictions, top_k=args.prediction_top_k)

        if args.attach_entity_refs and not pred_df.empty and files["entities"]:
            entity_ids = set(pred_df["head"].astype(str)).union(set(pred_df["tail"].astype(str)))
            entities = read_entities_subset(Path(files["entities"]), entity_ids)
            pred_df = attach_refs_from_entities(pred_df, entities)

        if pred_df.empty:
            pred_df = pd.DataFrame(columns=["head", "relation", "tail", "raw_score", "score"])
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
            "best_epoch": int(best_epoch),
            "best_validation_metric": checkpoint_metric,
            "best_validation_value": None if best_value == -float("inf") else float(best_value),
            "prediction_rows_written": int(len(pred_df)),
            "predictions_file": str(out_dir / "predictions.csv"),
            "model_file": str(out_dir / "best_model.pt"),
            "training_history_file": str(out_dir / "training_history.csv"),
            "score_note": "KGE scores are rank scores; the exported score column is sigmoid(raw_score), not a calibrated probability.",
            "runtime_seconds": float(time.time() - t0),
            "config": {
                "dim": args.dim,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "sparse_embeddings": args.sparse_embeddings,
                "max_graph_train_triples": args.max_graph_train_triples,
                "target_train_repeat": args.target_train_repeat,
                "score_candidates": args.score_candidates,
                "max_candidate_triples": args.max_candidate_triples,
            },
            "metrics": metrics,
            **{k: v for k, v in metrics.items()},
        }
        if args.export_neo4j and not pred_df.empty:
            summary["neo4j_export"] = export_predictions_dataframe(pred_df, model_name=f"stage2_{args.model}", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    dest = name.replace("--", "").replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument("--no-" + name[2:], dest=dest, action="store_false", help="Disable: " + help)
    parser.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train efficient PyTorch KG embedding baselines: DistMult, ComplEx, RotatE.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--model", choices=["distmult", "complex", "rotate"], default="rotate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--score-batch-size", type=int, default=262144)
    parser.add_argument("--dim", "--embedding-dim", dest="dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=6.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--loss", choices=["softplus", "bce", "margin"], default="softplus")
    parser.add_argument("--optimizer", choices=["auto", "sparse_adam", "adam", "adamw"], default="auto")
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--eval-negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-graph-train-triples", type=int, default=1000000, help="Cap graph-context training triples. Use 0 to read all.")
    parser.add_argument("--target-train-repeat", type=int, default=5, help="Repeat target CYP450 train triples to focus updates on the prediction relation.")
    parser.add_argument("--max-candidate-triples", "--max-score-triples", dest="max_candidate_triples", type=int, default=100000)
    parser.add_argument("--scoring-chunk-size", type=int, default=500000)
    parser.add_argument("--prediction-top-k", type=int, default=0, help="Keep only top K scored predictions. 0 writes all scored rows.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint-metric", choices=["average_precision", "roc_auc", "f1", "accuracy", "balanced_accuracy", "mcc"], default="average_precision")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    _add_bool_arg(parser, "--sparse-embeddings", True, "Use sparse embedding gradients and SparseAdam for large KG training.")
    _add_bool_arg(parser, "--pin-memory", True, "Enable DataLoader pinned memory when using CUDA.")
    _add_bool_arg(parser, "--amp", False, "Use CUDA AMP. Disabled by default because sparse embedding optimizers do not benefit reliably.")
    _add_bool_arg(parser, "--score-candidates", False, "Score candidate target triples after training. This can be expensive for millions of candidates.")
    _add_bool_arg(parser, "--score-only", False, "Load a trained checkpoint and score/evaluate without training.")
    parser.add_argument("--load-model", default=None, help="Path to best_model.pt for --score-only mode.")
    _add_bool_arg(parser, "--sort-predictions", True, "Sort prediction rows by score before writing.")
    _add_bool_arg(parser, "--attach-entity-refs", False, "Attach entity metadata/reference columns to scored predictions.")
    _add_bool_arg(parser, "--save-mappings", False, "Save full entity_to_id and relation_to_id JSON files. Large and slow for millions of entities.")
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
