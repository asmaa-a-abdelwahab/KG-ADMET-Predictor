from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch_geometric.nn import RGCNConv
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch Geometric is required for Stage 3 R-GCN. Install torch-geometric matching your PyTorch build.") from exc

from .common import binary_metrics, ensure_dir, get_device, save_json, set_seed
from .neo4j_export import export_predictions_dataframe
from .stage3_common import encode_pairs, load_candidate_pairs, load_heterodata, load_node_maps, load_pairs, resolve_stage3_dir


def make_homogeneous_rgcn_graph(data, add_reverse_edges: bool = True) -> dict[str, Any]:
    node_offsets: dict[str, int] = {}
    node_counts: dict[str, int] = {}
    cursor = 0
    for node_type in data.node_types:
        n = int(data[node_type].num_nodes)
        node_offsets[node_type] = cursor
        node_counts[node_type] = n
        cursor += n
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    relation_to_id: dict[str, int] = {}

    def add_edges(src_type: str, dst_type: str, edge_index: torch.Tensor, rel_name: str) -> None:
        if edge_index.numel() == 0:
            return
        rid = relation_to_id.setdefault(rel_name, len(relation_to_id))
        src = edge_index[0].long() + node_offsets[src_type]
        dst = edge_index[1].long() + node_offsets[dst_type]
        edge_indices.append(torch.stack([src, dst], dim=0))
        edge_types.append(torch.full((src.numel(),), rid, dtype=torch.long))

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index.cpu()
        rel_name = f"{src_type}__{rel}__{dst_type}"
        add_edges(src_type, dst_type, edge_index, rel_name)
        if add_reverse_edges:
            add_edges(dst_type, src_type, edge_index.flip(0), f"{dst_type}__rev_{rel}__{src_type}")
    if not edge_indices:
        raise ValueError("The HeteroData object contains no edges.")
    return {
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_type": torch.cat(edge_types, dim=0),
        "num_nodes": cursor,
        "num_relations": len(relation_to_id),
        "relation_to_id": relation_to_id,
        "node_offsets": node_offsets,
        "node_counts": node_counts,
    }


class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128, num_layers: int = 2, num_bases: int | None = 30, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        bases = min(num_bases or num_relations, num_relations)
        self.convs = nn.ModuleList([RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=bases) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)), nn.ReLU(), nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def score_pairs(self, z: torch.Tensor, compound_global_idx: torch.Tensor, protein_global_idx: torch.Tensor) -> torch.Tensor:
        c = z[compound_global_idx]
        p = z[protein_global_idx]
        pair = torch.cat([c, p, torch.abs(c - p), c * p], dim=-1)
        return self.decoder(pair).squeeze(-1)


def to_global_pair_indices(compound_local: torch.LongTensor, protein_local: torch.LongTensor, node_offsets: dict[str, int]):
    return compound_local + int(node_offsets["Compound"]), protein_local + int(node_offsets["Protein"])


def run_epoch(model, graph, loader, optim, device):
    model.train()
    total = 0.0
    edge_index = graph["edge_index"].to(device)
    edge_type = graph["edge_type"].to(device)
    for c, p, y in loader:
        c, p, y = c.to(device), p.to(device), y.to(device)
        optim.zero_grad()
        z = model.encode(edge_index, edge_type)
        logits = model.score_pairs(z, c, p)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optim.step()
        total += float(loss.item()) * y.numel()
    return total / max(1, len(loader.dataset))


def predict(model, graph, c_idx, p_idx, device, batch_size: int = 16384):
    model.eval()
    edge_index = graph["edge_index"].to(device)
    edge_type = graph["edge_type"].to(device)
    scores = []
    with torch.no_grad():
        z = model.encode(edge_index, edge_type)
        for start in range(0, len(c_idx), batch_size):
            cb = c_idx[start:start + batch_size].to(device)
            pb = p_idx[start:start + batch_size].to(device)
            scores.append(torch.sigmoid(model.score_pairs(z, cb, pb)).cpu())
    return torch.cat(scores).numpy()


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    resolved = resolve_stage3_dir(args.modeling_dir)
    out_dir = ensure_dir(args.output_dir)
    try:
        stage_dir = resolved.root
        data = load_heterodata(stage_dir)
        node_maps = load_node_maps(stage_dir)
        train_df, valid_df, test_df = load_pairs(stage_dir, args.seed)
        c_tr_local, p_tr_local, y_tr, _ = encode_pairs(train_df, node_maps)
        c_va_local, p_va_local, y_va, _ = encode_pairs(valid_df, node_maps)
        c_te_local, p_te_local, y_te, test_kept = encode_pairs(test_df, node_maps)
        graph = make_homogeneous_rgcn_graph(data, add_reverse_edges=not args.no_reverse_edges)
        c_tr, p_tr = to_global_pair_indices(c_tr_local, p_tr_local, graph["node_offsets"])
        c_va, p_va = to_global_pair_indices(c_va_local, p_va_local, graph["node_offsets"])
        c_te, p_te = to_global_pair_indices(c_te_local, p_te_local, graph["node_offsets"])

        device = get_device(args.device)
        model = RGCNLinkPredictor(graph["num_nodes"], graph["num_relations"], args.hidden_dim, args.num_layers, args.num_bases, args.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(c_tr, p_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        save_json({"node_offsets": graph["node_offsets"], "node_counts": graph["node_counts"], "relation_to_id": graph["relation_to_id"], "args": vars(args)}, out_dir / "rgcn_graph_metadata.json")
        best_ap = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = run_epoch(model, graph, loader, optim, device)
            val_score = predict(model, graph, c_va, p_va, device, args.score_batch_size)
            val_metrics = binary_metrics(y_va.numpy(), val_score, threshold=args.threshold)
            history.append({"epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in val_metrics.items()}})
            ap = val_metrics.get("average_precision") or -1.0
            if float(ap) > best_ap:
                best_ap = float(ap)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "best_model.pt")
            print(f"epoch={epoch:03d} loss={loss:.4f} valid={val_metrics}")
        ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # Score candidates when available, otherwise score held-out test pairs.
        cand_df = load_candidate_pairs(stage_dir)
        if cand_df is not None and args.score_candidates:
            c_can_local, p_can_local, _, score_rows = encode_pairs(cand_df.head(args.max_candidate_pairs) if args.max_candidate_pairs > 0 else cand_df, node_maps)
            c_score, p_score = to_global_pair_indices(c_can_local, p_can_local, graph["node_offsets"])
            score = predict(model, graph, c_score, p_score, device, args.score_batch_size)
        else:
            score_rows = test_kept.copy()
            score = predict(model, graph, c_te, p_te, device, args.score_batch_size)
        test_score = predict(model, graph, c_te, p_te, device, args.score_batch_size)
        metrics = binary_metrics(y_te.numpy(), test_score, threshold=args.threshold)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        preds = score_rows.copy().reset_index(drop=True)
        preds["score"] = score
        preds["predicted_label"] = (score >= args.threshold).astype(int)
        preds["model"] = "stage3_rgcn"
        preds["stage"] = "Stage 3 — R-GCN"
        preds.to_csv(out_dir / "predictions.csv", index=False)
        summary = {"stage": "Stage 3 — R-GCN", "model": "stage3_rgcn", "metrics": metrics, **{k: v for k, v in metrics.items()}, "prediction_rows_written": int(len(preds)), "predictions_file": str(out_dir / "predictions.csv")}
        if args.export_neo4j:
            summary["neo4j_export"] = export_predictions_dataframe(preds, model_name="stage3_rgcn", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train R-GCN + MLP decoder for compound-CYP450 link prediction.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score-batch-size", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-reverse-edges", action="store_true")
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
