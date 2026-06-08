from __future__ import annotations

import argparse
import json
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch_geometric.nn import HGTConv
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch Geometric is required for Stage 3 HGT. Install torch-geometric matching your PyTorch build.") from exc

from .common import binary_metrics, ensure_dir, get_device, save_json, set_seed
from .neo4j_export import export_predictions_dataframe
from .stage3_common import encode_pairs, load_candidate_pairs, load_heterodata, load_node_maps, load_pairs, resolve_stage3_dir


class HGTLinkPredictor(nn.Module):
    def __init__(self, data, hidden_dim: int = 128, num_layers: int = 2, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.node_types = list(data.node_types)
        self.input_proj = nn.ModuleDict()
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            num_nodes = data[node_type].num_nodes
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                self.input_proj[node_type] = nn.Linear(int(data[node_type].x.size(-1)), hidden_dim)
            else:
                self.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, data.metadata(), heads=heads) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)), nn.ReLU(), nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def initial_x_dict(self, data) -> dict[str, torch.Tensor]:
        x_dict = {}
        for node_type in self.node_types:
            if node_type in self.input_proj and hasattr(data[node_type], "x") and data[node_type].x is not None:
                x_dict[node_type] = self.input_proj[node_type](data[node_type].x.float())
            else:
                idx = torch.arange(data[node_type].num_nodes, device=next(self.parameters()).device)
                x_dict[node_type] = self.node_embeddings[node_type](idx)
        return x_dict

    def encode(self, data) -> dict[str, torch.Tensor]:
        x_dict = self.initial_x_dict(data)
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict

    def score_pairs(self, z_dict: dict[str, torch.Tensor], compound_idx: torch.Tensor, protein_idx: torch.Tensor) -> torch.Tensor:
        c = z_dict["Compound"][compound_idx]
        p = z_dict["Protein"][protein_idx]
        pair = torch.cat([c, p, torch.abs(c - p), c * p], dim=-1)
        return self.decoder(pair).squeeze(-1)


def run_epoch(model, data, loader, optim, device):
    model.train()
    total = 0.0
    for c, p, y in loader:
        c, p, y = c.to(device), p.to(device), y.to(device)
        optim.zero_grad()
        z = model.encode(data)
        logits = model.score_pairs(z, c, p)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optim.step()
        total += float(loss.item()) * y.numel()
    return total / max(1, len(loader.dataset))


def predict(model, data, c_idx, p_idx, device, batch_size: int = 16384):
    model.eval()
    scores = []
    with torch.no_grad():
        z = model.encode(data)
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
        c_tr, p_tr, y_tr, _ = encode_pairs(train_df, node_maps)
        c_va, p_va, y_va, _ = encode_pairs(valid_df, node_maps)
        c_te, p_te, y_te, test_kept = encode_pairs(test_df, node_maps)
        device = get_device(args.device)
        data = data.to(device)
        model = HGTLinkPredictor(data, args.hidden_dim, args.num_layers, args.heads, args.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(c_tr, p_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        best_ap = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = run_epoch(model, data, loader, optim, device)
            val_score = predict(model, data, c_va, p_va, device, args.score_batch_size)
            val_metrics = binary_metrics(y_va.numpy(), val_score, threshold=args.threshold)
            history.append({"epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in val_metrics.items()}})
            ap = val_metrics.get("average_precision") or -1.0
            if float(ap) > best_ap:
                best_ap = float(ap)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "best_model.pt")
            print(f"epoch={epoch:03d} loss={loss:.4f} valid={val_metrics}")
        ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        cand_df = load_candidate_pairs(stage_dir)
        if cand_df is not None and args.score_candidates:
            c_score, p_score, _, score_rows = encode_pairs(cand_df.head(args.max_candidate_pairs) if args.max_candidate_pairs > 0 else cand_df, node_maps)
            score = predict(model, data, c_score, p_score, device, args.score_batch_size)
        else:
            score_rows = test_kept.copy()
            score = predict(model, data, c_te, p_te, device, args.score_batch_size)
        test_score = predict(model, data, c_te, p_te, device, args.score_batch_size)
        metrics = binary_metrics(y_te.numpy(), test_score, threshold=args.threshold)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        preds = score_rows.copy().reset_index(drop=True)
        preds["score"] = score
        preds["predicted_label"] = (score >= args.threshold).astype(int)
        preds["model"] = "stage3_hgt"
        preds["stage"] = "Stage 3 — HGT"
        preds.to_csv(out_dir / "predictions.csv", index=False)
        summary = {"stage": "Stage 3 — HGT", "model": "stage3_hgt", "metrics": metrics, **{k: v for k, v in metrics.items()}, "prediction_rows_written": int(len(preds)), "predictions_file": str(out_dir / "predictions.csv")}
        if args.export_neo4j:
            summary["neo4j_export"] = export_predictions_dataframe(preds, model_name="stage3_hgt", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HGT + MLP decoder for compound-CYP450 link prediction.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score-batch-size", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
