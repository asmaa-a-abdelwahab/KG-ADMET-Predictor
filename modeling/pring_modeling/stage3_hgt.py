from __future__ import annotations

import argparse
import json
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.nn import HGTConv
except Exception:  # pragma: no cover
    LinkNeighborLoader = None  # type: ignore
    HGTConv = None  # type: ignore

from .common import ensure_dir, get_device, save_json, set_seed
from .model_diagnostics import (
    balanced_diagnostic_metrics,
    operating_point_metrics,
    per_group_binary_metrics,
    write_dataframe_if_not_empty,
)
from .neo4j_export import export_predictions_dataframe
from .stage3_common import encode_pairs, load_candidate_pairs, load_heterodata, load_node_maps, load_pairs, resolve_stage3_dir
from .stage3_training_utils import (
    amp_autocast,
    extended_binary_metrics,
    label_stats,
    make_grad_scaler,
    maybe_balance_training_tensors,
    metric_for_checkpoint,
    optimize_threshold,
    resolve_class_weights,
    weighted_or_focal_loss,
)

TARGET_EDGE_TYPE = ("Compound", "stage3_supervision", "Protein")

# HGTConv uses pyg-lib grouped_matmul, which can fail with mixed precision
# on some CUDA/PyG builds (Float weights vs Half inputs). Keep HGT in FP32.
HGT_ALLOW_AMP = False


def _parse_num_neighbors(value: str | int, num_layers: int) -> list[int]:
    if isinstance(value, int):
        return [value] * max(1, num_layers)
    vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        vals = [10]
    if len(vals) < num_layers:
        vals.extend([vals[-1]] * (num_layers - len(vals)))
    return vals[: max(1, num_layers)]


def _safe_key(name: str) -> str:
    return str(name).replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def _node_counts(data) -> dict[str, int]:
    return {nt: int(data[nt].num_nodes) for nt in data.node_types}


def _feature_dims(data) -> dict[str, int]:
    dims: dict[str, int] = {}
    for nt in data.node_types:
        x = getattr(data[nt], "x", None)
        dims[nt] = int(x.size(-1)) if x is not None else 0
    return dims


def _ensure_supervision_edge_type(data):
    if TARGET_EDGE_TYPE not in data.edge_types:
        data[TARGET_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data


class SampledHGTLinkPredictor(nn.Module):
    """Mini-batch HGT for large PRING heterogeneous link prediction.

    Uses LinkNeighborLoader sampled HeteroData batches instead of moving the full
    graph to GPU. For featureless node types, the default uses a single trainable
    type vector, which is much cheaper than global per-node embeddings on 8 GB GPUs.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        node_counts: dict[str, int],
        feature_dims: dict[str, int],
        hidden_dim: int = 32,
        num_layers: int = 1,
        heads: int = 1,
        dropout: float = 0.2,
        featureless_mode: str = "type",
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.dropout = dropout
        self.featureless_mode = featureless_mode
        self.input_proj = nn.ModuleDict()
        self.type_embeddings = nn.ParameterDict()
        self.global_embeddings = nn.ModuleDict()

        for nt in self.node_types:
            key = _safe_key(nt)
            dim = int(feature_dims.get(nt, 0))
            if dim > 0:
                self.input_proj[key] = nn.Linear(dim, hidden_dim)
            if featureless_mode == "global":
                self.global_embeddings[key] = nn.Embedding(int(node_counts[nt]), hidden_dim)
                nn.init.xavier_uniform_(self.global_embeddings[key].weight)
            else:
                self.type_embeddings[key] = nn.Parameter(torch.empty(hidden_dim))
                nn.init.normal_(self.type_embeddings[key], mean=0.0, std=0.02)

        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
            for _ in range(max(1, num_layers))
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def _initial_node_features(self, node_type: str, store) -> torch.Tensor:
        key = _safe_key(node_type)
        x = getattr(store, "x", None)
        n = int(store.num_nodes)
        if x is not None and key in self.input_proj:
            return self.input_proj[key](x.float())
        if self.featureless_mode == "global" and key in self.global_embeddings:
            if hasattr(store, "n_id"):
                ids = store.n_id.long().to(next(self.parameters()).device)
            else:
                ids = torch.arange(n, device=next(self.parameters()).device)
            return self.global_embeddings[key](ids)
        emb = self.type_embeddings[key].view(1, -1)
        return emb.expand(n, -1)

    def initial_x_dict(self, batch) -> dict[str, torch.Tensor]:
        return {nt: self._initial_node_features(nt, batch[nt]) for nt in batch.node_types}

    def encode_batch(self, batch) -> dict[str, torch.Tensor]:
        x_dict = self.initial_x_dict(batch)
        edge_index_dict = {et: ei for et, ei in batch.edge_index_dict.items() if ei is not None}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict

    def score_supervision_edges(self, z_dict: dict[str, torch.Tensor], batch) -> torch.Tensor:
        edge_label_index = batch[TARGET_EDGE_TYPE].edge_label_index.long().to(z_dict["Compound"].device)
        c = z_dict["Compound"][edge_label_index[0]]
        p = z_dict["Protein"][edge_label_index[1]]
        pair = torch.cat([c, p, torch.abs(c - p), c * p], dim=-1)
        return self.decoder(pair).squeeze(-1)


def _make_loader(data, c_idx, p_idx, y, args, shuffle: bool):
    if LinkNeighborLoader is None:
        raise RuntimeError("PyTorch Geometric LinkNeighborLoader is required. Install torch-geometric with pyg-lib or torch-sparse support.")
    edge_label_index = torch.stack([c_idx.long(), p_idx.long()], dim=0)
    return LinkNeighborLoader(
        data,
        num_neighbors=_parse_num_neighbors(args.num_neighbors, args.num_layers),
        edge_label_index=(TARGET_EDGE_TYPE, edge_label_index),
        edge_label=y.float(),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )


def _run_epoch(model, loader, optim, device, args, class_weights: dict[int, float]) -> float:
    model.train()
    total = 0.0
    n_total = 0
    scaler = make_grad_scaler(device, args.amp)
    for batch in loader:
        batch = batch.to(device)
        y = batch[TARGET_EDGE_TYPE].edge_label.float().to(device)
        optim.zero_grad(set_to_none=True)
        # Keep HGT in FP32. Mixed precision can make HGTConv/pyg-lib
        # grouped_matmul receive Half inputs while its grouped weights remain Float.
        # That produces: RuntimeError: expected scalar type Float but found Half.
        with amp_autocast(device, False):
            z_dict = model.encode_batch(batch)
            logits = model.score_supervision_edges(z_dict, batch)
            loss = weighted_or_focal_loss(logits, y, args, class_weights)
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(optim)
        scaler.update()
        total += float(loss.item()) * y.numel()
        n_total += int(y.numel())
    return total / max(1, n_total)


def _predict(model, loader, device, use_amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = batch[TARGET_EDGE_TYPE].edge_label.float().cpu()
            # Keep HGT validation/scoring in FP32 for the same grouped_matmul reason.
            with amp_autocast(device, False):
                z_dict = model.encode_batch(batch)
                logits = model.score_supervision_edges(z_dict, batch)
                score = torch.sigmoid(logits).detach().cpu()
            scores.append(score)
            labels.append(y)
    return torch.cat(scores), torch.cat(labels)


def _score_rows(model, data, rows: pd.DataFrame, node_maps, args, device) -> tuple[pd.DataFrame, torch.Tensor]:
    c, p, y, kept = encode_pairs(rows, node_maps)
    loader = _make_loader(data, c, p, y, args, shuffle=False)
    scores, _ = _predict(model, loader, device, args.amp)
    return kept.reset_index(drop=True), scores


def run(args: argparse.Namespace) -> dict:
    if HGTConv is None or LinkNeighborLoader is None:
        raise RuntimeError("PyTorch Geometric with LinkNeighborLoader and HGTConv is required for sampled Stage 3 HGT.")
    set_seed(args.seed)
    resolved = resolve_stage3_dir(args.modeling_dir)
    out_dir = ensure_dir(args.output_dir)
    try:
        stage_dir = resolved.root
        data = load_heterodata(stage_dir)
        data = _ensure_supervision_edge_type(data)
        node_maps = load_node_maps(stage_dir)
        train_df, valid_df, test_df = load_pairs(stage_dir, args.seed)
        c_tr, p_tr, y_tr, _ = encode_pairs(train_df, node_maps)
        c_va, p_va, y_va, _ = encode_pairs(valid_df, node_maps)
        c_te, p_te, y_te, test_kept = encode_pairs(test_df, node_maps)

        train_label_stats_before = label_stats(y_tr)
        # Compute class weights from the original training distribution, before any
        # oversampling. Otherwise balanced-batch oversampling makes the weights
        # collapse to 1:1 and the minority inactive/weak class remains under-penalized.
        class_weights = resolve_class_weights(y_tr, args)
        c_tr, p_tr, y_tr, balance_info = maybe_balance_training_tensors(c_tr, p_tr, y_tr, args)
        train_label_stats_after = label_stats(y_tr)

        train_loader = _make_loader(data, c_tr, p_tr, y_tr, args, shuffle=True)
        valid_loader = _make_loader(data, c_va, p_va, y_va, args, shuffle=False)
        test_loader = _make_loader(data, c_te, p_te, y_te, args, shuffle=False)

        device = get_device(args.device)
        if args.amp:
            print("WARNING: --amp was requested, but sampled HGT disables AMP internally to avoid pyg-lib grouped_matmul Float/Half dtype errors.")
        model = SampledHGTLinkPredictor(
            metadata=data.metadata(),
            node_counts=_node_counts(data),
            feature_dims=_feature_dims(data),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            featureless_mode=args.featureless_mode,
        ).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        save_json({
            "mode": "sampled_link_neighbor_loader",
            "target_edge_type": list(TARGET_EDGE_TYPE),
            "node_counts": _node_counts(data),
            "feature_dims": _feature_dims(data),
            "args": vars(args),
            "train_label_stats_before": train_label_stats_before,
            "train_label_stats_after": train_label_stats_after,
            "balance_info": balance_info,
            "class_weights": class_weights,
            "requested_amp": bool(args.amp),
            "effective_amp": False,
            "amp_note": "Sampled HGT forces FP32 because HGTConv/pyg-lib grouped_matmul can fail with Float/Half dtype mismatch.",
        }, out_dir / "hgt_sampled_metadata.json")

        best_score = float("-inf")
        best_threshold = float(args.threshold)
        best_epoch = 0
        bad_epochs = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = _run_epoch(model, train_loader, optim, device, args, class_weights)
            val_score, val_y = _predict(model, valid_loader, device, args.amp)
            if args.threshold_selection == "fixed":
                epoch_threshold = float(args.threshold)
                val_metrics = extended_binary_metrics(val_y.numpy(), val_score.numpy(), threshold=epoch_threshold)
            else:
                epoch_threshold, val_metrics = optimize_threshold(
                    val_y.numpy(),
                    val_score.numpy(),
                    metric=args.threshold_selection,
                    min_specificity=args.min_specificity,
                    min_recall=args.min_recall,
                )
            history.append({"epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in val_metrics.items()}})
            checkpoint_score = metric_for_checkpoint(val_metrics, args.early_stopping_metric)
            improved = checkpoint_score > (best_score + float(args.min_delta))
            if improved:
                best_score = checkpoint_score
                best_threshold = float(epoch_threshold)
                best_epoch = epoch
                bad_epochs = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_threshold": best_threshold,
                    "best_validation_metrics": val_metrics,
                    "class_weights": class_weights,
                    "balance_info": balance_info,
                }, out_dir / "best_model.pt")
            else:
                bad_epochs += 1
            print(f"epoch={epoch:03d} loss={loss:.4f} threshold={epoch_threshold:.2f} checkpoint_{args.early_stopping_metric}={checkpoint_score:.4f} valid={val_metrics}")
            if int(args.patience) > 0 and bad_epochs >= int(args.patience):
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_threshold={best_threshold:.2f}, best_score={best_score:.4f}")
                break

        ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_threshold = float(ckpt.get("best_threshold", best_threshold))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        test_score, test_y = _predict(model, test_loader, device, args.amp)
        metrics = extended_binary_metrics(test_y.numpy(), test_score.numpy(), threshold=best_threshold)
        operating_points = operating_point_metrics(
            test_y.numpy(),
            test_score.numpy(),
            extended_binary_metrics,
            min_specificity=max(args.min_specificity, args.report_min_specificity),
            high_specificity=args.report_high_specificity,
            min_recall=args.report_min_recall,
        )
        balanced_metrics = balanced_diagnostic_metrics(
            test_y.numpy(),
            test_score.numpy(),
            extended_binary_metrics,
            threshold=best_threshold,
            seed=args.seed,
            max_per_class=args.balanced_eval_max_per_class,
        )
        per_target = per_group_binary_metrics(
            test_kept.reset_index(drop=True),
            test_y.numpy(),
            test_score.numpy(),
            extended_binary_metrics,
            threshold=best_threshold,
            group_col=args.target_group_column if args.target_group_column != "auto" else None,
        )
        per_target_metrics_file = write_dataframe_if_not_empty(per_target, out_dir / "per_target_metrics.csv")

        cand_df = load_candidate_pairs(stage_dir)
        if cand_df is not None and args.score_candidates:
            cand = cand_df.head(args.max_candidate_pairs) if args.max_candidate_pairs > 0 else cand_df
            score_rows, score = _score_rows(model, data, cand, node_maps, args, device)
        else:
            score_rows = test_kept.copy().reset_index(drop=True)
            score = test_score

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        preds = score_rows.copy().reset_index(drop=True)
        preds["score"] = score.numpy()
        preds["predicted_label"] = (score.numpy() >= best_threshold).astype(int)
        preds["decision_threshold"] = best_threshold
        preds["model"] = "stage3_hgt_sampled"
        preds["stage"] = "Stage 3 — sampled HGT"
        preds.to_csv(out_dir / "predictions.csv", index=False)

        summary = {
            "stage": "Stage 3 — sampled HGT",
            "model": "stage3_hgt_sampled",
            "status": "trained",
            "metrics": metrics,
            **{k: v for k, v in metrics.items()},
            "prediction_rows_written": int(len(preds)),
            "predictions_file": str(out_dir / "predictions.csv"),
            "model_file": str(out_dir / "best_model.pt"),
            "training_history_file": str(out_dir / "training_history.csv"),
            "best_epoch": int(best_epoch),
            "selected_threshold": float(best_threshold),
            "early_stopping_metric": args.early_stopping_metric,
            "threshold_selection": args.threshold_selection,
            "min_specificity": float(args.min_specificity),
            "min_recall": float(args.min_recall),
            "operating_points": operating_points,
            "balanced_diagnostic_metrics": balanced_metrics,
            "per_target_metrics": per_target.to_dict(orient="records") if per_target_metrics_file else [],
            "per_target_metrics_file": per_target_metrics_file,
            "class_weights": class_weights,
            "balance_info": balance_info,
            "requested_amp": bool(args.amp),
            "effective_amp": False,
        }
        if args.export_neo4j:
            summary["neo4j_export"] = export_predictions_dataframe(preds, model_name="stage3_hgt_sampled", max_rows=args.max_neo4j_predictions)
        save_json(summary, out_dir / "metrics.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sampled HGT + MLP decoder for compound-CYP450 link prediction.")
    parser.add_argument("--modeling-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=4096)  # kept for CLI compatibility
    parser.add_argument("--hidden-dim", "--hidden-channels", dest="hidden_dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--num-neighbors", default="10", help="Comma-separated neighbors per GNN layer, e.g. '10' or '10,5'.")
    parser.add_argument("--featureless-mode", choices=["type", "global"], default="type", help="Use one type vector for featureless nodes or global node embeddings.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal", "bpr", "pairwise_bpr", "weighted_bce_bpr", "bce_bpr"], default="weighted_bce_bpr")
    parser.add_argument("--class-weighting", choices=["none", "balanced", "negative_ratio"], default="negative_ratio")
    parser.add_argument("--negative-class-weight", type=float, default=None, help="Manual weight for label 0 inactive/weak examples.")
    parser.add_argument("--positive-class-weight", type=float, default=None, help="Manual weight for label 1 active examples.")
    parser.add_argument("--balanced-batches", dest="balanced_batches", action="store_true", default=True, help="Oversample minority supervision class during training.")
    parser.add_argument("--no-balanced-batches", dest="balanced_batches", action="store_false", help="Disable supervision-pair balancing.")
    parser.add_argument("--balance-ratio", type=float, default=1.0, help="Target negative:positive ratio when --balanced-batches is enabled.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--bpr-weight", type=float, default=0.5, help="Weight of pairwise BPR loss when --loss weighted_bce_bpr is used.")
    parser.add_argument("--focal-alpha", type=float, default=-1.0, help="Optional focal alpha for positives. Use -1 to disable alpha.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-selection", choices=["fixed", "mcc", "balanced_accuracy", "youden", "f1", "recall", "specificity"], default="mcc")
    parser.add_argument("--min-specificity", type=float, default=0.50, help="Validation threshold must reach this specificity when possible.")
    parser.add_argument("--min-recall", type=float, default=0.0, help="Validation threshold must reach this recall when possible.")
    parser.add_argument("--report-min-specificity", type=float, default=0.50)
    parser.add_argument("--report-high-specificity", type=float, default=0.80)
    parser.add_argument("--report-min-recall", type=float, default=0.80)
    parser.add_argument("--target-group-column", default="auto", help="Column used for per-CYP/per-target metrics.")
    parser.add_argument("--balanced-eval-max-per-class", type=int, default=0)
    parser.add_argument("--early-stopping-metric", choices=["mcc", "balanced_accuracy", "youden", "roc_auc", "average_precision", "f1", "specificity", "recall"], default="mcc")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--score-candidates", action="store_true")
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--export-neo4j", action="store_true")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
