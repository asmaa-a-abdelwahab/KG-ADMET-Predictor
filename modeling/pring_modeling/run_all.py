from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from loguru import logger

from .common import ensure_dir, save_json
from .compare import compare_metrics, visualize


def _run_stage1(args, root: Path, export: bool) -> dict:
    from .stage1_tabular import build_parser, run
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(root / "stage1_tabular"),
        "--report-dir", str(args.report_dir),
        "--target-column", args.target_column,
        "--max-training-rows", str(args.max_training_rows),
        "--max-scoring-rows", str(args.max_scoring_rows),
        "--max-predictions-file-rows", str(args.max_predictions_file_rows),
        "--threshold", str(args.threshold),
        "--n-estimators", str(args.n_estimators),
        "--n-jobs", str(args.n_jobs),
    ]
    if args.max_depth is not None:
        argv += ["--max-depth", str(args.max_depth)]
    if export:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def _run_stage2(args, root: Path, export: bool) -> dict:
    from .stage2_kge import build_parser, run
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(root / f"stage2_{args.kge_model}"),
        "--model", args.kge_model,
        "--epochs", str(args.stage2_epochs),
        "--batch-size", str(args.batch_size),
        "--dim", str(args.kge_dim),
        "--max-graph-train-triples", str(args.max_graph_train_triples),
        "--max-candidate-triples", str(args.max_candidate_triples),
    ]
    if export:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def _run_stage3(args, root: Path, export: bool) -> dict:
    if args.stage3_model == "hgt":
        from .stage3_hgt import build_parser, run
        out = root / "stage3_hgt"
    else:
        from .stage3_rgcn import build_parser, run
        out = root / "stage3_rgcn"
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(out),
        "--epochs", str(args.stage3_epochs),
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--threshold", str(args.threshold),
    ]
    if args.score_candidates:
        argv += ["--score-candidates", "--max-candidate-pairs", str(args.max_candidate_pairs)]
    if export:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> dict:
    output_root = ensure_dir(args.output_dir)
    report_dir = ensure_dir(args.report_dir)
    export = bool(args.export_neo4j)
    summaries: dict[str, dict] = {}
    for stage in args.stages:
        try:
            if stage == "stage1":
                summaries[stage] = _run_stage1(args, output_root, export)
            elif stage == "stage2":
                summaries[stage] = _run_stage2(args, output_root, export)
            elif stage == "stage3":
                summaries[stage] = _run_stage3(args, output_root, export)
            else:
                logger.warning("Unknown stage requested: {}", stage)
        except Exception as exc:
            summaries[stage] = {"status": "skipped_or_failed", "error": str(exc)}
            logger.warning("{} did not complete: {}", stage, exc)
            if not args.continue_on_error:
                raise
    try:
        comparison_dir = ensure_dir(report_dir / "comparison")
        df = compare_metrics([], str(output_root), str(comparison_dir), args.primary_metric)
        visualize(str(comparison_dir / "model_comparison.csv"), str(comparison_dir / "figures"))
        summaries["comparison"] = {"status": "written", "comparison_csv": str(comparison_dir / "model_comparison.csv")}
    except Exception as exc:
        summaries["comparison"] = {"status": "skipped_or_failed", "error": str(exc)}
    save_json(summaries, output_root / "run_all_summary.json")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run integrated PRING CYP450 modeling stages inside the modeling Docker image.")
    p.add_argument("--modeling-dir", default=os.getenv("PRING_RUN_DIR", "/runs/current"), help="PRING run, graph/ml/modeling folder, standalone stage folder, or zip.")
    p.add_argument("--output-dir", default=os.getenv("MODEL_OUTPUT_DIR", "/models"))
    p.add_argument("--report-dir", default=os.getenv("MODEL_REPORT_DIR", "/reports/modeling"))
    p.add_argument("--stages", nargs="+", default=os.getenv("MODEL_STAGES", "stage1 stage2").split(), choices=["stage1", "stage2", "stage3"])
    p.add_argument("--target-column", default=os.getenv("MODEL_TARGET_COLUMN", "label"))
    p.add_argument("--threshold", type=float, default=float(os.getenv("MODEL_THRESHOLD", "0.5")))
    p.add_argument("--max-training-rows", type=int, default=int(os.getenv("MODEL_MAX_TRAINING_ROWS", "100000")))
    p.add_argument("--max-scoring-rows", type=int, default=int(os.getenv("MODEL_MAX_SCORING_ROWS", "100000")))
    p.add_argument("--max-predictions-file-rows", type=int, default=int(os.getenv("MODEL_MAX_PREDICTIONS_FILE_ROWS", "100000")))
    p.add_argument("--n-estimators", type=int, default=int(os.getenv("MODEL_N_ESTIMATORS", "200")))
    p.add_argument("--n-jobs", type=int, default=int(os.getenv("MODEL_N_JOBS", "1")))
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MODEL_MAX_DEPTH")) if os.getenv("MODEL_MAX_DEPTH") else None)
    p.add_argument("--kge-model", choices=["distmult", "complex", "rotate"], default=os.getenv("MODEL_KGE_MODEL", "rotate"))
    p.add_argument("--stage2-epochs", type=int, default=int(os.getenv("MODEL_STAGE2_EPOCHS", "20")))
    p.add_argument("--kge-dim", type=int, default=int(os.getenv("MODEL_KGE_DIM", "64")))
    p.add_argument("--max-graph-train-triples", type=int, default=int(os.getenv("MODEL_MAX_GRAPH_TRAIN_TRIPLES", "500000")))
    p.add_argument("--max-candidate-triples", type=int, default=int(os.getenv("MODEL_MAX_CANDIDATE_TRIPLES", "100000")))
    p.add_argument("--stage3-model", choices=["rgcn", "hgt"], default=os.getenv("MODEL_STAGE3_MODEL", "rgcn"))
    p.add_argument("--stage3-epochs", type=int, default=int(os.getenv("MODEL_STAGE3_EPOCHS", "50")))
    p.add_argument("--hidden-dim", type=int, default=int(os.getenv("MODEL_HIDDEN_DIM", "128")))
    p.add_argument("--num-layers", type=int, default=int(os.getenv("MODEL_NUM_LAYERS", "2")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("MODEL_BATCH_SIZE", "4096")))
    p.add_argument("--score-candidates", action="store_true", default=os.getenv("MODEL_SCORE_CANDIDATES", "false").lower() == "true")
    p.add_argument("--max-candidate-pairs", type=int, default=int(os.getenv("MODEL_MAX_CANDIDATE_PAIRS", "100000")))
    p.add_argument("--export-neo4j", action="store_true", default=os.getenv("MODEL_EXPORT_TO_NEO4J", "true").lower() == "true")
    p.add_argument("--max-neo4j-predictions", type=int, default=int(os.getenv("MODEL_MAX_NEO4J_PREDICTIONS", "25000")))
    p.add_argument("--primary-metric", default=os.getenv("MODEL_PRIMARY_METRIC", "average_precision"))
    p.add_argument("--continue-on-error", action="store_true", default=os.getenv("MODEL_CONTINUE_ON_ERROR", "true").lower() == "true")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
