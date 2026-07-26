from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import get_logger

logger = get_logger(__name__)

from .common import ensure_dir, save_json, read_table
from .compare import compare_metrics, visualize
from .neo4j_export import export_predictions_file


def _metric_value(summary: dict[str, Any], primary_metric: str) -> float:
    """Return a comparable score for model selection.

    The preferred metric is average precision because the CYP450 interaction
    task is imbalanced. If the requested metric is missing, fall back to common
    binary metrics before returning -inf.
    """
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    candidates = [primary_metric, "mcc", "balanced_accuracy", "roc_auc", "average_precision", "specificity", "f1", "accuracy"]
    for key in candidates:
        value = summary.get(key, metrics.get(key))
        if value is None:
            continue
        try:
            if value == value:  # not NaN
                return float(value)
        except Exception:
            continue
    return float("-inf")


def _safe_name(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _write_best(stage_key: str, summaries: list[dict[str, Any]], root: Path, primary_metric: str) -> dict[str, Any]:
    stage_best_dir = ensure_dir(root / "best")
    valid = [s for s in summaries if s.get("status") not in {"skipped_or_failed", "failed"}]
    if not valid:
        result = {"stage": stage_key, "status": "no_successful_models"}
        save_json(result, stage_best_dir / f"{stage_key}_best.json")
        return result
    best = max(valid, key=lambda s: _metric_value(s, primary_metric))
    best = dict(best)
    best["selection_metric"] = primary_metric
    best["selection_score"] = _metric_value(best, primary_metric)
    pred_file = best.get("predictions_file")
    if pred_file and Path(pred_file).exists():
        dest = stage_best_dir / f"{stage_key}_best_predictions.csv"
        shutil.copy2(pred_file, dest)
        best["best_predictions_file"] = str(dest)
    model_file = best.get("model_file")
    if model_file and Path(model_file).exists():
        dest_model = stage_best_dir / f"{stage_key}_best_model{Path(model_file).suffix}"
        try:
            shutil.copy2(model_file, dest_model)
            best["best_model_file"] = str(dest_model)
        except Exception as exc:
            best["best_model_copy_warning"] = str(exc)
    save_json(best, stage_best_dir / f"{stage_key}_best.json")
    return best


def _export_best_if_requested(best: dict[str, Any], export: bool, max_rows: int) -> dict[str, Any] | None:
    if not export:
        return None
    pred_file = best.get("best_predictions_file") or best.get("predictions_file")
    if not pred_file or not Path(pred_file).exists():
        return {"exported": 0, "reason": "best prediction file missing"}
    return export_predictions_file(pred_file, model_name=str(best.get("model") or "best_model"), max_rows=max_rows)


def _run_stage1_model(args: argparse.Namespace, root: Path, classifier: str, export_individual: bool) -> dict[str, Any]:
    from .stage1_tabular import build_parser, run

    out_dir = root / f"stage1_{_safe_name(classifier)}"
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(out_dir),
        "--report-dir", str(args.report_dir),
        "--target-column", args.target_column,
        "--max-training-rows", str(args.max_training_rows),
        "--max-scoring-rows", str(args.max_scoring_rows),
        "--max-predictions-file-rows", str(args.max_predictions_file_rows),
        "--prediction-scope", args.prediction_scope,
        "--threshold", str(args.threshold),
        "--threshold-selection", args.threshold_selection,
        "--min-specificity", str(args.min_specificity),
        "--min-recall", str(args.min_recall),
        "--report-min-specificity", str(args.report_min_specificity),
        "--report-high-specificity", str(args.report_high_specificity),
        "--report-min-recall", str(args.report_min_recall),
        "--balanced-eval-max-per-class", str(args.balanced_eval_max_per_class),
        "--feature-policy", args.stage1_feature_policy,
        "--classifier", classifier,
        "--n-estimators", str(args.n_estimators),
        "--n-jobs", str(args.n_jobs),
        "--min-samples-leaf", str(args.min_samples_leaf),
        "--cv-folds", str(args.stage1_cv_folds),
    ]
    if args.stage1_rdkit_features:
        argv += ["--rdkit-features", "--smiles-column", args.smiles_column, "--rdkit-fingerprint-bits", str(args.rdkit_fingerprint_bits)]
    if args.max_depth is not None:
        argv += ["--max-depth", str(args.max_depth)]
    if export_individual:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def _run_stage2_model(args: argparse.Namespace, root: Path, model_name: str, export_individual: bool) -> dict[str, Any]:
    from .stage2_kge import build_parser, run

    out_dir = root / f"stage2_{_safe_name(model_name)}"
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(out_dir),
        "--model", model_name,
        "--epochs", str(args.stage2_epochs),
        "--batch-size", str(args.batch_size),
        "--score-batch-size", str(args.score_batch_size),
        "--dim", str(args.kge_dim),
        "--margin", str(args.kge_margin),
        "--lr", str(args.lr),
        "--max-graph-train-triples", str(args.max_graph_train_triples),
        "--max-candidate-triples", str(args.max_candidate_triples),
        "--target-train-repeat", str(args.stage2_target_train_repeat),
        "--loss", args.stage2_loss,
        "--optimizer", args.stage2_optimizer,
        "--negatives-per-positive", str(args.stage2_negatives_per_positive),
        "--eval-negatives-per-positive", str(args.stage2_eval_negatives_per_positive),
        "--eval-every", str(args.stage2_eval_every),
        "--patience", str(args.stage2_patience),
        "--checkpoint-metric", args.stage2_checkpoint_metric,
        "--supervised-decoder", args.stage2_supervised_decoder,
        "--supervised-threshold-selection", args.stage2_supervised_threshold_selection,
        "--supervised-min-specificity", str(args.min_specificity),
        "--supervised-min-recall", str(args.min_recall),
        "--report-min-specificity", str(args.report_min_specificity),
        "--report-high-specificity", str(args.report_high_specificity),
        "--report-min-recall", str(args.report_min_recall),
        "--balanced-eval-max-per-class", str(args.balanced_eval_max_per_class),
        "--num-workers", str(args.num_workers),
        "--n-jobs", str(args.n_jobs),
        "--device", args.device,
    ]
    argv += ["--sparse-embeddings"] if args.stage2_sparse_embeddings else ["--no-sparse-embeddings"]
    argv += ["--score-candidates"] if args.stage2_score_candidates else ["--no-score-candidates"]
    argv += ["--export-eval-predictions"] if args.stage2_export_eval_predictions else ["--no-export-eval-predictions"]
    argv += ["--train-supervised-decoder"] if args.stage2_train_supervised_decoder else ["--no-train-supervised-decoder"]
    argv += ["--save-mappings"] if args.stage2_save_mappings else ["--no-save-mappings"]
    argv += ["--attach-entity-refs"] if args.stage2_attach_entity_refs else ["--no-attach-entity-refs"]
    if export_individual:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def _run_stage3_model(args: argparse.Namespace, root: Path, model_name: str, export_individual: bool) -> dict[str, Any]:
    if model_name == "hgt":
        from .stage3_hgt import build_parser, run
    else:
        from .stage3_rgcn import build_parser, run
    out_dir = root / f"stage3_{_safe_name(model_name)}"
    argv = [
        "--modeling-dir", str(args.modeling_dir),
        "--output-dir", str(out_dir),
        "--epochs", str(args.stage3_epochs),
        "--batch-size", str(args.batch_size),
        "--score-batch-size", str(args.score_batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--threshold", str(args.threshold),
        "--device", args.device,
        "--num-neighbors", str(args.stage3_num_neighbors),
        "--featureless-mode", args.stage3_featureless_mode,
        "--loss", args.stage3_loss,
        "--bpr-weight", str(args.stage3_bpr_weight),
        "--class-weighting", args.stage3_class_weighting,
        "--balance-ratio", str(args.stage3_balance_ratio),
        "--balanced-batches" if args.stage3_balanced_batches else "--no-balanced-batches",
        "--threshold-selection", args.threshold_selection,
        "--min-specificity", str(args.min_specificity),
        "--min-recall", str(args.min_recall),
        "--report-min-specificity", str(args.report_min_specificity),
        "--report-high-specificity", str(args.report_high_specificity),
        "--report-min-recall", str(args.report_min_recall),
        "--balanced-eval-max-per-class", str(args.balanced_eval_max_per_class),
        "--early-stopping-metric", args.stage3_early_stopping_metric,
        "--patience", str(args.stage3_patience),
    ]
    if model_name == "hgt":
        argv += ["--heads", str(args.hgt_heads)]
    else:
        argv += ["--num-bases", str(args.rgcn_num_bases)]
    if args.score_candidates:
        argv += ["--score-candidates", "--max-candidate-pairs", str(args.max_candidate_pairs)]
    if export_individual:
        argv += ["--export-neo4j", "--max-neo4j-predictions", str(args.max_neo4j_predictions)]
    return run(build_parser().parse_args(argv))


def _run_ensemble(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from .ensemble import build_parser, run

    out_dir = root / "ensemble_stacked"
    argv = [
        "--outputs-root", str(root),
        "--output-dir", str(out_dir),
        "--meta-classifier", args.ensemble_meta_classifier,
        "--threshold-selection", args.threshold_selection,
        "--min-specificity", str(args.min_specificity),
        "--min-recall", str(args.min_recall),
        "--report-min-specificity", str(args.report_min_specificity),
        "--report-high-specificity", str(args.report_high_specificity),
        "--report-min-recall", str(args.report_min_recall),
        "--balanced-eval-max-per-class", str(args.balanced_eval_max_per_class),
        "--n-jobs", str(args.n_jobs),
        "--seed", str(args.seed),
    ]
    return run(build_parser().parse_args(argv))


def _run_explanations(args: argparse.Namespace, output_root: Path, best_by_stage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from .stage4_explain import build_parser, run

    out_root = ensure_dir(Path(args.report_dir) / "stage4_explainability")
    results: dict[str, Any] = {}
    # Explain the best model from each successful stage. This gives stage-specific
    # evidence reports and avoids relying only on one overall winner.
    for stage_key, best in best_by_stage.items():
        pred_file = best.get("best_predictions_file") or best.get("predictions_file")
        if not pred_file or not Path(pred_file).exists():
            results[stage_key] = {"status": "skipped", "reason": "no prediction file"}
            continue
        argv = [
            "--predictions", str(pred_file),
            "--neo4j-uri", args.neo4j_uri,
            "--neo4j-user", args.neo4j_user,
            "--neo4j-password", args.neo4j_password,
            "--database", args.neo4j_database,
            "--output-dir", str(out_root / stage_key),
            "--limit", str(args.explain_limit),
            "--model-output-dir", str(Path(pred_file).parent.parent if "best" in Path(pred_file).parts else Path(pred_file).parent),
            "--stage-name", stage_key,
        ]
        try:
            results[stage_key] = run(build_parser().parse_args(argv))
        except Exception as exc:
            results[stage_key] = {"status": "skipped_or_failed", "error": str(exc)}
            if not args.continue_on_error:
                raise
    save_json(results, out_root / "stage4_explainability_summary.json")
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_dir)
    report_dir = ensure_dir(args.report_dir)
    export_individual = args.export_neo4j and args.export_scope == "all"
    export_best = args.export_neo4j and args.export_scope in {"best", "best_only", "all"}

    summaries: dict[str, Any] = {
        "config": {
            "modeling_dir": str(args.modeling_dir),
            "output_dir": str(output_root),
            "report_dir": str(report_dir),
            "stages": args.stages,
            "export_scope": args.export_scope,
            "primary_metric": args.primary_metric,
        }
    }
    best_by_stage: dict[str, dict[str, Any]] = {}

    if "stage1" in args.stages:
        stage_summaries: list[dict[str, Any]] = []
        for classifier in args.stage1_models:
            try:
                logger.info("Running Stage 1 classifier: {}", classifier)
                stage_summaries.append(_run_stage1_model(args, output_root, classifier, export_individual))
            except Exception as exc:
                logger.warning("Stage 1 model {} failed: {}", classifier, exc)
                stage_summaries.append({"stage": "Stage 1", "model": f"stage1_{classifier}", "status": "skipped_or_failed", "error": str(exc)})
                if not args.continue_on_error:
                    raise
        summaries["stage1"] = stage_summaries
        best = _write_best("stage1", stage_summaries, output_root, args.primary_metric)
        if export_best and best.get("status") != "no_successful_models":
            best["neo4j_export_best"] = _export_best_if_requested(best, True, args.max_neo4j_predictions)
            save_json(best, output_root / "best" / "stage1_best.json")
        best_by_stage["stage1"] = best

    if "stage2" in args.stages:
        stage_summaries = []
        for model_name in args.stage2_models:
            try:
                logger.info("Running Stage 2 KGE model: {}", model_name)
                stage_summaries.append(_run_stage2_model(args, output_root, model_name, export_individual))
            except Exception as exc:
                logger.warning("Stage 2 model {} failed: {}", model_name, exc)
                stage_summaries.append({"stage": "Stage 2", "model": f"stage2_{model_name}", "status": "skipped_or_failed", "error": str(exc)})
                if not args.continue_on_error:
                    raise
        summaries["stage2"] = stage_summaries
        best = _write_best("stage2", stage_summaries, output_root, args.primary_metric)
        if export_best and best.get("status") != "no_successful_models":
            best["neo4j_export_best"] = _export_best_if_requested(best, True, args.max_neo4j_predictions)
            save_json(best, output_root / "best" / "stage2_best.json")
        best_by_stage["stage2"] = best

    if "stage3" in args.stages:
        stage_summaries = []
        for model_name in args.stage3_models:
            try:
                logger.info("Running Stage 3 GNN model: {}", model_name)
                stage_summaries.append(_run_stage3_model(args, output_root, model_name, export_individual))
            except Exception as exc:
                logger.warning("Stage 3 model {} failed: {}", model_name, exc)
                stage_summaries.append({"stage": "Stage 3", "model": f"stage3_{model_name}", "status": "skipped_or_failed", "error": str(exc)})
                if not args.continue_on_error:
                    raise
        summaries["stage3"] = stage_summaries
        best = _write_best("stage3", stage_summaries, output_root, args.primary_metric)
        if export_best and best.get("status") != "no_successful_models":
            best["neo4j_export_best"] = _export_best_if_requested(best, True, args.max_neo4j_predictions)
            save_json(best, output_root / "best" / "stage3_best.json")
        best_by_stage["stage3"] = best

    if "ensemble" in args.stages or args.run_ensemble:
        try:
            logger.info("Running stacked ensemble")
            ensemble_summary = _run_ensemble(args, output_root)
            summaries["ensemble"] = ensemble_summary
            best = _write_best("ensemble", [ensemble_summary], output_root, args.primary_metric)
            best_by_stage["ensemble"] = best
        except Exception as exc:
            logger.warning("Ensemble failed: {}", exc)
            summaries["ensemble"] = {"stage": "Ensemble", "model": "ensemble_stacked", "status": "skipped_or_failed", "error": str(exc)}
            if not args.continue_on_error:
                raise

    # Comparison and visualizations across every metrics.json under output_root.
    try:
        comparison_dir = ensure_dir(report_dir / "comparison")
        df = compare_metrics([], str(output_root), str(comparison_dir), args.primary_metric)
        figures = visualize(str(comparison_dir / "model_comparison.csv"), str(comparison_dir / "figures"))
        summaries["comparison"] = {
            "status": "written",
            "comparison_csv": str(comparison_dir / "model_comparison.csv"),
            "figures": [str(p) for p in figures],
        }
        # Overall best across all successful stages.
        if len(df):
            row = df.sort_values("rank").iloc[0].to_dict()
            summaries["overall_best"] = row
            save_json(row, output_root / "best" / "overall_best.json")
    except Exception as exc:
        summaries["comparison"] = {"status": "skipped_or_failed", "error": str(exc)}
        if not args.continue_on_error:
            raise

    if "stage4" in args.stages or args.run_stage4:
        summaries["stage4"] = _run_explanations(args, output_root, best_by_stage)

    save_json(best_by_stage, output_root / "best" / "best_by_stage.json")
    save_json(summaries, output_root / "run_all_summary.json")
    return summaries


def _split_env(name: str, default: str) -> list[str]:
    return [x for x in os.getenv(name, default).replace(",", " ").split() if x]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run, compare, explain, and optionally export all PRING CYP450 modeling stages.")
    p.add_argument("--modeling-dir", default=os.getenv("PRING_RUN_DIR", "/runs/current"), help="Full PRING run, graph/ml/modeling folder, standalone stage folder, or zip.")
    p.add_argument("--output-dir", default=os.getenv("MODEL_OUTPUT_DIR", "/models"))
    p.add_argument("--report-dir", default=os.getenv("MODEL_REPORT_DIR", "/reports/modeling"))
    p.add_argument("--stages", nargs="+", default=_split_env("MODEL_STAGES", "stage1 stage2 stage3 ensemble stage4"), choices=["stage1", "stage2", "stage3", "ensemble", "stage4"])
    p.add_argument("--primary-metric", default=os.getenv("MODEL_PRIMARY_METRIC", "mcc"))
    p.add_argument("--continue-on-error", action="store_true", default=os.getenv("MODEL_CONTINUE_ON_ERROR", "true").lower() == "true")

    # Stage 1
    p.add_argument("--stage1-models", nargs="+", default=_split_env("MODEL_STAGE1_MODELS", "random_forest extra_trees"), choices=["random_forest", "extra_trees", "hist_gradient_boosting", "logistic_regression"])
    p.add_argument("--target-column", default=os.getenv("MODEL_TARGET_COLUMN", "label"))
    p.add_argument("--threshold", type=float, default=float(os.getenv("MODEL_THRESHOLD", "0.5")))
    p.add_argument("--threshold-selection", default=os.getenv("MODEL_THRESHOLD_SELECTION", "mcc"), choices=["mcc", "balanced_accuracy", "youden", "f1", "accuracy", "recall", "specificity"])
    p.add_argument("--min-specificity", type=float, default=float(os.getenv("MODEL_MIN_SPECIFICITY", "0.50")))
    p.add_argument("--min-recall", type=float, default=float(os.getenv("MODEL_MIN_RECALL", "0.0")))
    p.add_argument("--report-min-specificity", type=float, default=float(os.getenv("MODEL_REPORT_MIN_SPECIFICITY", "0.50")))
    p.add_argument("--report-high-specificity", type=float, default=float(os.getenv("MODEL_REPORT_HIGH_SPECIFICITY", "0.80")))
    p.add_argument("--report-min-recall", type=float, default=float(os.getenv("MODEL_REPORT_MIN_RECALL", "0.80")))
    p.add_argument("--balanced-eval-max-per-class", type=int, default=int(os.getenv("MODEL_BALANCED_EVAL_MAX_PER_CLASS", "0")))
    p.add_argument("--stage1-feature-policy", default=os.getenv("MODEL_STAGE1_FEATURE_POLICY", "leakage_safe"), choices=["leakage_safe", "structural_only", "allow_all"])
    p.add_argument("--max-training-rows", type=int, default=int(os.getenv("MODEL_MAX_TRAINING_ROWS", "0")))
    p.add_argument("--max-scoring-rows", type=int, default=int(os.getenv("MODEL_MAX_SCORING_ROWS", "0")))
    p.add_argument("--max-predictions-file-rows", type=int, default=int(os.getenv("MODEL_MAX_PREDICTIONS_FILE_ROWS", "0")))
    p.add_argument("--prediction-scope", default=os.getenv("MODEL_PREDICTION_SCOPE", "candidates"), choices=["supervised", "candidates"])
    p.add_argument("--n-estimators", type=int, default=int(os.getenv("MODEL_N_ESTIMATORS", "300")))
    p.add_argument("--n-jobs", type=int, default=int(os.getenv("MODEL_N_JOBS", "1")))
    p.add_argument("--min-samples-leaf", type=int, default=int(os.getenv("MODEL_MIN_SAMPLES_LEAF", "2")))
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MODEL_MAX_DEPTH")) if os.getenv("MODEL_MAX_DEPTH") else None)
    p.add_argument("--stage1-cv-folds", type=int, default=int(os.getenv("MODEL_STAGE1_CV_FOLDS", "5")))
    p.add_argument("--stage1-rdkit-features", action="store_true", default=os.getenv("MODEL_STAGE1_RDKIT_FEATURES", "false").lower() == "true")
    p.add_argument("--smiles-column", default=os.getenv("MODEL_SMILES_COLUMN", "auto"))
    p.add_argument("--rdkit-fingerprint-bits", type=int, default=int(os.getenv("MODEL_RDKIT_FINGERPRINT_BITS", "0")))

    # Stage 2
    p.add_argument("--stage2-models", nargs="+", default=_split_env("MODEL_STAGE2_MODELS", "distmult complex rotate"), choices=["distmult", "complex", "rotate"])
    p.add_argument("--stage2-epochs", type=int, default=int(os.getenv("MODEL_STAGE2_EPOCHS", "30")))
    p.add_argument("--kge-dim", type=int, default=int(os.getenv("MODEL_KGE_DIM", "128")))
    p.add_argument("--kge-margin", type=float, default=float(os.getenv("MODEL_KGE_MARGIN", "6.0")))
    p.add_argument("--max-graph-train-triples", type=int, default=int(os.getenv("MODEL_MAX_GRAPH_TRAIN_TRIPLES", "0")))
    p.add_argument("--max-candidate-triples", type=int, default=int(os.getenv("MODEL_MAX_CANDIDATE_TRIPLES", "100000")))
    p.add_argument("--stage2-target-train-repeat", type=int, default=int(os.getenv("MODEL_STAGE2_TARGET_TRAIN_REPEAT", "5")))
    p.add_argument("--stage2-loss", default=os.getenv("MODEL_STAGE2_LOSS", "softplus"), choices=["softplus", "bce", "margin"])
    p.add_argument("--stage2-optimizer", default=os.getenv("MODEL_STAGE2_OPTIMIZER", "auto"), choices=["auto", "sparse_adam", "adam", "adamw"])
    p.add_argument("--stage2-negatives-per-positive", type=int, default=int(os.getenv("MODEL_STAGE2_NEGATIVES_PER_POSITIVE", "1")))
    p.add_argument("--stage2-eval-negatives-per-positive", type=int, default=int(os.getenv("MODEL_STAGE2_EVAL_NEGATIVES_PER_POSITIVE", "1")))
    p.add_argument("--stage2-eval-every", type=int, default=int(os.getenv("MODEL_STAGE2_EVAL_EVERY", "1")))
    p.add_argument("--stage2-patience", type=int, default=int(os.getenv("MODEL_STAGE2_PATIENCE", "5")))
    p.add_argument("--stage2-checkpoint-metric", default=os.getenv("MODEL_STAGE2_CHECKPOINT_METRIC", "average_precision"), choices=["average_precision", "roc_auc", "f1", "accuracy", "balanced_accuracy", "mcc"])
    p.add_argument("--stage2-sparse-embeddings", action="store_true", default=os.getenv("MODEL_STAGE2_SPARSE_EMBEDDINGS", "true").lower() == "true")
    p.add_argument("--stage2-score-candidates", action="store_true", default=os.getenv("MODEL_STAGE2_SCORE_CANDIDATES", os.getenv("MODEL_SCORE_CANDIDATES", "false")).lower() == "true")
    p.add_argument("--stage2-save-mappings", action="store_true", default=os.getenv("MODEL_STAGE2_SAVE_MAPPINGS", "false").lower() == "true")
    p.add_argument("--stage2-attach-entity-refs", action="store_true", default=os.getenv("MODEL_STAGE2_ATTACH_ENTITY_REFS", "false").lower() == "true")
    p.add_argument("--stage2-export-eval-predictions", action="store_true", default=os.getenv("MODEL_STAGE2_EXPORT_EVAL_PREDICTIONS", "true").lower() == "true")
    p.add_argument("--stage2-train-supervised-decoder", action="store_true", default=os.getenv("MODEL_STAGE2_TRAIN_SUPERVISED_DECODER", "true").lower() == "true")
    p.add_argument("--stage2-supervised-decoder", default=os.getenv("MODEL_STAGE2_SUPERVISED_DECODER", "hist_gradient_boosting"), choices=["hist_gradient_boosting", "logistic_regression", "random_forest", "extra_trees"])
    p.add_argument("--stage2-supervised-threshold-selection", default=os.getenv("MODEL_STAGE2_SUPERVISED_THRESHOLD_SELECTION", "mcc"), choices=["mcc", "balanced_accuracy", "youden", "f1", "accuracy", "recall", "specificity"])

    # Stage 3
    p.add_argument("--stage3-models", nargs="+", default=_split_env("MODEL_STAGE3_MODELS", "rgcn hgt"), choices=["rgcn", "hgt"])
    p.add_argument("--stage3-epochs", type=int, default=int(os.getenv("MODEL_STAGE3_EPOCHS", "50")))
    p.add_argument("--hidden-dim", type=int, default=int(os.getenv("MODEL_HIDDEN_DIM", "128")))
    p.add_argument("--num-layers", type=int, default=int(os.getenv("MODEL_NUM_LAYERS", "2")))
    p.add_argument("--dropout", type=float, default=float(os.getenv("MODEL_DROPOUT", "0.2")))
    p.add_argument("--rgcn-num-bases", type=int, default=int(os.getenv("MODEL_RGCN_NUM_BASES", "30")))
    p.add_argument("--hgt-heads", type=int, default=int(os.getenv("MODEL_HGT_HEADS", "2")))
    p.add_argument("--score-candidates", action="store_true", default=os.getenv("MODEL_SCORE_CANDIDATES", "true").lower() == "true")
    p.add_argument("--max-candidate-pairs", type=int, default=int(os.getenv("MODEL_MAX_CANDIDATE_PAIRS", "0")))
    p.add_argument("--stage3-num-neighbors", type=int, default=int(os.getenv("MODEL_NUM_NEIGHBORS", "10")))
    p.add_argument("--stage3-featureless-mode", default=os.getenv("MODEL_FEATURELESS_MODE", "type"), choices=["type", "global"])
    p.add_argument("--stage3-loss", default=os.getenv("MODEL_LOSS", "weighted_bce_bpr"), choices=["bce", "weighted_bce", "focal", "bpr", "pairwise_bpr", "weighted_bce_bpr", "bce_bpr"])
    p.add_argument("--stage3-bpr-weight", type=float, default=float(os.getenv("MODEL_BPR_WEIGHT", "0.5")))
    p.add_argument("--stage3-class-weighting", default=os.getenv("MODEL_CLASS_WEIGHTING", "negative_ratio"), choices=["none", "balanced", "negative_ratio"])
    p.add_argument("--stage3-early-stopping-metric", default=os.getenv("MODEL_EARLY_STOPPING_METRIC", "mcc"), choices=["roc_auc", "average_precision", "mcc", "balanced_accuracy", "f1", "specificity", "recall"])
    p.add_argument("--stage3-patience", type=int, default=int(os.getenv("MODEL_PATIENCE", "10")))
    p.add_argument("--stage3-balanced-batches", action="store_true", default=os.getenv("MODEL_BALANCED_BATCHES", "true").lower() == "true")
    p.add_argument("--stage3-balance-ratio", type=float, default=float(os.getenv("MODEL_BALANCE_RATIO", "1.0")))

    # Ensemble
    p.add_argument("--run-ensemble", action="store_true", default=os.getenv("MODEL_RUN_ENSEMBLE", "true").lower() == "true")
    p.add_argument("--ensemble-meta-classifier", default=os.getenv("MODEL_ENSEMBLE_META_CLASSIFIER", "extra_trees"), choices=["extra_trees", "hist_gradient_boosting", "random_forest", "logistic_regression"])
    p.add_argument("--seed", type=int, default=int(os.getenv("MODEL_SEED", "42")))

    # Shared torch
    p.add_argument("--batch-size", type=int, default=int(os.getenv("MODEL_BATCH_SIZE", "4096")))
    p.add_argument("--score-batch-size", type=int, default=int(os.getenv("MODEL_SCORE_BATCH_SIZE", "262144")))
    p.add_argument("--num-workers", type=int, default=int(os.getenv("MODEL_NUM_WORKERS", "0")))
    p.add_argument("--lr", type=float, default=float(os.getenv("MODEL_LR", "0.001")))
    p.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"))

    # Export and explanation
    p.add_argument("--export-neo4j", dest="export_neo4j", action="store_true", default=os.getenv("MODEL_EXPORT_TO_NEO4J", "true").lower() == "true")
    p.add_argument("--no-export-neo4j", dest="export_neo4j", action="store_false")
    p.add_argument("--export-scope", default=os.getenv("MODEL_EXPORT_SCOPE", "best_only"), choices=["none", "best", "best_only", "all"])
    p.add_argument("--max-neo4j-predictions", type=int, default=int(os.getenv("MODEL_MAX_NEO4J_PREDICTIONS", "25000")))
    p.add_argument("--run-stage4", action="store_true", default=os.getenv("MODEL_RUN_STAGE4", "true").lower() == "true")
    p.add_argument("--explain-limit", type=int, default=int(os.getenv("MODEL_EXPLAIN_LIMIT", "100")))
    p.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://neo4j:7687"))
    p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "cyp450kg"))
    p.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    return p


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
