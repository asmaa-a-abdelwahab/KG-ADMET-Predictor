#!/usr/bin/env python3
"""Final reproducible PRING five-CYP MSc thesis pipeline.

This runner intentionally orchestrates the existing PRING-PACKAGE and canonical
PRING-APP modeling modules rather than reimplementing scientific algorithms.
It adds validation, immutable run provenance, stage checkpoints/resume support,
HPC-aware resource controls, final candidate inference, publication figures,
and a strict final thesis quality-control gate.

The default scientific contract is the current canonical PRING final workflow:
  * five human CYP targets;
  * uncapped source collection;
  * registered leakage-safe splits;
  * Stage 1 Extra Trees + Stage 2 KGE baselines + sampled R-GCN/HGT;
  * fixed equal-weight deployable ensemble;
  * Platt calibration and MCC-selected validation threshold;
  * final candidate ranking only after model selection/evaluation.

No password is read from YAML. Neo4j credentials are taken from the configured
environment-variable name.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install PRING-PACKAGE dependencies first.") from exc


PIPELINE_VERSION = "1.0.0"
EXPECTED_FIVE_CYPS = {
    "CYP1A2": "P05177",
    "CYP2C9": "P11712",
    "CYP2C19": "P33261",
    "CYP2D6": "P10635",
    "CYP3A4": "P08684",
}
DEPLOYABLE_SCORE_COLUMNS = [
    "score__stage1_tabular_extra_trees",
    "score__stage3_rgcn_sampled",
    "score__stage3_hgt_sampled",
]
STAGES = [
    "preflight",
    "data_collection",
    "modeling_data",
    "eda",
    "knowledge_graph",
    "features_embeddings",
    "component_models",
    "final_validation",
    "candidate_predictions",
    "thesis_reporting",
    "final_qc",
]


class PipelineError(RuntimeError):
    pass


@dataclasses.dataclass
class StageResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    expected_outputs: list[str]
    warnings: list[str]
    details: dict[str, Any]


@dataclasses.dataclass
class PipelineContext:
    config_path: Path
    cfg: dict[str, Any]
    package_dir: Path
    app_dir: Path
    run_root: Path
    output_root: Path
    raw_run_id: str
    ready_run_id: str
    raw_run_dir: Path
    ready_run_dir: Path
    target_map: dict[str, str]
    python: str
    resume: bool
    force_stages: set[str]
    selected_stages: list[str]
    dry_run: bool
    selected_cyps: list[str]

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "00_manifest"

    @property
    def validation_dir(self) -> Path:
        return self.output_root / "01_data_validation"

    @property
    def eda_dir(self) -> Path:
        return self.output_root / "02_eda"

    @property
    def modeling_snapshot_dir(self) -> Path:
        return self.output_root / "03_modeling_data"

    @property
    def kg_dir(self) -> Path:
        return self.output_root / "04_knowledge_graph"

    @property
    def features_dir(self) -> Path:
        return self.output_root / "05_features_embeddings"

    @property
    def models_dir(self) -> Path:
        return self.output_root / "06_models"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "07_evaluation"

    @property
    def predictions_dir(self) -> Path:
        return self.output_root / "08_predictions"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "09_figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "10_tables"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "11_logs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.manifest_dir / "checkpoints"

    @property
    def state_file(self) -> Path:
        return self.manifest_dir / "run_state.json"

    @property
    def model_input_dir(self) -> Path:
        return self.ready_run_dir / "graph" / "ml" / "modeling"

    @property
    def model_provenance_manifest(self) -> Path:
        return self.model_input_dir / "modeling_stage_manifest.json"


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_file(path: Path, *, nonempty: bool = True) -> Path:
    if not path.is_file():
        raise PipelineError(f"Required file is missing: {path}")
    if nonempty and path.stat().st_size <= 0:
        raise PipelineError(f"Required file is empty: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise PipelineError(f"Required directory is missing: {path}")
    return path


def hash_file(path: Path, chunk_mb: int = 8) -> str:
    digest = hashlib.sha256()
    chunk_size = max(1, int(chunk_mb)) * 1024 * 1024
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_json(payload: Any) -> str:
    raw = json.dumps(safe_json(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def none_arg(value: Any) -> str:
    return "none" if value is None else str(value)


def expand_path(value: Any, *, default: Path | None = None) -> Path:
    """Resolve user/HPC paths while allowing ${ENV_VAR} values in YAML."""
    raw = str(value) if value not in (None, "") else (str(default) if default is not None else "")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if "$" in expanded:
        raise PipelineError(f"Unresolved environment variable in configured path: {raw}")
    return Path(expanded).resolve()


def shell_quote(arg: str) -> str:
    import shlex
    return shlex.quote(str(arg))


def log(ctx: PipelineContext, message: str) -> None:
    text = f"[{utc_now()}] {message}"
    print(text, flush=True)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    with (ctx.logs_dir / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_command(
    ctx: PipelineContext,
    cmd: Sequence[str],
    *,
    stage: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(shell_quote(str(x)) for x in cmd)
    log(ctx, f"[{stage}] COMMAND: {rendered}")
    stage_log = ctx.logs_dir / f"{STAGES.index(stage):02d}_{stage}.log"
    if ctx.dry_run:
        with stage_log.open("a", encoding="utf-8") as handle:
            handle.write("DRY-RUN: " + rendered + "\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    start = time.time()
    if capture:
        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        with stage_log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {rendered}\n{proc.stdout}\n")
    else:
        with stage_log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {rendered}\n")
            handle.flush()
            proc = subprocess.run(
                list(map(str, cmd)),
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    elapsed = time.time() - start
    log(ctx, f"[{stage}] exit={proc.returncode} elapsed={elapsed:.1f}s")
    if check and proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(stage_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        except Exception:
            pass
        raise PipelineError(f"Command failed in stage {stage}: {rendered}\n{tail}")
    return proc


def git_info(path: Path) -> dict[str, Any]:
    def call(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    commit = call("rev-parse", "HEAD")
    branch = call("rev-parse", "--abbrev-ref", "HEAD")
    try:
        status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        dirty_files = [line for line in status.splitlines() if line.strip()]
    except Exception:
        dirty_files = [] if commit != "unknown" else ["git_status_unavailable"]
    return {"commit": commit, "branch": branch, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def installed_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    try:
        from importlib.metadata import version
    except Exception:  # pragma: no cover
        return {name: None for name in names}
    for name in names:
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = None
    return versions


def parse_memory_mb(text: str | None) -> int | None:
    if not text:
        return None
    value = str(text).strip().upper()
    m = re.match(r"^([0-9.]+)\s*([KMGTP]?)(?:B)?$", value)
    if not m:
        return None
    number = float(m.group(1))
    unit = m.group(2)
    multiplier = {"": 1 / (1024 * 1024), "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 * 1024 * 1024}[unit]
    return int(number * multiplier)


def symlink_or_pointer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except Exception:
        # HPC filesystems can disable symlinks. A pointer file remains unambiguous.
        pointer = dst.with_suffix(dst.suffix + ".path.txt")
        pointer.write_text(str(src.resolve()) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Configuration and stage engine
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    require_file(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineError("Configuration root must be a YAML mapping.")
    return payload


def resolve_context(args: argparse.Namespace) -> PipelineContext:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    project = cfg.get("project") or {}
    package_dir = expand_path(project.get("package_dir"))
    app_dir = expand_path(project.get("app_dir"))
    run_root = expand_path(project.get("run_root"), default=package_dir / "runs")
    configured_output = expand_path(project.get("output_root"), default=app_dir / "results" / "final_thesis_run")
    output_root = expand_path(args.output) if args.output else configured_output
    raw_run_id = str(project.get("raw_run_id") or "cyp450_5enzymes_final_raw")
    ready_run_id = str(project.get("ready_run_id") or "cyp450_5enzymes_final_ready")
    targets = dict((cfg.get("cyp450") or {}).get("targets") or {})
    selected_cyps = list(args.cyp or targets.keys())
    unknown = sorted(set(selected_cyps) - set(targets))
    if unknown:
        raise PipelineError(f"Unknown --cyp value(s): {unknown}; configured targets={sorted(targets)}")
    target_map = {name: str(targets[name]) for name in selected_cyps}

    selected_stages = parse_stage_selection(args.stage)
    return PipelineContext(
        config_path=config_path,
        cfg=cfg,
        package_dir=package_dir,
        app_dir=app_dir,
        run_root=run_root,
        output_root=output_root,
        raw_run_id=raw_run_id,
        ready_run_id=ready_run_id,
        raw_run_dir=run_root / raw_run_id,
        ready_run_dir=run_root / ready_run_id,
        target_map=target_map,
        python=sys.executable,
        resume=bool(args.resume),
        force_stages=set(args.force_stage or []),
        selected_stages=selected_stages,
        dry_run=bool(args.dry_run),
        selected_cyps=selected_cyps,
    )


def parse_stage_selection(values: list[str] | None) -> list[str]:
    if not values:
        return list(STAGES)
    chosen: list[str] = []
    for raw in values:
        for item in str(raw).replace(",", " ").split():
            if ":" in item:
                start, end = item.split(":", 1)
                if start not in STAGES or end not in STAGES:
                    raise PipelineError(f"Invalid stage range {item!r}; valid={STAGES}")
                a, b = STAGES.index(start), STAGES.index(end)
                if a > b:
                    a, b = b, a
                chosen.extend(STAGES[a : b + 1])
            else:
                if item not in STAGES:
                    raise PipelineError(f"Invalid stage {item!r}; valid={STAGES}")
                chosen.append(item)
    return [stage for stage in STAGES if stage in set(chosen)]


def init_output_dirs(ctx: PipelineContext) -> None:
    for path in [
        ctx.output_root,
        ctx.manifest_dir,
        ctx.validation_dir,
        ctx.eda_dir,
        ctx.modeling_snapshot_dir,
        ctx.kg_dir,
        ctx.features_dir,
        ctx.models_dir,
        ctx.evaluation_dir,
        ctx.predictions_dir,
        ctx.figures_dir,
        ctx.tables_dir,
        ctx.logs_dir,
        ctx.checkpoint_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def checkpoint_path(ctx: PipelineContext, stage: str) -> Path:
    return ctx.checkpoint_dir / f"{STAGES.index(stage):02d}_{stage}.json"


def execution_contract(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "config_sha256": hash_file(ctx.config_path),
        "runner_sha256": hash_file(Path(__file__).resolve()),
        "package_commit": git_info(ctx.package_dir).get("commit"),
        "app_commit": git_info(ctx.app_dir).get("commit"),
        "targets": ctx.target_map,
    }


def checkpoint_valid(ctx: PipelineContext, stage: str) -> bool:
    path = checkpoint_path(ctx, stage)
    if not path.exists():
        return False
    payload = read_json(path, {}) or {}
    if payload.get("status") != "complete":
        return False
    stored_contract = (payload.get("details") or {}).get("execution_contract")
    if stored_contract != execution_contract(ctx):
        return False
    for raw in payload.get("expected_outputs") or []:
        p = Path(raw)
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            return False
    return True


def stage_wrapper(
    ctx: PipelineContext,
    stage: str,
    fn: Callable[[PipelineContext], tuple[list[Path], dict[str, Any], list[str]]],
) -> None:
    if stage not in ctx.selected_stages:
        return
    if stage in ctx.force_stages:
        checkpoint_path(ctx, stage).unlink(missing_ok=True)
    elif ctx.resume and checkpoint_valid(ctx, stage):
        log(ctx, f"[{stage}] resume: validated checkpoint found; skipping.")
        return

    started = utc_now()
    t0 = time.time()
    log(ctx, f"========== START {stage} ==========")
    try:
        outputs, details, warnings = fn(ctx)
        details = dict(details or {})
        details["execution_contract"] = execution_contract(ctx)
        if not ctx.dry_run:
            for output in outputs:
                require_file(output) if output.suffix else require_dir(output)
        result = StageResult(
            name=stage,
            status="complete",
            started_at=started,
            finished_at=utc_now(),
            elapsed_seconds=time.time() - t0,
            expected_outputs=[str(p) for p in outputs],
            warnings=warnings,
            details=details,
        )
        write_json(checkpoint_path(ctx, stage), dataclasses.asdict(result))
        log(ctx, f"========== COMPLETE {stage} ({result.elapsed_seconds:.1f}s) ==========")
    except Exception as exc:
        payload = {
            "name": stage,
            "status": "failed",
            "started_at": started,
            "failed_at": utc_now(),
            "elapsed_seconds": time.time() - t0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(checkpoint_path(ctx, stage), payload)
        log(ctx, f"========== FAILED {stage}: {type(exc).__name__}: {exc} ==========")
        raise


# ---------------------------------------------------------------------------
# Preflight and provenance
# ---------------------------------------------------------------------------


def target_file(ctx: PipelineContext) -> Path:
    return ctx.manifest_dir / "cyp450_targets.txt"


def write_target_file(ctx: PipelineContext) -> Path:
    path = target_file(ctx)
    path.write_text("\n".join(ctx.target_map.values()) + "\n", encoding="utf-8")
    return path


def validate_scientific_config(ctx: PipelineContext) -> list[str]:
    warnings: list[str] = []
    scientific = ctx.cfg.get("scientific") or {}
    collection = ctx.cfg.get("collection") or {}
    project = ctx.cfg.get("project") or {}

    if len(ctx.target_map) == 5:
        if ctx.target_map != EXPECTED_FIVE_CYPS:
            raise PipelineError(
                "Final five-CYP configuration does not exactly match the canonical target accession map: "
                f"expected={EXPECTED_FIVE_CYPS}, configured={ctx.target_map}"
            )
    else:
        warnings.append(
            "A CYP subset was requested. The run is useful for smoke/debug work but cannot pass the final thesis five-CYP gate."
        )
    if scientific.get("candidate_pair_mode") != "all":
        raise PipelineError("Final thesis run requires scientific.candidate_pair_mode=all.")
    if str(scientific.get("split_strategy", "registered")) != "registered":
        raise PipelineError("Final thesis run requires the registered split registry; refusing diagnostic re-splitting.")
    if str(scientific.get("final_combiner", "fixed_mean")) != "fixed_mean":
        warnings.append(
            "A non-default final combiner was requested. This is a methodological change from the current deployable fixed-mean PRING contract."
        )
    for key in (
        "max_compounds_per_target",
        "max_targets_per_compound",
        "max_substances_per_compound",
        "max_measuregroups_per_target",
        "max_measuregroups_per_compound",
        "max_endpoints_per_pair",
        "max_similar_compounds_per_compound",
        "max_textmine_records",
        "max_textmine_records_per_target",
        "max_textmine_references_per_pair",
        "max_enrichment_records_per_entity",
        "max_candidate_missing_pairs",
    ):
        if collection.get(key) is not None:
            raise PipelineError(f"Final thesis run must be uncapped; collection.{key} must be null.")
    if not as_bool(project.get("require_clean_git"), True):
        warnings.append("project.require_clean_git=false weakens final run provenance and will block final-thesis status.")
    return warnings


def collect_runtime_manifest(ctx: PipelineContext) -> dict[str, Any]:
    resources = ctx.cfg.get("resources") or {}
    package_git = git_info(ctx.package_dir)
    app_git = git_info(ctx.app_dir)
    pring_version = None
    modeling_version = None
    try:
        mod = importlib.import_module("pring")
        pring_version = getattr(mod, "__version__", None)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        modeling_version = version("pring-app-modeling")
    except Exception:
        pass
    gpu: dict[str, Any] = {}
    try:
        import torch
        gpu = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        gpu = {"available": False, "error": str(exc)}
    allocation = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION") or os.getenv("SLURM_PARTITION"),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_cpu": os.getenv("SLURM_MEM_PER_CPU"),
        "slurm_gpus": os.getenv("SLURM_GPUS") or os.getenv("SLURM_GPUS_ON_NODE"),
        "configured_cpus": resources.get("cpus"),
        "configured_memory_mb": resources.get("memory_mb"),
        "configured_device": resources.get("model_device"),
    }
    return {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": utc_now(),
        "config_path": str(ctx.config_path),
        "config_sha256": hash_file(ctx.config_path),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pring_version": pring_version,
        "pring_app_modeling_version": modeling_version,
        "package_git": package_git,
        "app_git": app_git,
        "allocation": allocation,
        "gpu": gpu,
        "environment_thread_controls": {
            key: os.getenv(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TORCH_NUM_THREADS",
                "TORCH_NUM_INTEROP_THREADS",
            )
        },
        "targets": ctx.target_map,
        "scientific_configuration": ctx.cfg.get("scientific") or {},
        "package_versions": installed_versions(
            [
                "pring",
                "pring-app-modeling",
                "pandas",
                "numpy",
                "scikit-learn",
                "neo4j",
                "matplotlib",
                "torch",
                "torch-geometric",
                "pyarrow",
                "PyYAML",
            ]
        ),
    }


def require_clean_git(ctx: PipelineContext, manifest: dict[str, Any]) -> None:
    if not as_bool((ctx.cfg.get("project") or {}).get("require_clean_git"), True):
        return
    for label in ("package_git", "app_git"):
        info = manifest[label]
        if info.get("commit") == "unknown":
            raise PipelineError(f"Final run requires a resolvable Git commit for {label}.")
        if info.get("dirty"):
            raise PipelineError(f"Final run requires a clean Git checkout for {label}: {info.get('dirty_files')}")


def validate_runtime_resources(ctx: PipelineContext, manifest: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    resources = ctx.cfg.get("resources") or {}
    expected_cpus = int(resources.get("cpus") or 1)
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus and int(slurm_cpus) < expected_cpus:
        raise PipelineError(
            f"Slurm allocated {slurm_cpus} CPU(s), below config resources.cpus={expected_cpus}."
        )
    try:
        import psutil
        available_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        manifest["host_memory_total_mb"] = available_mb
        requested = int(resources.get("memory_mb") or 0)
        if requested and available_mb < requested * 0.90:
            warnings.append(
                f"Visible host memory ({available_mb} MB) is below configured memory ({requested} MB); cgroup accounting may explain this."
            )
    except Exception as exc:
        warnings.append(f"Could not inspect host memory with psutil: {exc}")
    if str(resources.get("model_device", "cuda")) == "cuda":
        gpu = manifest.get("gpu") or {}
        if not gpu.get("cuda_available"):
            raise PipelineError("resources.model_device=cuda but PyTorch reports no CUDA device.")
    return warnings


def cli_help(ctx: PipelineContext, module: str, stage: str = "preflight") -> str:
    proc = run_command(ctx, [ctx.python, "-m", module, "--help"], stage=stage, capture=True)
    return proc.stdout or ""


def validate_required_runtime_imports(ctx: PipelineContext) -> dict[str, str]:
    """Fail before expensive stages if a runtime dependency needed by the final run is unavailable."""
    required = {
        "pring": "PRING-PACKAGE",
        "pring_modeling": "PRING-APP modeling package",
        "pandas": "pandas",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "neo4j": "Neo4j Python driver",
        "matplotlib": "Matplotlib",
        "torch": "PyTorch",
        "torch_geometric": "PyTorch Geometric",
        "pyarrow": "PyArrow (Parquet prediction export)",
        "joblib": "joblib",
        "scipy": "SciPy",
    }
    loaded: dict[str, str] = {}
    failures: list[str] = []
    for module_name, label in required.items():
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:
            failures.append(f"{label} ({module_name}): {type(exc).__name__}: {exc}")
    if failures:
        raise PipelineError("Missing/incompatible final-run dependencies:\n  " + "\n  ".join(failures))
    return loaded


def validate_cli_contract(ctx: PipelineContext) -> dict[str, Any]:
    if ctx.dry_run:
        return {"status": "dry_run", "note": "CLI contract checks execute during a real preflight."}
    checks: dict[str, Any] = {}
    pring_help = cli_help(ctx, "pring")
    for token in ("build", "load-run", "eda"):
        checks[f"pring_{token}"] = token in pring_help
    fv_help = cli_help(ctx, "pring_modeling.final_validation")
    for token in ("registered", "fixed_mean", "--provenance-manifest", "--strict-leakage-free"):
        checks[f"final_validation_{token}"] = token in fv_help
    s1_help = cli_help(ctx, "pring_modeling.stage1_tabular")
    checks["stage1_candidate_scope"] = "--prediction-scope" in s1_help and "candidates" in s1_help
    for model in ("stage3_rgcn", "stage3_hgt"):
        help_text = cli_help(ctx, f"pring_modeling.{model}")
        checks[f"{model}_score_candidates"] = "--score-candidates" in help_text
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise PipelineError(f"Canonical PRING CLI contract mismatch: {failed}")
    return checks


def smoke_demo(ctx: PipelineContext) -> Path:
    smoke_root = ctx.manifest_dir / "smoke"
    smoke_id = "preflight_demo"
    smoke_dir = smoke_root / smoke_id
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    run_command(
        ctx,
        [
            ctx.python,
            "-m",
            "pring",
            "demo",
            "--load-neo4j",
            "false",
            "--out-dir",
            str(smoke_root),
            "--run-id",
            smoke_id,
        ],
        stage="preflight",
        cwd=ctx.package_dir,
    )
    return smoke_dir / "manifest.json"


def neo4j_connectivity(ctx: PipelineContext, *, require_empty: bool | None = None) -> dict[str, Any]:
    if ctx.dry_run:
        return {"status": "dry_run"}
    neo = ctx.cfg.get("neo4j") or {}
    password_name = str(neo.get("password_env") or "NEO4J_PASSWORD")
    password = os.getenv(password_name)
    if not password:
        raise PipelineError(f"Neo4j password environment variable is missing: {password_name}")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(str(neo.get("uri")), auth=(str(neo.get("user")), password))
    try:
        driver.verify_connectivity()
        with driver.session(database=str(neo.get("database") or "neo4j")) as session:
            node_count = int(session.run("MATCH (n) RETURN count(n) AS n").single()["n"])
            gds_version = session.run("RETURN gds.version() AS version").single()["version"]
        should_be_empty = as_bool(neo.get("require_empty_database"), True) if require_empty is None else require_empty
        if should_be_empty and node_count:
            raise PipelineError(
                f"Neo4j contains {node_count} node(s). A stale graph can contaminate GDS features; use a dedicated empty database."
            )
        return {"status": "ok", "node_count": node_count, "gds_version": str(gds_version)}
    finally:
        driver.close()


def stage_preflight(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    require_dir(ctx.package_dir)
    require_dir(ctx.app_dir)
    require_file(ctx.package_dir / "pyproject.toml")
    require_file(ctx.app_dir / "modeling" / "pyproject.toml")
    ctx.run_root.mkdir(parents=True, exist_ok=True)
    warnings = validate_scientific_config(ctx)
    tfile = write_target_file(ctx)
    manifest = collect_runtime_manifest(ctx)
    require_clean_git(ctx, manifest)
    dependency_imports = {"status": "dry_run"} if ctx.dry_run else validate_required_runtime_imports(ctx)
    manifest["validated_runtime_imports"] = dependency_imports
    warnings.extend(validate_runtime_resources(ctx, manifest))

    smoke_cfg = ctx.cfg.get("smoke_tests") or {}
    contract = {}
    if as_bool(smoke_cfg.get("run_cli_contract_checks"), True):
        contract = validate_cli_contract(ctx)
    demo_manifest = None
    if as_bool(smoke_cfg.get("run_pring_demo"), True):
        demo_manifest = smoke_demo(ctx)
        if not ctx.dry_run:
            require_file(demo_manifest)
    neo = neo4j_connectivity(ctx, require_empty=as_bool((ctx.cfg.get("neo4j") or {}).get("require_empty_database"), True))

    manifest["cli_contract"] = contract
    manifest["neo4j_preflight"] = neo
    manifest["warnings"] = warnings
    manifest["final_thesis_scope"] = len(ctx.target_map) == 5 and ctx.target_map == EXPECTED_FIVE_CYPS
    runtime_manifest_path = ctx.manifest_dir / "run_manifest_initial.json"
    write_json(runtime_manifest_path, manifest)
    write_json(ctx.state_file, {
        "raw_run_dir": str(ctx.raw_run_dir),
        "ready_run_dir": str(ctx.ready_run_dir),
        "target_file": str(tfile),
        "created_at": utc_now(),
    })
    if as_bool((ctx.cfg.get("provenance") or {}).get("record_pip_freeze"), True) and not ctx.dry_run:
        proc = subprocess.run([ctx.python, "-m", "pip", "freeze"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (ctx.manifest_dir / "pip_freeze.txt").write_text(proc.stdout, encoding="utf-8")

    outputs = [runtime_manifest_path, tfile]
    if demo_manifest is not None:
        outputs.append(demo_manifest)
    return outputs, {"cli_contract": contract, "neo4j": neo}, warnings


# ---------------------------------------------------------------------------
# Source collection, quality validation, and modeling materialization
# ---------------------------------------------------------------------------


def assert_source_quality(report_path: Path) -> dict[str, Any]:
    quality = read_json(require_file(report_path), {}) or {}
    cap_status = (quality.get("cap_completeness_report") or {}).get("data_completeness_status")
    readiness = quality.get("cyp450_gcn_readiness_report") or {}
    candidate_mode = (readiness.get("ml_pair_summary") or {}).get("candidate_missing_pair_mode")
    blockers = readiness.get("blockers") or []
    problems: list[str] = []
    if cap_status != "uncapped_or_no_internal_caps_detected":
        problems.append(f"data completeness={cap_status!r}")
    if readiness.get("pipeline_validation_ready") is not True:
        problems.append("pipeline_validation_ready is not true")
    if blockers:
        problems.append(f"readiness blockers={blockers}")
    if str(candidate_mode).lower() != "all":
        problems.append(f"candidate pair mode={candidate_mode!r}, expected 'all'")
    similarity = quality.get("similarity_report") or {}
    dangling = similarity.get("dangling_similarity_edges")
    if dangling not in (None, 0):
        problems.append(f"dangling similarity edges={dangling}")
    if problems:
        raise PipelineError("Source run quality gate failed: " + "; ".join(problems))
    return quality


def build_collection_command(ctx: PipelineContext) -> list[str]:
    scientific = ctx.cfg.get("scientific") or {}
    collection = ctx.cfg.get("collection") or {}
    resources = ctx.cfg.get("resources") or {}
    plugins = [str(x) for x in scientific.get("plugins") or []]
    cmd = [
        ctx.python,
        "-m",
        "pring",
        "build",
        "--mode",
        str(collection.get("mode") or "sparql"),
        "--scope",
        "expand-from-targets",
        "--target-ids",
        str(target_file(ctx)),
        "--taxid",
        str(scientific.get("taxid") or 9606),
        "--case-study-mode",
        "final-cyp450",
        "--resource-profile",
        "high",
        "--max-workers",
        str(int(resources.get("cpus") or 1)),
        "--max-memory-mb",
        str(int(resources.get("memory_mb") or 0)),
        "--memory-safety-margin-mb",
        str(int(resources.get("memory_safety_margin_mb") or 8192)),
        "--reserve-system-memory-mb",
        str(int(resources.get("reserve_system_memory_mb") or 8192)),
        "--sparql-page-size",
        str(int(collection.get("sparql_page_size") or 10)),
        "--sparql-timeout-s",
        str(int(collection.get("sparql_timeout_s") or 300)),
        "--sparql-evidence-timeout-s",
        str(int(collection.get("sparql_evidence_timeout_s") or 300)),
        "--sparql-max-retries",
        str(int(collection.get("sparql_max_retries") or 5)),
        "--sparql-evidence-max-retries",
        str(int(collection.get("sparql_evidence_max_retries") or 2)),
        "--sparql-adaptive-chunking",
        "true",
        "--sparql-min-page-size",
        "1",
        "--sparql-skip-failed-chunks",
        "false",
        "--max-compounds-per-target",
        none_arg(collection.get("max_compounds_per_target")),
        "--max-targets-per-compound",
        none_arg(collection.get("max_targets_per_compound")),
        "--max-substances-per-compound",
        none_arg(collection.get("max_substances_per_compound")),
        "--max-measuregroups-per-target",
        none_arg(collection.get("max_measuregroups_per_target")),
        "--max-measuregroups-per-compound",
        none_arg(collection.get("max_measuregroups_per_compound")),
        "--max-endpoints-per-pair",
        none_arg(collection.get("max_endpoints_per_pair")),
        "--max-similar-compounds-per-compound",
        none_arg(collection.get("max_similar_compounds_per_compound")),
        "--max-textmine-records",
        none_arg(collection.get("max_textmine_records")),
        "--max-textmine-records-per-target",
        none_arg(collection.get("max_textmine_records_per_target")),
        "--max-textmine-references-per-pair",
        none_arg(collection.get("max_textmine_references_per_pair")),
        "--max-enrichment-records-per-entity",
        none_arg(collection.get("max_enrichment_records_per_entity")),
        "--activity-threshold-um",
        str(scientific.get("activity_threshold_um") or 10.0),
        "--weak-activity-as-negative",
        str(as_bool(scientific.get("weak_activity_as_negative"), False)).lower(),
        "--candidate-pair-mode",
        "all",
        "--max-candidate-missing-pairs",
        none_arg(collection.get("max_candidate_missing_pairs")),
        "--include-compound-similarity",
        str(as_bool(scientific.get("include_compound_similarity"), True)).lower(),
        "--compound-similarity-threshold",
        str(scientific.get("compound_similarity_threshold") or 90),
        "--include-textmining",
        str(as_bool(scientific.get("include_textmining"), False)).lower(),
        "--include-optional-context",
        "true",
        "--include-endpoint-metadata",
        "true",
        "--include-endpoint-references",
        str(as_bool(scientific.get("include_endpoint_references"), False)).lower(),
        "--write-csv-mirrors",
        "true",
        "--save-raw",
        str(as_bool(collection.get("save_raw"), True)).lower(),
        "--save-extracted",
        str(as_bool(collection.get("save_extracted"), True)).lower(),
        "--cache-dir",
        str(expand_path((ctx.cfg.get("project") or {}).get("cache_dir"), default=ctx.package_dir / "data" / "cache")),
        "--load-neo4j",
        "false",
        "--overwrite-run",
        "false",
        "--out-dir",
        str(ctx.run_root),
        "--run-id",
        ctx.raw_run_id,
    ]
    if plugins:
        cmd += ["--plugins", *plugins]
    return cmd


def file_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            records.append({
                "relative_path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    return records


def hash_source_inputs(ctx: PipelineContext, source_run: Path) -> Path:
    prov = ctx.cfg.get("provenance") or {}
    output = ctx.manifest_dir / "source_input_checksums.csv"
    if not as_bool(prov.get("hash_source_inputs"), True):
        output.write_text("relative_path,size_bytes,sha256\n", encoding="utf-8")
        return output
    chunk_mb = int(prov.get("hash_chunk_mb") or 8)
    preferred_roots = [source_run / "raw", source_run / "extracted"]
    paths: list[Path] = []
    for root in preferred_roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    if not paths:
        # Reused historical runs may not keep raw downloads. Hash the immutable
        # manifest and canonical graph artifacts instead.
        for rel in ("manifest.json", "graph/run_quality_report.json"):
            p = source_run / rel
            if p.exists():
                paths.append(p)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        for index, path in enumerate(sorted(set(paths))):
            digest = hash_file(path, chunk_mb=chunk_mb)
            writer.writerow({"relative_path": str(path.relative_to(source_run)), "size_bytes": path.stat().st_size, "sha256": digest})
            if index and index % 100 == 0:
                log(ctx, f"[data_collection] checksummed {index}/{len(paths)} source input files")
    return output


def stage_data_collection(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    project = ctx.cfg.get("project") or {}
    source_value = project.get("source_run_dir")
    warnings: list[str] = []
    if source_value:
        source_run = expand_path(source_value)
        require_dir(source_run)
        log(ctx, f"[data_collection] reusing immutable source run: {source_run}")
    else:
        source_run = ctx.raw_run_dir
        if source_run.exists() and not ctx.resume and "data_collection" not in ctx.force_stages:
            raise PipelineError(f"Raw run already exists: {source_run}. Use --resume or a new run_id.")
        if not source_run.exists() or "data_collection" in ctx.force_stages:
            if source_run.exists() and "data_collection" in ctx.force_stages:
                raise PipelineError(
                    "Refusing to delete/overwrite an existing raw scientific run. Change project.raw_run_id for a forced re-collection."
                )
            run_command(ctx, build_collection_command(ctx), stage="data_collection", cwd=ctx.package_dir)
    quality_path = source_run / "graph" / "run_quality_report.json"
    manifest_path = source_run / "manifest.json"
    if not ctx.dry_run:
        require_file(manifest_path)
        quality = assert_source_quality(quality_path)
    else:
        quality = {}
    checksum_path = hash_source_inputs(ctx, source_run) if not ctx.dry_run else ctx.manifest_dir / "source_input_checksums.csv"
    source_summary = {
        "source_run_dir": str(source_run),
        "source_manifest": str(manifest_path),
        "source_quality_report": str(quality_path),
        "source_manifest_sha256": hash_file(manifest_path) if manifest_path.exists() else None,
        "source_quality_sha256": hash_file(quality_path) if quality_path.exists() else None,
        "quality_gate": {
            "cap_status": (quality.get("cap_completeness_report") or {}).get("data_completeness_status"),
            "pipeline_validation_ready": (quality.get("cyp450_gcn_readiness_report") or {}).get("pipeline_validation_ready"),
            "candidate_mode": ((quality.get("cyp450_gcn_readiness_report") or {}).get("ml_pair_summary") or {}).get("candidate_missing_pair_mode"),
        },
    }
    write_json(ctx.validation_dir / "source_run_validation.json", source_summary)
    state = read_json(ctx.state_file, {}) or {}
    state["source_run_dir"] = str(source_run)
    write_json(ctx.state_file, state)
    return [manifest_path, quality_path, checksum_path, ctx.validation_dir / "source_run_validation.json"], source_summary, warnings


def source_run_dir(ctx: PipelineContext) -> Path:
    state = read_json(ctx.state_file, {}) or {}
    value = state.get("source_run_dir") or (ctx.cfg.get("project") or {}).get("source_run_dir") or ctx.raw_run_dir
    return expand_path(value)


def stage_modeling_data(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    source_run = source_run_dir(ctx)
    require_dir(source_run)
    if ctx.ready_run_dir.exists() and not ctx.resume:
        raise PipelineError(f"Modeling-ready run already exists: {ctx.ready_run_dir}; use --resume or a new ready_run_id.")
    if not ctx.ready_run_dir.exists():
        scientific = ctx.cfg.get("scientific") or {}
        collection = ctx.cfg.get("collection") or {}
        cmd = [
            ctx.python,
            "-m",
            "pring",
            "load-run",
            "--run-dir",
            str(source_run),
            "--out-dir",
            str(ctx.run_root),
            "--run-id",
            ctx.ready_run_id,
            "--schema-dot",
            str(ctx.package_dir / "schema" / "pring-implementation-ready-schema.dot"),
            "--rematerialize-schema",
            "true",
            "--rematerialize-csv",
            "true",
            "--ensure-neo4j-schema",
            "false",
            "--validate-dot-schema",
            "true",
            "--complete-similar-compound-nodes",
            "false",
            "--allow-network",
            "false",
            "--load-neo4j",
            "false",
            "--case-study-mode",
            "final-cyp450",
            "--activity-threshold-um",
            str(scientific.get("activity_threshold_um") or 10.0),
            "--weak-activity-as-negative",
            str(as_bool(scientific.get("weak_activity_as_negative"), False)).lower(),
            "--candidate-pair-mode",
            "all",
            "--max-candidate-missing-pairs",
            none_arg(collection.get("max_candidate_missing_pairs")),
            "--write-csv-mirrors",
            "true",
        ]
        run_command(ctx, cmd, stage="modeling_data", cwd=ctx.package_dir)
    outputs = [
        ctx.ready_run_dir / "manifest.json",
        ctx.ready_run_dir / "graph" / "run_quality_report.json",
        ctx.model_provenance_manifest,
        ctx.model_input_dir / "stage3_heterogeneous_gnn" / "edge_index_train_only.csv",
    ]
    if not ctx.dry_run:
        for p in outputs:
            require_file(p)
        assert_source_quality(ctx.ready_run_dir / "graph" / "run_quality_report.json")
    manifest = read_json(ctx.model_provenance_manifest, {}) or {}
    required = ["dataset_id", "split_registry_id", "feature_schema_id", "label_policy_id"]
    missing = [key for key in required if not manifest.get(key)]
    if missing and not ctx.dry_run:
        raise PipelineError(f"Modeling provenance manifest is missing: {missing}")
    if manifest.get("graph_scope") != "cold_compound_inductive_train_only" and not ctx.dry_run:
        raise PipelineError(f"Unexpected leakage-control graph_scope={manifest.get('graph_scope')!r}")
    symlink_or_pointer(ctx.model_input_dir, ctx.modeling_snapshot_dir / "modeling_exports")
    details = {key: manifest.get(key) for key in required + ["graph_scope", "prediction_contamination_control"]}
    write_json(ctx.validation_dir / "modeling_data_validation.json", details)
    return outputs + [ctx.validation_dir / "modeling_data_validation.json"], details, []


# ---------------------------------------------------------------------------
# EDA, graph load, graph statistics, and leakage-safe GDS features
# ---------------------------------------------------------------------------


def run_artifact_validator(ctx: PipelineContext, phase: str, extra: list[str] | None = None) -> tuple[Path, Path]:
    validator = ctx.app_dir / "examples" / "hpc" / "validate_pipeline_artifacts.py"
    json_out = ctx.validation_dir / f"{phase}_readiness.json"
    md_out = ctx.validation_dir / f"{phase}_readiness.md"
    cmd = [
        ctx.python,
        str(validator),
        "--run-dir",
        str(ctx.ready_run_dir),
        "--phase",
        phase,
        "--output-json",
        str(json_out),
        "--output-markdown",
        str(md_out),
        "--strict",
    ]
    if extra:
        cmd.extend(extra)
    run_command(ctx, cmd, stage="eda" if phase == "prepared" else "final_qc", cwd=ctx.app_dir)
    return json_out, md_out


def stage_eda(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    eda_cfg = ctx.cfg.get("eda") or {}
    run_command(
        ctx,
        [
            ctx.python,
            "-m",
            "pring",
            "eda",
            "--run-path",
            str(ctx.ready_run_dir),
            "--output-dir",
            str(ctx.eda_dir),
            "--top-n",
            str(int(eda_cfg.get("top_n") or 30)),
        ],
        stage="eda",
        cwd=ctx.package_dir,
    )
    outputs = [ctx.eda_dir / "eda_report.html", ctx.eda_dir / "eda_report.md", ctx.eda_dir / "eda_summary.json"]
    if not ctx.dry_run:
        for p in outputs:
            require_file(p)
    ready_json, ready_md = run_artifact_validator(
        ctx,
        "prepared",
        extra=["--eda-dir", str(ctx.eda_dir)],
    )
    return outputs + [ready_json, ready_md], {"eda_dir": str(ctx.eda_dir)}, []


def load_graph_to_neo4j(ctx: PipelineContext) -> None:
    neo4j_connectivity(ctx, require_empty=as_bool((ctx.cfg.get("neo4j") or {}).get("require_empty_database"), True))
    resources = ctx.cfg.get("resources") or {}
    cmd = [
        ctx.python,
        "-m",
        "pring",
        "load-run",
        "--run-dir",
        str(ctx.ready_run_dir),
        "--schema-dot",
        str(ctx.package_dir / "schema" / "pring-implementation-ready-schema.dot"),
        "--rematerialize-schema",
        "false",
        "--rematerialize-csv",
        "false",
        "--ensure-neo4j-schema",
        "true",
        "--validate-dot-schema",
        "true",
        "--complete-similar-compound-nodes",
        "false",
        "--allow-network",
        "false",
        "--load-neo4j",
        "true",
        "--batch-size",
        str(int(resources.get("neo4j_load_batch_size") or 1000)),
    ]
    run_command(ctx, cmd, stage="knowledge_graph", cwd=ctx.package_dir)


def neo4j_graph_statistics(ctx: PipelineContext) -> dict[str, Any]:
    if ctx.dry_run:
        return {"status": "dry_run"}
    neo = ctx.cfg.get("neo4j") or {}
    password_name = str(neo.get("password_env") or "NEO4J_PASSWORD")
    password = os.getenv(password_name)
    if not password:
        raise PipelineError(f"Missing Neo4j password env: {password_name}")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(str(neo.get("uri")), auth=(str(neo.get("user")), password))
    db = str(neo.get("database") or "neo4j")
    result: dict[str, Any] = {"queried_at_utc": utc_now()}
    try:
        with driver.session(database=db) as session:
            result["total_nodes"] = int(session.run("MATCH (n) RETURN count(n) AS n").single()["n"])
            result["total_relationships"] = int(session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"])
            node_counts = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC"
            )
            result["node_counts_by_label"] = {str(r["label"]): int(r["n"]) for r in node_counts}
            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS n ORDER BY n DESC"
            )
            result["relationship_counts_by_type"] = {str(r["rel_type"]): int(r["n"]) for r in rel_counts}
            cyp_counts: dict[str, Any] = {}
            for name, accession in ctx.target_map.items():
                record = session.run(
                    """
                    MATCH (p:Protein)
                    WHERE toString(coalesce(p.protein_id,p.uniprot_id,p.accession,p.id,''))=$acc
                    OPTIONAL MATCH (i:Interaction)-[:ASSERTS_TARGET]->(p)
                    RETURN count(DISTINCT p) AS proteins, count(DISTINCT i) AS interactions
                    """,
                    acc=accession,
                ).single()
                cyp_counts[name] = {
                    "accession": accession,
                    "protein_nodes": int(record["proteins"]),
                    "interaction_nodes": int(record["interactions"]),
                }
            result["cyp450"] = cyp_counts

            # Degree summaries are computed with one graph scan and percentileCont.
            degree = session.run(
                """
                MATCH (n)
                WITH n, count { (n)--() } AS degree
                RETURN count(n) AS nodes,
                       avg(toFloat(degree)) AS mean,
                       percentileCont(toFloat(degree),0.5) AS median,
                       percentileCont(toFloat(degree),0.95) AS p95,
                       max(degree) AS max,
                       sum(CASE WHEN degree=0 THEN 1 ELSE 0 END) AS isolated
                """
            ).single()
            result["degree_summary"] = {
                "nodes": int(degree["nodes"] or 0),
                "mean": float(degree["mean"] or 0.0),
                "median": float(degree["median"] or 0.0),
                "p95": float(degree["p95"] or 0.0),
                "max": int(degree["max"] or 0),
                "isolated_nodes": int(degree["isolated"] or 0),
            }
            n = result["total_nodes"]
            m = result["total_relationships"]
            result["directed_density"] = float(m / (n * (n - 1))) if n > 1 else None

            if as_bool(neo.get("compute_wcc"), True):
                graph_name = str(neo.get("wcc_graph_name") or "pring_final_thesis_wcc")
                # Drop a stale projection with the same name only; this never mutates the stored KG.
                existing = session.run("CALL gds.graph.exists($name) YIELD exists RETURN exists", name=graph_name).single()["exists"]
                if existing:
                    session.run("CALL gds.graph.drop($name) YIELD graphName", name=graph_name).consume()
                projection = session.run(
                    "CALL gds.graph.project($name, '*', '*') "
                    "YIELD graphName, nodeCount, relationshipCount "
                    "RETURN graphName, nodeCount, relationshipCount",
                    name=graph_name,
                ).single()
                result["wcc_projection"] = {
                    "graph_name": str(projection["graphName"]),
                    "node_count": int(projection["nodeCount"]),
                    "relationship_count": int(projection["relationshipCount"]),
                    "note": "WCC is weak connectivity, so stored relationship direction does not separate components.",
                }
                wcc = session.run(
                    """
                    CALL gds.wcc.stats($name)
                    YIELD componentCount, componentDistribution
                    RETURN componentCount, componentDistribution
                    """,
                    name=graph_name,
                ).single()
                result["connected_components"] = {
                    "component_count": int(wcc["componentCount"]),
                    "component_distribution": dict(wcc["componentDistribution"] or {}),
                }
                session.run("CALL gds.graph.drop($name) YIELD graphName", name=graph_name).consume()
    finally:
        driver.close()
    return result


def write_dict_csv(path: Path, key_name: str, value_name: str, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, value_name])
        for key, value in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([key, value])


def stage_knowledge_graph(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    load_graph_to_neo4j(ctx)
    stats = neo4j_graph_statistics(ctx)
    stats_path = ctx.kg_dir / "neo4j_graph_statistics.json"
    write_json(stats_path, stats)
    node_csv = ctx.tables_dir / "knowledge_graph_node_counts.csv"
    rel_csv = ctx.tables_dir / "knowledge_graph_relationship_counts.csv"
    if not ctx.dry_run:
        write_dict_csv(node_csv, "node_label", "count", stats.get("node_counts_by_label") or {})
        write_dict_csv(rel_csv, "relationship_type", "count", stats.get("relationship_counts_by_type") or {})
        if int(stats.get("total_nodes") or 0) == 0 or int(stats.get("total_relationships") or 0) == 0:
            raise PipelineError("Loaded Neo4j graph is empty.")
        missing_cyps = [name for name, x in (stats.get("cyp450") or {}).items() if int(x.get("protein_nodes") or 0) != 1]
        if missing_cyps:
            raise PipelineError(f"Expected exactly one Protein node for each CYP; failed={missing_cyps}")
    return [stats_path, node_csv, rel_csv], stats, []


def stage_features_embeddings(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    resources = ctx.cfg.get("resources") or {}
    scientific = ctx.cfg.get("scientific") or {}
    stage1_dir = ctx.model_input_dir / "stage1_neo4j_gds_baselines"
    pair_file = stage1_dir / "compound_target_training_pairs_for_gds.csv"
    require_file(pair_file) if not ctx.dry_run else None
    gds_summary = stage1_dir / "stage1_outcome_safe_gds_summary.json"
    run_command(
        ctx,
        [
            ctx.python,
            str(ctx.app_dir / "examples" / "hpc" / "prepare_leakage_safe_gds.py"),
            "--pair-file",
            str(pair_file),
            "--provenance-manifest",
            str(ctx.model_provenance_manifest),
            "--output-json",
            str(gds_summary),
            "--graph-name",
            "pring_stage1_outcome_safe",
            "--write-property",
            "pringFastRP",
            "--embedding-dimension",
            "128",
            "--seed",
            str(int(scientific.get("master_seed") or 42)),
            "--batch-size",
            str(int(resources.get("stage1_neo4j_batch_size") or 5000)),
        ],
        stage="features_embeddings",
        cwd=ctx.app_dir,
    )
    run_command(
        ctx,
        [
            ctx.python,
            "-m",
            "pring_modeling.stage1_export_gds_features",
            "--modeling-dir",
            str(ctx.model_input_dir),
            "--include-candidates",
            "--max-training-rows",
            "0",
            "--max-candidate-rows",
            "0",
            "--chunk-size",
            str(int(resources.get("stage1_export_chunk_size") or 50000)),
            "--neo4j-batch-size",
            str(int(resources.get("stage1_neo4j_batch_size") or 5000)),
        ],
        stage="features_embeddings",
        cwd=ctx.app_dir / "modeling",
    )
    outputs = [
        gds_summary,
        stage1_dir / "compound_target_training_pairs_gds_features.csv",
        stage1_dir / "candidate_pairs_gds_features.csv",
        stage1_dir / "stage1_gds_feature_export_summary.json",
    ]
    if not ctx.dry_run:
        for p in outputs:
            require_file(p)
    symlink_or_pointer(stage1_dir, ctx.features_dir / "stage1_gds_exports")
    details = read_json(gds_summary, {}) or {}
    return outputs, details, []


# ---------------------------------------------------------------------------
# Component modeling and final validation
# ---------------------------------------------------------------------------


def model_env(ctx: PipelineContext) -> dict[str, str]:
    resources = ctx.cfg.get("resources") or {}
    return {
        "PYTHONPATH": f"{ctx.package_dir}:{ctx.app_dir / 'modeling'}" + (f":{os.environ['PYTHONPATH']}" if os.getenv("PYTHONPATH") else ""),
        "OMP_NUM_THREADS": str(resources.get("omp_threads") or 1),
        "MKL_NUM_THREADS": str(resources.get("omp_threads") or 1),
        "OPENBLAS_NUM_THREADS": str(resources.get("omp_threads") or 1),
        "NUMEXPR_NUM_THREADS": str(resources.get("omp_threads") or 1),
        "TORCH_NUM_THREADS": str(resources.get("torch_threads") or 1),
        "TORCH_NUM_INTEROP_THREADS": str(resources.get("torch_interop_threads") or 1),
        "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    }


def common_model_args(ctx: PipelineContext) -> dict[str, Any]:
    scientific = ctx.cfg.get("scientific") or {}
    resources = ctx.cfg.get("resources") or {}
    return {
        "seed": int(scientific.get("master_seed") or 42),
        "device": str(resources.get("model_device") or "cuda"),
        "n_jobs": int(resources.get("cpus") or 1),
        "threshold_selection": str(scientific.get("threshold_selection") or "mcc"),
        "min_specificity": float(scientific.get("min_specificity") or 0.50),
        "min_recall": float(scientific.get("min_recall") or 0.0),
    }


def run_stage1_model(ctx: PipelineContext) -> Path:
    cfg = ((ctx.cfg.get("models") or {}).get("stage1") or {})
    common = common_model_args(ctx)
    out = ctx.models_dir / f"stage1_gds_{cfg.get('classifier', 'extra_trees')}"
    cmd = [
        ctx.python,
        "-m",
        "pring_modeling.stage1_tabular",
        "--modeling-dir",
        str(ctx.model_input_dir),
        "--output-dir",
        str(out),
        "--report-dir",
        str(ctx.evaluation_dir / "stage1"),
        "--feature-policy",
        "leakage_safe",
        "--prediction-scope",
        "candidates" if as_bool((ctx.cfg.get("predictions") or {}).get("enabled"), True) else "supervised",
        "--classifier",
        str(cfg.get("classifier") or "extra_trees"),
        "--threshold-selection",
        common["threshold_selection"],
        "--min-specificity",
        str(common["min_specificity"]),
        "--min-recall",
        str(common["min_recall"]),
        "--report-min-specificity",
        "0.50",
        "--report-high-specificity",
        "0.80",
        "--report-min-recall",
        "0.80",
        "--n-estimators",
        str(int(cfg.get("n_estimators") or 1000)),
        "--min-samples-leaf",
        str(int(cfg.get("min_samples_leaf") or 5)),
        "--cv-folds",
        str(int(cfg.get("cv_folds") or 5)),
        "--group-column",
        str(cfg.get("group_column") or "compound_node_id"),
        "--balanced-eval-max-per-class",
        "0",
        "--max-training-rows",
        str(int(cfg.get("max_training_rows") or 0)),
        "--max-scoring-rows",
        str(int(cfg.get("max_candidate_rows") or 0)),
        "--max-predictions-file-rows",
        str(int(cfg.get("max_predictions_file_rows") or 0)),
        "--seed",
        str(common["seed"]),
        "--n-jobs",
        str(common["n_jobs"]),
    ]
    if as_bool(cfg.get("rdkit_features"), False):
        cmd += ["--rdkit-features"]
    run_command(ctx, cmd, stage="component_models", env=model_env(ctx), cwd=ctx.app_dir / "modeling")
    return out


def run_stage2_model(ctx: PipelineContext, model: str) -> Path:
    cfg = ((ctx.cfg.get("models") or {}).get("stage2") or {})
    common = common_model_args(ctx)
    out = ctx.models_dir / f"stage2_{model}_supervised"
    cmd = [
        ctx.python,
        "-m",
        "pring_modeling.stage2_kge",
        "--modeling-dir",
        str(ctx.model_input_dir),
        "--output-dir",
        str(out),
        "--model",
        model,
        "--epochs",
        str(int(cfg.get("epochs") or 150)),
        "--dim",
        str(int(cfg.get("dim") or 128)),
        "--batch-size",
        str(int(cfg.get("batch_size") or 16384)),
        "--score-batch-size",
        str(int(cfg.get("score_batch_size") or 65536)),
        "--max-graph-train-triples",
        str(int(cfg.get("max_graph_train_triples") or 0)),
        "--max-candidate-triples",
        str(int(cfg.get("max_candidate_triples") or 0)),
        "--target-train-repeat",
        str(int(cfg.get("target_train_repeat") or 20)),
        "--loss",
        "softplus",
        "--optimizer",
        "auto",
        "--checkpoint-metric",
        str(cfg.get("checkpoint_metric") or "roc_auc"),
        "--negatives-per-positive",
        str(int(cfg.get("negatives_per_positive") or 5)),
        "--eval-negatives-per-positive",
        str(int(cfg.get("eval_negatives_per_positive") or 10)),
        "--supervised-min-specificity",
        str(common["min_specificity"]),
        "--supervised-min-recall",
        str(common["min_recall"]),
        "--report-min-specificity",
        "0.50",
        "--report-high-specificity",
        "0.80",
        "--report-min-recall",
        "0.80",
        "--seed",
        str(common["seed"]),
        "--device",
        common["device"],
        "--n-jobs",
        str(common["n_jobs"]),
        "--sparse-embeddings",
        "--export-eval-predictions",
        "--train-supervised-decoder",
        "--supervised-decoder",
        str(cfg.get("supervised_decoder") or "extra_trees"),
    ]
    if as_bool(cfg.get("score_candidates"), False):
        cmd.append("--score-candidates")
    else:
        cmd.append("--no-score-candidates")
    run_command(ctx, cmd, stage="component_models", env=model_env(ctx), cwd=ctx.app_dir / "modeling")
    return out


def run_stage3_model(ctx: PipelineContext, kind: str) -> Path:
    cfg = ((ctx.cfg.get("models") or {}).get("stage3") or {})
    common = common_model_args(ctx)
    if kind == "rgcn":
        module = "pring_modeling.stage3_rgcn"
        out = ctx.models_dir / "stage3_rgcn_sampled"
        epochs = int(cfg.get("rgcn_epochs") or 100)
        hidden = int(cfg.get("rgcn_hidden_dim") or 128)
        layers = int(cfg.get("rgcn_layers") or 2)
        neighbors = str(cfg.get("rgcn_neighbors") or "15,10")
        batch = int(cfg.get("rgcn_batch_size") or 128)
        extra: list[str] = []
    else:
        module = "pring_modeling.stage3_hgt"
        out = ctx.models_dir / "stage3_hgt_sampled"
        epochs = int(cfg.get("hgt_epochs") or 100)
        hidden = int(cfg.get("hgt_hidden_dim") or 64)
        layers = int(cfg.get("hgt_layers") or 2)
        neighbors = str(cfg.get("hgt_neighbors") or "10,5")
        batch = int(cfg.get("hgt_batch_size") or 64)
        extra = ["--heads", str(int(cfg.get("hgt_heads") or 2))]
        if as_bool(cfg.get("amp"), False):
            extra.append("--amp")
    cmd = [
        ctx.python,
        "-m",
        module,
        "--modeling-dir",
        str(ctx.model_input_dir),
        "--output-dir",
        str(out),
        "--epochs",
        str(epochs),
        "--hidden-dim",
        str(hidden),
        "--num-layers",
        str(layers),
        "--num-neighbors",
        neighbors,
        "--batch-size",
        str(batch),
        "--dropout",
        str(float(cfg.get("dropout") or 0.2)),
        "--lr",
        str(float(cfg.get("learning_rate") or 0.001)),
        "--seed",
        str(common["seed"]),
        "--device",
        common["device"],
        "--featureless-mode",
        "type",
        "--loss",
        str(cfg.get("loss") or "weighted_bce_bpr"),
        "--bpr-weight",
        str(float(cfg.get("bpr_weight") or 0.5)),
        "--class-weighting",
        str(cfg.get("class_weighting") or "negative_ratio"),
        "--balanced-batches",
        "--balance-ratio",
        str(float(cfg.get("balance_ratio") or 1.0)),
        "--threshold-selection",
        common["threshold_selection"],
        "--min-specificity",
        str(common["min_specificity"]),
        "--min-recall",
        str(common["min_recall"]),
        "--report-min-specificity",
        "0.50",
        "--report-high-specificity",
        "0.80",
        "--report-min-recall",
        "0.80",
        "--early-stopping-metric",
        str(cfg.get("early_stopping_metric") or "mcc"),
        "--patience",
        str(int(cfg.get("patience") or 12)),
    ] + extra
    if as_bool(cfg.get("score_candidates"), True):
        cmd += ["--score-candidates", "--max-candidate-pairs", str(int(cfg.get("max_candidate_pairs") or 0))]
    run_command(ctx, cmd, stage="component_models", env=model_env(ctx), cwd=ctx.app_dir / "modeling")
    return out


def stage_component_models(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    reuse_partial = ctx.resume and "component_models" not in ctx.force_stages and not ctx.dry_run
    reused: list[str] = []

    def ready(directory: Path, required_names: Sequence[str]) -> bool:
        return reuse_partial and all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in required_names)

    s1_expected = ctx.models_dir / "stage1_gds_extra_trees"
    if ready(s1_expected, ["metrics.json", "eval_predictions.csv", "predictions.csv", "stage1_tabular_extra_trees.joblib"]):
        s1 = s1_expected
        reused.append("stage1_gds_extra_trees")
        log(ctx, "[component_models] resume: reusing complete Stage 1 artifacts.")
    else:
        s1 = run_stage1_model(ctx)

    s2_cfg = ((ctx.cfg.get("models") or {}).get("stage2") or {})
    stage2_dirs: list[Path] = []
    if as_bool(s2_cfg.get("enabled"), True):
        for model in s2_cfg.get("models") or ["complex", "distmult", "rotate"]:
            expected = ctx.models_dir / f"stage2_{model}_supervised"
            if ready(expected, ["metrics.json", "supervised_eval_predictions.csv"]):
                stage2_dirs.append(expected)
                reused.append(expected.name)
                log(ctx, f"[component_models] resume: reusing complete {expected.name} artifacts.")
            else:
                stage2_dirs.append(run_stage2_model(ctx, str(model)))

    rgcn_expected = ctx.models_dir / "stage3_rgcn_sampled"
    if ready(rgcn_expected, ["metrics.json", "best_model.pt", "supervised_eval_predictions.csv", "predictions.csv"]):
        rgcn = rgcn_expected
        reused.append(rgcn.name)
        log(ctx, "[component_models] resume: reusing complete R-GCN artifacts.")
    else:
        rgcn = run_stage3_model(ctx, "rgcn")

    hgt_expected = ctx.models_dir / "stage3_hgt_sampled"
    if ready(hgt_expected, ["metrics.json", "best_model.pt", "supervised_eval_predictions.csv", "predictions.csv"]):
        hgt = hgt_expected
        reused.append(hgt.name)
        log(ctx, "[component_models] resume: reusing complete HGT artifacts.")
    else:
        hgt = run_stage3_model(ctx, "hgt")

    outputs = [s1 / "metrics.json", rgcn / "metrics.json", hgt / "metrics.json"] + [p / "metrics.json" for p in stage2_dirs]
    if not ctx.dry_run:
        for p in outputs:
            require_file(p)
    details = {
        "stage1": str(s1),
        "stage2": [str(p) for p in stage2_dirs],
        "stage3_rgcn": str(rgcn),
        "stage3_hgt": str(hgt),
        "reused_completed_component_models": reused,
    }
    return outputs, details, []


def stage_final_validation(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    scientific = ctx.cfg.get("scientific") or {}
    resources = ctx.cfg.get("resources") or {}
    fv = ((ctx.cfg.get("models") or {}).get("final_validation") or {})
    out = ctx.models_dir / "finalized_v2"
    seeds = " ".join(map(str, scientific.get("final_validation_seeds") or [1, 2, 3, 4, 5]))
    cmd = [
        ctx.python,
        "-m",
        "pring_modeling.final_validation",
        "--outputs-root",
        str(ctx.models_dir),
        "--output-dir",
        str(out),
        "--meta-classifier",
        str(scientific.get("final_combiner") or "fixed_mean"),
        "--split-strategy",
        "registered",
        "--calibration",
        str(scientific.get("final_calibration") or "platt"),
        "--seeds",
        seeds,
        "--threshold-selection",
        str(scientific.get("threshold_selection") or "mcc"),
        "--min-specificity",
        str(float(scientific.get("min_specificity") or 0.50)),
        "--min-recall",
        str(float(scientific.get("min_recall") or 0.0)),
        "--report-min-specificity",
        "0.50",
        "--report-high-specificity",
        "0.80",
        "--report-min-recall",
        "0.80",
        "--balanced-eval-max-per-class",
        str(int(fv.get("balanced_eval_max_per_class") or 0)),
        "--bootstrap-resamples",
        str(int(fv.get("bootstrap_resamples") or 1000)),
        "--top-k-per-target",
        str(int(fv.get("top_k_per_target") or 50)),
        "--uncertain-top-n",
        str(int(fv.get("uncertain_top_n") or 200)),
        "--per-target-min-rows",
        str(int(fv.get("per_target_min_rows") or 100)),
        "--n-jobs",
        str(int(resources.get("cpus") or 1)),
        "--strict-leakage-free",
        "--provenance-manifest",
        str(ctx.model_provenance_manifest),
        "--score-columns",
        *DEPLOYABLE_SCORE_COLUMNS,
    ]
    run_command(ctx, cmd, stage="final_validation", env=model_env(ctx), cwd=ctx.app_dir / "modeling")
    comparison = ctx.evaluation_dir / "comparison"
    run_command(
        ctx,
        [
            ctx.python,
            "-m",
            "pring_modeling.compare",
            "metrics",
            "--outputs-root",
            str(ctx.models_dir),
            "--output-dir",
            str(comparison),
            "--primary-metric",
            str(scientific.get("threshold_selection") or "mcc"),
        ],
        stage="final_validation",
        env=model_env(ctx),
        cwd=ctx.app_dir / "modeling",
    )
    run_command(
        ctx,
        [
            ctx.python,
            "-m",
            "pring_modeling.compare",
            "visualize",
            "--comparison-csv",
            str(comparison / "model_comparison.csv"),
            "--output-dir",
            str(comparison / "figures"),
        ],
        stage="final_validation",
        env=model_env(ctx),
        cwd=ctx.app_dir / "modeling",
    )
    metrics_path = out / "metrics.json"
    outputs = [metrics_path, out / "seed_metrics.csv", out / "seed_metric_summary.csv", comparison / "model_comparison.csv"]
    if not ctx.dry_run:
        for p in outputs:
            require_file(p)
        metrics = read_json(metrics_path, {}) or {}
        if metrics.get("publishable") is not True or metrics.get("scientific_release_blockers"):
            raise PipelineError(
                f"Final validation is not publishable: blockers={metrics.get('scientific_release_blockers')}"
            )
        best_seed = int(metrics["best_seed"])
        best_dir = out / f"seed_{best_seed}"
        training_frame = best_dir / "finalized_training_frame.csv"
        production_dir = ctx.models_dir / "production_bundle"
        prod_cmd = [
            ctx.python,
            "-m",
            "pring_modeling.production_bundle",
            "--training-frame",
            str(training_frame),
            "--output-dir",
            str(production_dir),
            "--seed",
            str(best_seed),
            "--source-metrics",
            str(metrics_path),
            "--combiner",
            "fixed_mean",
        ]
        s1_importance = ctx.models_dir / "stage1_gds_extra_trees" / "feature_importance.csv"
        if s1_importance.exists():
            prod_cmd += ["--stage1-feature-importance", str(s1_importance)]
        per_target = best_dir / "per_target_metrics.csv"
        if per_target.exists():
            prod_cmd += ["--per-target-metrics", str(per_target)]
        run_command(ctx, prod_cmd, stage="final_validation", env=model_env(ctx), cwd=ctx.app_dir / "modeling")
        require_file(production_dir / "production_ensemble.joblib")
        require_file(production_dir / "manifest.json")
        outputs += [production_dir / "production_ensemble.joblib", production_dir / "manifest.json"]
        details = metrics
    else:
        details = {}
    return outputs, details, []


# ---------------------------------------------------------------------------
# Final candidate ensemble scoring
# ---------------------------------------------------------------------------


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str], contains: str | None = None) -> str | None:
    cols = list(columns)
    for name in candidates:
        if name in cols:
            return name
    if contains:
        for name in cols:
            if contains.lower() in name.lower():
                return name
    return None


def load_candidate_component(path: Path, score_name: str) -> "Any":
    import pandas as pd
    require_file(path)
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    compound_col = _first_existing_column(
        cols,
        ["compound_node_ref", "compound_entity_id", "compound_node_id", "compound_id", "compound", "head", "source"],
        "compound",
    )
    target_col = _first_existing_column(
        cols,
        ["protein_node_ref", "target_node_ref", "protein_entity_id", "protein_node_id", "target_id", "protein_id", "target", "tail"],
        "protein",
    )
    score_col = _first_existing_column(cols, ["score", "probability", "predicted_probability", "prediction", "raw_score"])
    if not compound_col or not target_col or not score_col:
        raise PipelineError(f"Candidate file lacks compound/target/score columns: {path}; columns={cols}")
    metadata_candidates = [
        "cid", "compound_name", "name", "canonical_smiles", "isomeric_smiles", "inchi_key", "inchikey",
        "protein_id", "target_name", "cyp_symbol", "gene_symbol",
    ]
    usecols = list(dict.fromkeys([compound_col, target_col, score_col] + [c for c in metadata_candidates if c in cols]))
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={compound_col: "compound_key", target_col: "target_key", score_col: score_name})
    df["compound_key"] = df["compound_key"].astype(str)
    df["target_key"] = df["target_key"].astype(str)
    df[score_name] = pd.to_numeric(df[score_name], errors="coerce")
    df = df.dropna(subset=[score_name])
    duplicates = int(df.duplicated(["compound_key", "target_key"]).sum())
    if duplicates:
        # Duplicate scoring rows are not expected. Averaging would hide upstream
        # data problems, so fail explicitly instead.
        raise PipelineError(f"{path} contains {duplicates} duplicate candidate pair(s).")
    return df


def target_name_from_key(value: str, target_map: dict[str, str]) -> str | None:
    text = str(value)
    for name, accession in target_map.items():
        if accession in text or name in text:
            return name
    return None


def stage_candidate_predictions(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    pred_cfg = ctx.cfg.get("predictions") or {}
    if not as_bool(pred_cfg.get("enabled"), True):
        marker = ctx.predictions_dir / "candidate_predictions_disabled.json"
        write_json(marker, {"enabled": False})
        return [marker], {"enabled": False}, ["Candidate prediction stage disabled by configuration."]
    if ctx.dry_run:
        marker = ctx.predictions_dir / "candidate_prediction_plan.json"
        write_json(marker, {"status": "dry_run", "components": DEPLOYABLE_SCORE_COLUMNS})
        return [marker], {"status": "dry_run"}, []

    import joblib
    import numpy as np
    import pandas as pd

    files = {
        DEPLOYABLE_SCORE_COLUMNS[0]: ctx.models_dir / "stage1_gds_extra_trees" / "predictions.csv",
        DEPLOYABLE_SCORE_COLUMNS[1]: ctx.models_dir / "stage3_rgcn_sampled" / "predictions.csv",
        DEPLOYABLE_SCORE_COLUMNS[2]: ctx.models_dir / "stage3_hgt_sampled" / "predictions.csv",
    }
    frames = []
    audit: dict[str, Any] = {"component_files": {k: str(v) for k, v in files.items()}}
    for score_name, path in files.items():
        frame = load_candidate_component(path, score_name)
        audit.setdefault("component_rows", {})[score_name] = int(len(frame))
        frames.append((score_name, frame))
    merged = frames[0][1]
    for _, frame in frames[1:]:
        meta_cols = [c for c in frame.columns if c not in {"compound_key", "target_key"} and not c.startswith("score__")]
        merged = merged.merge(frame, on=["compound_key", "target_key"], how="outer", suffixes=("", "__dup"))
        for col in [c for c in merged.columns if c.endswith("__dup")]:
            base = col[:-5]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[col])
            else:
                merged[base] = merged[col]
            merged.drop(columns=[col], inplace=True)
    missing_counts = {col: int(merged[col].isna().sum()) for col in DEPLOYABLE_SCORE_COLUMNS}
    audit["merged_rows"] = int(len(merged))
    audit["missing_component_scores"] = missing_counts
    if as_bool(pred_cfg.get("strict_component_coverage"), True) and any(missing_counts.values()):
        raise PipelineError(
            "Final candidate ensemble requires complete Stage1/R-GCN/HGT coverage; "
            f"missing={missing_counts}. Check candidate caps/order/filtering across component scorers."
        )
    merged = merged.dropna(subset=DEPLOYABLE_SCORE_COLUMNS).reset_index(drop=True)

    final_metrics = read_json(ctx.models_dir / "finalized_v2" / "metrics.json", {}) or {}
    best_seed = int(final_metrics.get("best_seed"))
    ensemble_file = ctx.models_dir / "finalized_v2" / f"seed_{best_seed}" / "finalized_ensemble.joblib"
    require_file(ensemble_file)
    bundle = joblib.load(ensemble_file)
    score_columns = list(bundle.get("score_columns") or [])
    if score_columns != DEPLOYABLE_SCORE_COLUMNS:
        raise PipelineError(
            f"Final ensemble score schema drift: artifact={score_columns}, expected={DEPLOYABLE_SCORE_COLUMNS}"
        )
    raw_score = bundle["model"].predict_proba(merged[score_columns])[:, 1]
    calibrator = bundle.get("calibrator")
    if calibrator is None:
        calibrated = np.asarray(raw_score, dtype=float).reshape(-1)
    elif hasattr(calibrator, "predict"):
        calibrated = np.asarray(calibrator.predict(raw_score), dtype=float).reshape(-1)
    else:
        raise PipelineError(f"Unsupported calibrator artifact type: {type(calibrator).__name__}")
    threshold = float(bundle["threshold"])
    merged["raw_ensemble_score"] = raw_score
    merged["predicted_probability"] = calibrated
    merged["predicted_class"] = (calibrated >= threshold).astype(int)
    merged["decision_threshold"] = threshold
    merged["model_used"] = "finalized_v2_fixed_mean_platt"
    merged["model_seed"] = best_seed
    merged["base_score_mean"] = merged[score_columns].mean(axis=1)
    merged["base_score_std"] = merged[score_columns].std(axis=1)
    merged["cyp_enzyme"] = merged["target_key"].map(lambda x: target_name_from_key(str(x), ctx.target_map))
    merged["compound_identifier"] = merged["compound_key"]
    merged["target_identifier"] = merged["target_key"]
    merged["cyp_accession"] = merged["cyp_enzyme"].map(ctx.target_map)
    unknown_targets = int(merged["cyp_enzyme"].isna().sum())
    if unknown_targets:
        raise PipelineError(f"Could not map {unknown_targets} candidate target rows to configured CYP enzymes.")
    # Rank independently within each CYP. Ties use the minimum rank to avoid
    # implying arbitrary precision beyond the score.
    merged["rank"] = merged.groupby("cyp_enzyme")["predicted_probability"].rank(method="min", ascending=False).astype("int64")
    merged = merged.sort_values(["cyp_enzyme", "rank", "compound_key"]).reset_index(drop=True)

    outputs: list[Path] = []
    if as_bool(pred_cfg.get("write_parquet"), True):
        parquet = ctx.predictions_dir / "all_candidate_predictions.parquet"
        merged.to_parquet(parquet, index=False)
        outputs.append(parquet)
    if as_bool(pred_cfg.get("write_full_csv"), True):
        full_csv = ctx.predictions_dir / "all_candidate_predictions.csv"
        merged.to_csv(full_csv, index=False)
        outputs.append(full_csv)

    summary_rows: list[dict[str, Any]] = []
    top_n = int(pred_cfg.get("top_n_per_cyp") or 100)
    for cyp in ctx.selected_cyps:
        cyp_dir = ctx.predictions_dir / cyp
        cyp_dir.mkdir(parents=True, exist_ok=True)
        subset = merged.loc[merged["cyp_enzyme"] == cyp].copy()
        if subset.empty:
            raise PipelineError(f"No candidate predictions were generated for {cyp}.")
        full_path = cyp_dir / f"{cyp}_ranked_predictions.csv"
        top_path = cyp_dir / f"{cyp}_top_{top_n}_predictions.csv"
        subset.to_csv(full_path, index=False)
        subset.head(top_n).to_csv(top_path, index=False)
        outputs += [full_path, top_path]
        summary_rows.append({
            "cyp_enzyme": cyp,
            "accession": ctx.target_map[cyp],
            "candidate_count": int(len(subset)),
            "predicted_positive_count": int(subset["predicted_class"].sum()),
            "threshold": threshold,
            "max_probability": float(subset["predicted_probability"].max()),
            "median_probability": float(subset["predicted_probability"].median()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = ctx.tables_dir / "candidate_prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)
    audit.update({
        "best_seed": best_seed,
        "threshold": threshold,
        "final_rows": int(len(merged)),
        "per_cyp": summary_rows,
        "ensemble_artifact": str(ensemble_file),
        "ensemble_artifact_sha256": hash_file(ensemble_file),
    })
    audit_path = ctx.predictions_dir / "candidate_prediction_audit.json"
    write_json(audit_path, audit)
    outputs.append(audit_path)
    return outputs, audit, []


# ---------------------------------------------------------------------------
# Thesis reporting
# ---------------------------------------------------------------------------


def save_figure_formats(fig: Any, base: Path, reporting: dict[str, Any]) -> list[Path]:
    outputs: list[Path] = []
    dpi = int(reporting.get("figure_dpi") or 400)
    base.parent.mkdir(parents=True, exist_ok=True)
    if as_bool(reporting.get("save_png"), True):
        path = base.with_suffix(".png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    if as_bool(reporting.get("save_pdf"), True):
        path = base.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    if as_bool(reporting.get("save_svg"), True):
        path = base.with_suffix(".svg")
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    return outputs


def roc_curve_numpy(y_true: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    tp = np.cumsum(y == 1) / pos
    fp = np.cumsum(y == 0) / neg
    return np.concatenate([[0.0], fp, [1.0]]), np.concatenate([[0.0], tp, [1.0]])


def pr_curve_numpy(y_true: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int((y == 1).sum()))
    return np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision])


def confusion_counts(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    import numpy as np
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    return tn, fp, fn, tp


def stage_thesis_reporting(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    if ctx.dry_run:
        marker = ctx.figures_dir / "thesis_reporting_plan.json"
        write_json(marker, {"status": "dry_run"})
        return [marker], {"status": "dry_run"}, []
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    reporting = ctx.cfg.get("reporting") or {}
    outputs: list[Path] = []
    details: dict[str, Any] = {}
    final_metrics = read_json(ctx.models_dir / "finalized_v2" / "metrics.json", {}) or {}
    best_seed = int(final_metrics["best_seed"])
    best_dir = ctx.models_dir / "finalized_v2" / f"seed_{best_seed}"
    eval_pred = pd.read_csv(best_dir / "predictions.csv")
    if "target_key" not in eval_pred.columns:
        raise PipelineError("Final evaluation predictions lack target_key.")
    eval_pred["cyp_enzyme"] = eval_pred["target_key"].map(lambda x: target_name_from_key(str(x), ctx.target_map))
    eval_pred = eval_pred.dropna(subset=["cyp_enzyme"])

    # 1. ROC curves per CYP on the registered test partition.
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for cyp in ctx.selected_cyps:
        g = eval_pred[eval_pred["cyp_enzyme"] == cyp]
        if g.empty or g["label"].nunique() < 2:
            continue
        fpr, tpr = roc_curve_numpy(g["label"], g["calibrated_score"])
        ax.plot(fpr, tpr, label=cyp)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PRING Final Ensemble — CYP-specific ROC Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    outputs += save_figure_formats(fig, ctx.figures_dir / "final_ensemble_roc_by_cyp", reporting)
    plt.close(fig)

    # 2. PR curves per CYP.
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for cyp in ctx.selected_cyps:
        g = eval_pred[eval_pred["cyp_enzyme"] == cyp]
        if g.empty or g["label"].nunique() < 2:
            continue
        recall, precision = pr_curve_numpy(g["label"], g["calibrated_score"])
        ax.plot(recall, precision, label=cyp)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PRING Final Ensemble — CYP-specific Precision–Recall Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    outputs += save_figure_formats(fig, ctx.figures_dir / "final_ensemble_pr_by_cyp", reporting)
    plt.close(fig)

    # 3. Confusion matrices and exact table.
    confusion_rows = []
    for cyp in ctx.selected_cyps:
        g = eval_pred[eval_pred["cyp_enzyme"] == cyp]
        if g.empty:
            continue
        tn, fp, fn, tp = confusion_counts(g["label"], g["predicted_label"])
        confusion_rows.append({"cyp_enzyme": cyp, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        matrix = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(matrix)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(f"{cyp} — Final Ensemble Confusion Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        outputs += save_figure_formats(fig, ctx.figures_dir / f"{cyp}_confusion_matrix", reporting)
        plt.close(fig)
    confusion_table = ctx.tables_dir / "final_ensemble_confusion_matrices.csv"
    pd.DataFrame(confusion_rows).to_csv(confusion_table, index=False)
    outputs.append(confusion_table)

    # 4. Per-CYP metric table from the final evaluation artifact.
    per_target_file = best_dir / "per_target_metrics.csv"
    if per_target_file.exists():
        per_target = pd.read_csv(per_target_file)
        if "target_key" in per_target.columns:
            per_target["cyp_enzyme"] = per_target["target_key"].map(lambda x: target_name_from_key(str(x), ctx.target_map))
        per_target_out = ctx.tables_dir / "final_ensemble_per_cyp_metrics.csv"
        per_target.to_csv(per_target_out, index=False)
        outputs.append(per_target_out)
        metric_candidates = [m for m in ["mcc", "balanced_accuracy", "roc_auc", "average_precision", "f1", "recall", "specificity"] if m in per_target.columns]
        if metric_candidates and "cyp_enzyme" in per_target.columns:
            plot = per_target.dropna(subset=["cyp_enzyme"]).set_index("cyp_enzyme")[metric_candidates]
            fig, ax = plt.subplots(figsize=(10, 5.8))
            plot.plot(kind="bar", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Metric value")
            ax.set_title("Final Ensemble Performance by CYP450 Enzyme")
            ax.legend(frameon=False, ncol=2)
            ax.grid(axis="y", alpha=0.2)
            outputs += save_figure_formats(fig, ctx.figures_dir / "final_ensemble_per_cyp_metrics", reporting)
            plt.close(fig)

    # 5. Model comparison table/figure.
    comp_file = ctx.evaluation_dir / "comparison" / "model_comparison.csv"
    if comp_file.exists():
        comp = pd.read_csv(comp_file)
        comp_out = ctx.tables_dir / "model_comparison.csv"
        comp.to_csv(comp_out, index=False)
        outputs.append(comp_out)
        metric_col = next((c for c in ["mcc", "balanced_accuracy", "average_precision", "roc_auc"] if c in comp.columns), None)
        model_col = next((c for c in ["model", "model_name", "stage"] if c in comp.columns), None)
        if metric_col and model_col:
            top = comp.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.32)))
            ax.barh(top[model_col].astype(str), top[metric_col].astype(float))
            ax.invert_yaxis()
            ax.set_xlabel(metric_col.replace("_", " ").title())
            ax.set_title("PRING Model Comparison")
            ax.grid(axis="x", alpha=0.2)
            outputs += save_figure_formats(fig, ctx.figures_dir / "model_comparison", reporting)
            plt.close(fig)

    # 6. Per-CYP comparison across all evaluated component models and final ensemble.
    per_cyp_sources = [
        ("stage1_gds_extra_trees", "Stage 1 / Extra Trees", ctx.models_dir / "stage1_gds_extra_trees" / "per_target_metrics.csv", False),
        ("stage2_complex_supervised", "Stage 2 / ComplEx + supervised decoder", ctx.models_dir / "stage2_complex_supervised" / "supervised_per_target_metrics.csv", False),
        ("stage2_distmult_supervised", "Stage 2 / DistMult + supervised decoder", ctx.models_dir / "stage2_distmult_supervised" / "supervised_per_target_metrics.csv", False),
        ("stage2_rotate_supervised", "Stage 2 / RotatE + supervised decoder", ctx.models_dir / "stage2_rotate_supervised" / "supervised_per_target_metrics.csv", False),
        ("stage3_rgcn_sampled", "Stage 3 / R-GCN", ctx.models_dir / "stage3_rgcn_sampled" / "per_target_metrics.csv", False),
        ("stage3_hgt_sampled", "Stage 3 / HGT", ctx.models_dir / "stage3_hgt_sampled" / "per_target_metrics.csv", False),
        ("finalized_v2_fixed_mean_platt", "Final / fixed mean + Platt", best_dir / "per_target_metrics.csv", True),
    ]
    per_cyp_rows = []
    for model_name, family, path, deployed in per_cyp_sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        target_col = next((c for c in ["target_key", "group", "target", "protein_node_ref", "protein_node_id"] if c in frame.columns), None)
        if target_col is None:
            continue
        for _, row in frame.iterrows():
            cyp = target_name_from_key(str(row[target_col]), ctx.target_map)
            if cyp is None:
                continue
            record = {
                "cyp_enzyme": cyp,
                "cyp_accession": ctx.target_map[cyp],
                "model": model_name,
                "model_family": family,
                "deployed_final_model": bool(deployed),
            }
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "mcc", "n"]:
                if metric in frame.columns:
                    record[metric] = row[metric]
            per_cyp_rows.append(record)
    if per_cyp_rows:
        per_cyp_models = pd.DataFrame(per_cyp_rows)
        per_cyp_models["best_evaluated_by_mcc"] = False
        if "mcc" in per_cyp_models.columns:
            for cyp, index in per_cyp_models.groupby("cyp_enzyme")["mcc"].idxmax().items():
                per_cyp_models.loc[index, "best_evaluated_by_mcc"] = True
        per_cyp_model_path = ctx.tables_dir / "per_cyp_model_comparison.csv"
        per_cyp_models.to_csv(per_cyp_model_path, index=False)
        outputs.append(per_cyp_model_path)
        selection_cols = [c for c in ["cyp_enzyme", "cyp_accession", "model", "model_family", "mcc", "balanced_accuracy", "roc_auc", "average_precision"] if c in per_cyp_models.columns]
        selected = per_cyp_models.loc[per_cyp_models["best_evaluated_by_mcc"], selection_cols].copy()
        selected = selected.rename(columns={"model": "best_evaluated_model_by_mcc", "model_family": "best_evaluated_model_family"})
        selected["production_prediction_model"] = "finalized_v2_fixed_mean_platt"
        selected["production_model_note"] = "The deployable final ensemble remains fixed by the validated PRING contract; per-CYP best-MCC is descriptive reporting only."
        selection_path = ctx.tables_dir / "per_cyp_model_selection_summary.csv"
        selected.to_csv(selection_path, index=False)
        outputs.append(selection_path)

    # 7. Final candidate probability distributions.
    candidate_parquet = ctx.predictions_dir / "all_candidate_predictions.parquet"
    candidate_csv = ctx.predictions_dir / "all_candidate_predictions.csv"
    if candidate_parquet.exists():
        candidates = pd.read_parquet(candidate_parquet, columns=["cyp_enzyme", "predicted_probability"])
    elif candidate_csv.exists():
        candidates = pd.read_csv(candidate_csv, usecols=["cyp_enzyme", "predicted_probability"])
    else:
        candidates = pd.DataFrame()
    if not candidates.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for cyp in ctx.selected_cyps:
            values = candidates.loc[candidates["cyp_enzyme"] == cyp, "predicted_probability"].dropna()
            if len(values):
                ax.hist(values, bins=50, histtype="step", density=True, label=cyp)
        ax.set_xlabel("Calibrated predicted interaction probability")
        ax.set_ylabel("Density")
        ax.set_title("Candidate Prediction Probability Distributions")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        outputs += save_figure_formats(fig, ctx.figures_dir / "candidate_probability_distributions", reporting)
        plt.close(fig)

    # 8. Exact graph composition tables are already created; add a figure.
    node_counts = ctx.tables_dir / "knowledge_graph_node_counts.csv"
    if node_counts.exists():
        nodes = pd.read_csv(node_counts).head(20)
        fig, ax = plt.subplots(figsize=(9, max(5, len(nodes) * 0.3)))
        ax.barh(nodes["node_label"].astype(str), nodes["count"].astype(float))
        ax.invert_yaxis()
        ax.set_xlabel("Node count")
        ax.set_title("PRING Knowledge Graph Composition — Top Node Labels")
        ax.grid(axis="x", alpha=0.2)
        outputs += save_figure_formats(fig, ctx.figures_dir / "knowledge_graph_node_composition", reporting)
        plt.close(fig)

    details["best_seed"] = best_seed
    details["figures_created"] = [str(p) for p in outputs if p.suffix.lower() in {".png", ".pdf", ".svg"}]
    details["tables_created"] = [str(p) for p in outputs if p.suffix.lower() == ".csv"]
    manifest = ctx.manifest_dir / "thesis_artifact_manifest.json"
    write_json(manifest, details)
    outputs.append(manifest)
    return outputs, details, []


# ---------------------------------------------------------------------------
# Final QC
# ---------------------------------------------------------------------------


def stage_elapsed_summary(ctx: PipelineContext) -> dict[str, float]:
    result: dict[str, float] = {}
    for stage in STAGES:
        payload = read_json(checkpoint_path(ctx, stage), {}) or {}
        if payload.get("status") == "complete":
            result[stage] = float(payload.get("elapsed_seconds") or 0.0)
    return result


def final_qc_payload(ctx: PipelineContext) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": safe_json(detail)})

    add("five canonical CYP targets", ctx.target_map == EXPECTED_FIVE_CYPS, ctx.target_map)
    initial = read_json(ctx.manifest_dir / "run_manifest_initial.json", {}) or {}
    package_git = initial.get("package_git") or {}
    app_git = initial.get("app_git") or {}
    require_clean = as_bool((ctx.cfg.get("project") or {}).get("require_clean_git"), True)
    add("PRING-PACKAGE Git provenance", bool(package_git.get("commit") and package_git.get("commit") != "unknown"), package_git)
    add("PRING-APP Git provenance", bool(app_git.get("commit") and app_git.get("commit") != "unknown"), app_git)
    if require_clean:
        add("clean Git worktrees", not package_git.get("dirty") and not app_git.get("dirty"), {"package": package_git.get("dirty"), "app": app_git.get("dirty")})

    source_validation = read_json(ctx.validation_dir / "source_run_validation.json", {}) or {}
    q = source_validation.get("quality_gate") or {}
    add("uncapped source run", q.get("cap_status") == "uncapped_or_no_internal_caps_detected", q.get("cap_status"))
    add("all candidate pairs exported", str(q.get("candidate_mode")).lower() == "all", q.get("candidate_mode"))
    add("source pipeline validation ready", q.get("pipeline_validation_ready") is True, q.get("pipeline_validation_ready"))

    prepared = read_json(ctx.validation_dir / "prepared_readiness.json", {}) or {}
    add("prepared artifact release gate", prepared.get("status") == "pass", prepared.get("status"))

    kg = read_json(ctx.kg_dir / "neo4j_graph_statistics.json", {}) or {}
    add("Neo4j graph populated", int(kg.get("total_nodes") or 0) > 0 and int(kg.get("total_relationships") or 0) > 0, {"nodes": kg.get("total_nodes"), "relationships": kg.get("total_relationships")})
    cyp_graph = kg.get("cyp450") or {}
    add("five CYP nodes present in Neo4j", all(int((cyp_graph.get(name) or {}).get("protein_nodes") or 0) == 1 for name in EXPECTED_FIVE_CYPS), cyp_graph)

    model_metrics = {}
    for name, path in {
        "stage1": ctx.models_dir / "stage1_gds_extra_trees" / "metrics.json",
        "stage2_complex": ctx.models_dir / "stage2_complex_supervised" / "metrics.json",
        "stage2_distmult": ctx.models_dir / "stage2_distmult_supervised" / "metrics.json",
        "stage2_rotate": ctx.models_dir / "stage2_rotate_supervised" / "metrics.json",
        "stage3_rgcn": ctx.models_dir / "stage3_rgcn_sampled" / "metrics.json",
        "stage3_hgt": ctx.models_dir / "stage3_hgt_sampled" / "metrics.json",
    }.items():
        present = path.exists() and path.stat().st_size > 0
        add(f"{name} completed", present, str(path))
        if present:
            model_metrics[name] = read_json(path, {}) or {}

    final = read_json(ctx.models_dir / "finalized_v2" / "metrics.json", {}) or {}
    add("final ensemble publishable", final.get("publishable") is True, final.get("scientific_release_blockers"))
    final_split_audit = final.get("split_audit") or {}
    registered_split_ok = (
        final_split_audit.get("split_strategy") == "registered"
        and final_split_audit.get("supplied_split_found") is True
        and final_split_audit.get("diagnostic_split_created") is False
        and int(final_split_audit.get("compound_groups_crossing_partitions") or 0) == 0
        and int(final_split_audit.get("conflicting_rows") or 0) == 0
        and int(final_split_audit.get("unknown_rows") or 0) == 0
    )
    add("registered final split", registered_split_ok, final_split_audit)
    add("fixed mean preselected combiner", final.get("base_score_protocol") == "fixed_equal_weight_combiner_selected_before_test_no_meta_training", final.get("base_score_protocol"))
    add("five final validation seeds", len(final.get("seeds") or []) >= 5, final.get("seeds"))

    candidate_audit = read_json(ctx.predictions_dir / "candidate_prediction_audit.json", {}) or {}
    per_cyp = candidate_audit.get("per_cyp") or []
    candidate_names = {row.get("cyp_enzyme") for row in per_cyp}
    add("candidate predictions for all five CYPs", set(EXPECTED_FIVE_CYPS).issubset(candidate_names), per_cyp)
    add("complete ensemble candidate coverage", not any((candidate_audit.get("missing_component_scores") or {}).values()), candidate_audit.get("missing_component_scores"))
    add("non-empty final candidate prediction set", int(candidate_audit.get("final_rows") or 0) > 0, candidate_audit.get("final_rows"))

    checksum = ctx.manifest_dir / "source_input_checksums.csv"
    add("input checksum manifest", checksum.exists() and checksum.stat().st_size > 0, str(checksum))
    thesis_manifest = ctx.manifest_dir / "thesis_artifact_manifest.json"
    add("thesis figures/tables manifest", thesis_manifest.exists() and thesis_manifest.stat().st_size > 0, str(thesis_manifest))

    failed = [c for c in checks if c["status"] != "pass"]
    return {
        "generated_at_utc": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "PASS_FINAL_THESIS_RUN" if not failed else "FAIL_NOT_FINAL_THESIS_READY",
        "suitable_as_final_thesis_run": not failed,
        "checks": checks,
        "failed_checks": failed,
        "final_ensemble": final,
        "candidate_prediction_summary": per_cyp,
        "knowledge_graph": {
            "total_nodes": kg.get("total_nodes"),
            "total_relationships": kg.get("total_relationships"),
            "node_counts_by_label": kg.get("node_counts_by_label"),
            "relationship_counts_by_type": kg.get("relationship_counts_by_type"),
            "connected_components": kg.get("connected_components"),
            "degree_summary": kg.get("degree_summary"),
        },
        "stage_elapsed_seconds": stage_elapsed_summary(ctx),
    }


def render_qc_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PRING Final Thesis Run — Quality-Control Report",
        "",
        f"**Status:** `{payload['status']}`",
        "",
        f"**Suitable as final MSc thesis run:** **{'YES' if payload['suitable_as_final_thesis_run'] else 'NO'}**",
        "",
        "## Release checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in payload.get("checks") or []:
        detail = json.dumps(check.get("detail"), ensure_ascii=False)
        if len(detail) > 240:
            detail = detail[:237] + "..."
        lines.append(f"| {check['name']} | {check['status'].upper()} | `{detail.replace('|', '/')}` |")
    lines += ["", "## Stage run times", "", "| Stage | Seconds |", "|---|---:|"]
    for stage, seconds in (payload.get("stage_elapsed_seconds") or {}).items():
        lines.append(f"| {stage} | {seconds:.1f} |")
    lines += ["", "## Final candidate prediction counts", "", "| CYP | Candidates | Predicted positive |", "|---|---:|---:|"]
    for row in payload.get("candidate_prediction_summary") or []:
        lines.append(f"| {row.get('cyp_enzyme')} | {row.get('candidate_count')} | {row.get('predicted_positive_count')} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "A PASS means the configured run satisfied the implemented reproducibility, leakage-control, graph, model, prediction, and artifact gates. It does not constitute experimental or clinical validation of individual compound–CYP predictions.",
        "",
    ]
    return "\n".join(lines)


def stage_final_qc(ctx: PipelineContext) -> tuple[list[Path], dict[str, Any], list[str]]:
    final_validator_json, final_validator_md = run_artifact_validator(
        ctx,
        "final",
        extra=[
            "--eda-dir",
            str(ctx.eda_dir),
            "--model-output-dir",
            str(ctx.models_dir),
            "--model-report-dir",
            str(ctx.evaluation_dir),
            "--min-final-seeds",
            "5",
        ],
    )
    if ctx.dry_run:
        marker = ctx.validation_dir / "final_qc_report.json"
        write_json(marker, {"status": "dry_run"})
        return [marker], {"status": "dry_run"}, []
    payload = final_qc_payload(ctx)
    json_path = ctx.validation_dir / "final_qc_report.json"
    md_path = ctx.validation_dir / "final_qc_report.md"
    write_json(json_path, payload)
    md_path.write_text(render_qc_markdown(payload), encoding="utf-8")

    # Final immutable manifest folds in stage timings and key artifact digests.
    initial = read_json(ctx.manifest_dir / "run_manifest_initial.json", {}) or {}
    final_manifest = {
        **initial,
        "finalized_at_utc": utc_now(),
        "stage_elapsed_seconds": payload.get("stage_elapsed_seconds"),
        "source_run_dir": str(source_run_dir(ctx)),
        "ready_run_dir": str(ctx.ready_run_dir),
        "output_root": str(ctx.output_root),
        "final_qc_status": payload.get("status"),
        "suitable_as_final_thesis_run": payload.get("suitable_as_final_thesis_run"),
        "modeling_provenance": read_json(ctx.model_provenance_manifest, {}),
        "final_ensemble_metrics_sha256": hash_file(ctx.models_dir / "finalized_v2" / "metrics.json"),
        "candidate_prediction_audit_sha256": hash_file(ctx.predictions_dir / "candidate_prediction_audit.json"),
        "graph_statistics_sha256": hash_file(ctx.kg_dir / "neo4j_graph_statistics.json"),
        "source_checksums_sha256": hash_file(ctx.manifest_dir / "source_input_checksums.csv"),
    }
    final_manifest_path = ctx.manifest_dir / "run_manifest_final.json"
    write_json(final_manifest_path, final_manifest)

    # Concise handoff index.
    index = ctx.output_root / "THESIS_RESULTS_INDEX.md"
    index.write_text(
        textwrap.dedent(
            f"""
            # PRING Final Thesis Results Index

            - Final QC: `{md_path}`
            - Final run manifest: `{final_manifest_path}`
            - Source validation: `{ctx.validation_dir / 'source_run_validation.json'}`
            - EDA report: `{ctx.eda_dir / 'eda_report.html'}`
            - Neo4j graph statistics: `{ctx.kg_dir / 'neo4j_graph_statistics.json'}`
            - Modeling provenance: `{ctx.model_provenance_manifest}`
            - Model comparison: `{ctx.evaluation_dir / 'comparison' / 'model_comparison.csv'}`
            - Final ensemble metrics: `{ctx.models_dir / 'finalized_v2' / 'metrics.json'}`
            - Final candidate prediction audit: `{ctx.predictions_dir / 'candidate_prediction_audit.json'}`
            - Thesis figures: `{ctx.figures_dir}`
            - Thesis tables: `{ctx.tables_dir}`

            Final status: **{payload['status']}**
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    if not payload["suitable_as_final_thesis_run"]:
        failed_names = [x["name"] for x in payload.get("failed_checks") or []]
        raise PipelineError(f"Final thesis QC failed: {failed_names}. See {md_path}")
    return [final_validator_json, final_validator_md, json_path, md_path, final_manifest_path, index], payload, []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the final reproducible PRING five-CYP MSc thesis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=False, help="YAML configuration file.")
    p.add_argument("--output", default=None, help="Override project.output_root.")
    p.add_argument("--resume", action="store_true", help="Reuse a completed stage only when its checkpoint and required outputs validate.")
    p.add_argument("--force-stage", action="append", choices=STAGES, default=[], help="Rerun a stage even if it has a valid checkpoint. Never overwrites an existing raw source run.")
    p.add_argument("--stage", action="append", default=None, help="Run one/more stages, comma-separated names, or a range such as eda:final_validation.")
    p.add_argument("--cyp", action="append", choices=list(EXPECTED_FIVE_CYPS), default=None, help="Run a CYP subset for smoke/debug use. A subset cannot pass final-thesis QC.")
    p.add_argument("--dry-run", action="store_true", help="Validate configuration and print commands without executing expensive stages.")
    p.add_argument("--list-stages", action="store_true", help="Print stage names and exit.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_stages:
        print("\n".join(STAGES))
        return 0
    if not args.config:
        build_parser().error("--config is required unless --list-stages is used")
    ctx = resolve_context(args)
    init_output_dirs(ctx)
    log(ctx, f"PRING final thesis runner v{PIPELINE_VERSION}")
    log(ctx, f"Config: {ctx.config_path}")
    log(ctx, f"Output: {ctx.output_root}")
    log(ctx, f"Stages: {ctx.selected_stages}")
    log(ctx, f"Targets: {ctx.target_map}")
    if ctx.dry_run:
        log(ctx, "DRY-RUN mode: commands are rendered but expensive subprocesses are not executed.")

    stage_functions: dict[str, Callable[[PipelineContext], tuple[list[Path], dict[str, Any], list[str]]]] = {
        "preflight": stage_preflight,
        "data_collection": stage_data_collection,
        "modeling_data": stage_modeling_data,
        "eda": stage_eda,
        "knowledge_graph": stage_knowledge_graph,
        "features_embeddings": stage_features_embeddings,
        "component_models": stage_component_models,
        "final_validation": stage_final_validation,
        "candidate_predictions": stage_candidate_predictions,
        "thesis_reporting": stage_thesis_reporting,
        "final_qc": stage_final_qc,
    }

    try:
        for stage in STAGES:
            stage_wrapper(ctx, stage, stage_functions[stage])
    except Exception as exc:
        log(ctx, f"PIPELINE FAILED: {type(exc).__name__}: {exc}")
        return 2
    log(ctx, "PIPELINE COMPLETED SUCCESSFULLY")
    if not ctx.dry_run and "final_qc" in ctx.selected_stages:
        log(ctx, f"Final QC: {ctx.validation_dir / 'final_qc_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
