from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move legacy production-inference rows out of finalized_training_frame.csv into a separate prediction cache."
    )
    parser.add_argument("--reference-frame", required=True, type=Path)
    parser.add_argument("--cache-frame", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Write the migration. Without this flag, only print a dry-run summary.")
    args = parser.parse_args()

    if not args.reference_frame.exists():
        raise SystemExit(f"Reference frame not found: {args.reference_frame}")
    frame = pd.read_csv(args.reference_frame, low_memory=False)
    mask = pd.Series(False, index=frame.index)
    if "record_type" in frame.columns:
        mask |= frame["record_type"].astype(str).str.casefold().eq("production_prediction_cache")
    if "final_split" in frame.columns:
        mask |= frame["final_split"].astype(str).str.casefold().eq("production_inference")

    cache_rows = frame.loc[mask].copy()
    reference_rows = frame.loc[~mask].copy()
    print(f"Reference rows before: {len(frame)}")
    print(f"Legacy production-cache rows found: {len(cache_rows)}")
    print(f"Reference rows after: {len(reference_rows)}")
    print(f"Target cache: {args.cache_frame}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to write files.")
        return 0
    if cache_rows.empty:
        print("No legacy cache rows found; no files changed.")
        return 0

    args.cache_frame.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_frame.exists():
        existing = pd.read_csv(args.cache_frame, low_memory=False)
        combined = pd.concat([existing, cache_rows], ignore_index=True)
        combined = combined.drop_duplicates(["compound_key", "target_key"], keep="last")
    else:
        combined = cache_rows

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.reference_frame.with_name(f"{args.reference_frame.stem}.before_cache_migration_{stamp}{args.reference_frame.suffix}")
    shutil.copy2(args.reference_frame, backup)

    reference_tmp = args.reference_frame.with_suffix(args.reference_frame.suffix + ".tmp")
    cache_tmp = args.cache_frame.with_suffix(args.cache_frame.suffix + ".tmp")
    reference_rows.to_csv(reference_tmp, index=False)
    combined.to_csv(cache_tmp, index=False)
    reference_tmp.replace(args.reference_frame)
    cache_tmp.replace(args.cache_frame)

    print(f"Backup: {backup}")
    print(f"Updated immutable reference frame: {args.reference_frame}")
    print(f"Updated production cache: {args.cache_frame}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
