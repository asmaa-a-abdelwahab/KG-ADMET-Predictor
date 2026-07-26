from pathlib import Path
import json, sys
import joblib, pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
model_dir = root / "artifacts/models/production"
score_file = root / "artifacts/results/production/finalized_training_frame.csv"
required = [model_dir/"production_ensemble.joblib", model_dir/"manifest.json", score_file]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing production assets:\n- " + "\n- ".join(missing))
bundle = joblib.load(model_dir/"production_ensemble.joblib")
manifest = json.loads((model_dir/"manifest.json").read_text())
score_cols = list(bundle["score_columns"])
frame = pd.read_csv(score_file, nrows=10)
needed = ["compound_key", "target_key", *score_cols]
absent = [c for c in needed if c not in frame.columns]
if absent:
    raise SystemExit("Score frame missing columns: " + ", ".join(absent))
print("Production assets valid")
print("Model:", manifest.get("model_name"))
print("Score columns:", score_cols)
print("Score file:", score_file)
