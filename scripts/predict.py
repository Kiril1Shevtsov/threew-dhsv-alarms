"""Score a 1 Hz 3W Parquet recording with the saved HGB model."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import features, BURNIN, STEP


def predict(path, model_name="calibrated_HGB", policy_name="events_1_day"):
    frame = pd.read_parquet(path)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Index must be unique and chronological.")
    if len(frame) <= BURNIN:
        raise ValueError(
            "Recording must be longer than the 3600-second normalization period."
        )
    if not np.all(np.diff(frame.index.view("int64")) == 10**9):
        raise ValueError("Expected a 1 Hz datetime index.")
    policy = json.loads((ROOT / "models" / f"{model_name}_policy.json").read_text())
    # Only load the trusted model distributed with this repository.
    model = joblib.load(ROOT / "models" / f"{model_name}.joblib")
    scores = model.predict_proba(
        features(frame, not model_name.startswith("raw_")).to_numpy()
    )[:, 1]
    threshold = policy["thresholds"][policy_name]
    hit = scores >= threshold
    confirmed = np.zeros(len(hit), bool)
    confirmed[2:] = hit[2:] & hit[1:-1] & hit[:-2]
    return pd.DataFrame(
        {
            "timestamp": frame.index[BURNIN::STEP],
            "second": np.arange(BURNIN, len(frame), STEP),
            "score": scores,
            "threshold": threshold,
            "alarm": confirmed,
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="calibrated_HGB")
    p.add_argument("--policy", default="events_1_day")
    a = p.parse_args()
    predict(a.parquet, a.model, a.policy).to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
