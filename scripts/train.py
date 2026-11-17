"""Re-train the selected HGB model and grouped threshold experiment from supplied features.

Writes to a separate directory and never overwrites the published reference results.
"""

import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import model
from src.alarms import choose, evaluate, aggregate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/development.csv")
    X = np.load(ROOT / "data/development_features.npz")["calibrated"]
    y = meta.y.to_numpy()
    groups = meta.well.to_numpy()
    outer_scores = np.full(len(y), np.nan)
    events = []
    thresholds = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        inner = np.full(len(tr), np.nan)
        for itr, ite in GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]):
            m = model("HGB")
            m.fit(X[tr[itr]], y[tr[itr]])
            inner[ite] = m.predict_proba(X[tr[ite]])[:, 1]
        selected, _ = choose(meta.iloc[tr], inner)
        m = model("HGB")
        m.fit(X[tr], y[tr])
        scores = m.predict_proba(X[te])[:, 1]
        outer_scores[te] = scores
        for policy, t in selected.items():
            events.extend(
                dict(policy=policy, threshold=t, **r)
                for r in evaluate(meta.iloc[te], scores, t)
            )
            thresholds.append(dict(well=groups[te][0], policy=policy, threshold=t))
        print(groups[te][0], flush=True)
    selected, _ = choose(meta, outer_scores)
    m = model("HGB")
    m.fit(X, y)
    joblib.dump(m, a.output / "calibrated_HGB.joblib")
    (a.output / "calibrated_HGB_policy.json").write_text(
        json.dumps({"model": "calibrated_HGB", "thresholds": selected}, indent=2)
    )
    meta.assign(calibrated_HGB=outer_scores).to_csv(
        a.output / "predictions.csv", index=False
    )
    pd.DataFrame(events).to_csv(a.output / "events.csv", index=False)
    pd.DataFrame(thresholds).to_csv(a.output / "thresholds.csv", index=False)
    summary = [
        dict(policy=p, **aggregate([r for r in events if r["policy"] == p]))
        for p in selected
    ]
    (a.output / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
