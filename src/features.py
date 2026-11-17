"""Causal sensor features and baseline classifiers for the 3W DHSV study."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

SENSORS = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "QGL"]
FLOORS = np.array([1e4, 1e4, 1.0, 1e4, 1.0, 0.001])
WINDOW = 180
STEP = 15
BURNIN = 3600
SEED = 20261109


def features(d, normalized):
    a = d[SENSORS].astype(float)
    # First hour is used without its labels. Never fit normalization to a full event.
    if normalized:
        ref = a.iloc[:BURNIN]
        center = ref.median()
        scale = np.maximum(
            np.maximum(
                (ref.quantile(0.75) - ref.quantile(0.25)).to_numpy(),
                0.01 * center.abs().to_numpy(),
            ),
            FLOORS,
        )
        a = (a - center) / scale
    r = a.rolling(WINDOW, min_periods=30)
    out = {}
    for name, z in [
        ("mean", r.mean()),
        ("std", r.std()),
        ("min", r.min()),
        ("max", r.max()),
        ("last", a),
        ("change", a - a.shift(WINDOW - 1)),
        ("missing", a.isna().rolling(WINDOW, min_periods=1).mean()),
    ]:
        for c in SENSORS:
            out[c + "_" + name] = z[c]
    return pd.DataFrame(out).iloc[BURNIN::STEP].replace([np.inf, -np.inf], np.nan)


def model(kind):
    m = (
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=3,
        )
        if kind == "RF"
        else HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.1,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=SEED,
        )
    )
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        m,
    )
