"""Check pinned raw files, derived features, and saved-model predictions."""

import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import features
from scripts.predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "raw")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    paths = {}
    for file in manifest["files"]:
        path = args.data / file["path"]
        content = path.read_bytes()
        assert (
            hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            == file["sha"]
        )
        paths[path.name] = path
    meta = pd.read_csv(ROOT / "data/development.csv")
    matrix = np.load(ROOT / "data/development_features.npz")
    for name, group in meta.groupby("file", sort=False):
        frame = pd.read_parquet(paths[name])
        for key, norm in [("raw", False), ("calibrated", True)]:
            values = features(frame, norm)
            values.index = np.arange(3600, len(frame), 15)
            np.testing.assert_allclose(
                values.loc[group.second].to_numpy(),
                matrix[key][group.index],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
    hold = pd.read_csv(ROOT / "data/holdout_predictions.csv")
    count = 0
    for name, group in hold.groupby("file", sort=False):
        for model in [
            "raw_RF",
            "calibrated_RF",
            "calibrated_HGB",
            "original_calibrated_RF",
            "original_calibrated_HGB",
        ]:
            output = predict(paths[name], model, "window_q99").set_index("second")
            np.testing.assert_allclose(
                output.loc[group.second, "score"].to_numpy(),
                group[model].to_numpy(),
                rtol=0,
                atol=1e-12,
            )
            count += 1
    print(
        f"PASS: {len(paths)} raw files, 45 feature records, {count} model/record predictions."
    )


if __name__ == "__main__":
    main()
