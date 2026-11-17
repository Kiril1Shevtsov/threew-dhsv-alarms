"""Recalculate event counts from published prediction tables; no model fitting."""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alarms import evaluate, aggregate


def main():
    dev = pd.read_csv(ROOT / "data/development.csv")
    hold = pd.read_csv(ROOT / "data/holdout_predictions.csv")
    assert set(dev.well).isdisjoint(hold.well)
    for s in json.loads((ROOT / "data/splits.json").read_text()):
        assert set(s["train_wells"]).isdisjoint(s["validation_wells"])
        assert s["outer_well"] not in s["train_wells"] + s["validation_wells"]
    saved = pd.read_csv(ROOT / "results/development_events.csv")
    checked = 0
    for t in pd.read_csv(ROOT / "data/thresholds.csv").itertuples():
        g = dev[dev.well == t.outer_well]
        rows = evaluate(g, g[t.model].to_numpy(), t.threshold)
        expected = saved[
            (saved.model == t.model)
            & (saved.policy == t.policy)
            & (saved.well == t.outer_well)
        ].set_index("file")
        for row in rows:
            for k in [
                "false_episodes",
                "false_windows",
                "normal_windows",
                "detected_300",
                "clean_300",
            ]:
                assert row[k] == expected.loc[row["file"], k], (
                    t.model,
                    t.policy,
                    row["file"],
                    k,
                )
            checked += 1
    for record in json.loads((ROOT / "results/development_results.json").read_text())[
        "summary"
    ]:
        actual = aggregate(
            saved[
                (saved.model == record["model"]) & (saved.policy == record["policy"])
            ].to_dict("records")
        )
        for k, v in actual.items():
            assert np.isclose(v, record[k]), (k, v, record[k])
    hold_saved = pd.read_csv(ROOT / "results/holdout_events.csv")
    for p in sorted((ROOT / "models").glob("*_policy.json")):
        policy = json.loads(p.read_text())
        name = policy["model"]
        for key, t in policy["thresholds"].items():
            rows = evaluate(hold, hold[name].to_numpy(), t)
            expected = hold_saved[
                (hold_saved.model == name) & (hold_saved.policy == key)
            ].set_index("file")
            for row in rows:
                for k in [
                    "false_episodes",
                    "false_windows",
                    "normal_windows",
                    "detected_300",
                    "clean_300",
                ]:
                    assert row[k] == expected.loc[row["file"], k], (
                        name,
                        key,
                        row["file"],
                        k,
                    )
                checked += 1
    summary = json.loads((ROOT / "results/holdout_results.json").read_text())["summary"]
    for record in summary:
        subset = hold_saved[
            (hold_saved.model == record["model"])
            & (hold_saved.policy == record["policy"])
        ]
        actual = aggregate(subset.to_dict("records"))
        mapping = {
            "events": "known_other_fault_onsets",
            "detected_300": "other_fault_files_signalled_300",
            "clean_300": "clean_other_fault_files_signalled_300",
        }
        for key, value in actual.items():
            assert np.isclose(value, record[mapping.get(key, key)]), (
                key,
                value,
                record,
            )
        signals = 0
        for _, group in hold.groupby("file", sort=False):
            above = group[record["model"]].to_numpy() >= record["threshold"]
            seconds = group.second.to_numpy()
            alarm = np.zeros(len(group), dtype=bool)
            alarm[2:] = (
                above[2:]
                & above[1:-1]
                & above[:-2]
                & (seconds[2:] - seconds[:-2] == 30)
            )
            signals += int((alarm & (group.label.to_numpy() > 0)).any())
        assert signals == record["other_fault_files_signalled_any"]
    print(
        f"PASS: {checked} file/policy combinations; disjoint well groups; all aggregates."
    )


if __name__ == "__main__":
    main()
