"""Event-level alarm metrics and threshold selection."""

import numpy as np

GRID = np.r_[np.arange(10, 100) / 100, [0.995, 0.999, 0.9999, 1.000000001]]


def evaluate(meta, p, th):
    out = []
    for file, g in meta.assign(score=p).groupby("file", sort=False):
        hit = g.score.to_numpy() >= th
        s = g.second.to_numpy()
        confirm = np.zeros(len(g), bool)
        confirm[2:] = hit[2:] & hit[1:-1] & hit[:-2] & (s[2:] - s[:-2] == 30)
        normal = g.label.to_numpy() == 0
        false = confirm & normal
        starts = false & ~np.r_[False, false[:-1]]
        on = float(g.onset.iloc[0])
        d = s[confirm & (s >= on)]
        delay = float(d[0] - on) if len(d) else None
        is_event = bool(np.isfinite(on))
        out.append(
            dict(
                file=file,
                well=g.well.iloc[0],
                is_event=is_event,
                normal_windows=int(normal.sum()),
                false_windows=int(false.sum()),
                normal_hours=float(normal.sum() * 15 / 3600),
                false_episodes=int(starts.sum()),
                delay_seconds=delay,
                detected_300=is_event and delay is not None and delay <= 300,
                clean_300=bool(
                    is_event
                    and delay is not None
                    and delay <= 300
                    and starts.sum() == 0
                ),
            )
        )
    return out


def aggregate(rows):
    h = sum(r["normal_hours"] for r in rows)
    normal = sum(r["normal_windows"] for r in rows)
    fa = sum(r["false_episodes"] for r in rows)
    return dict(
        events=sum(r["is_event"] for r in rows),
        detected_300=sum(r["detected_300"] for r in rows),
        clean_300=sum(r["clean_300"] for r in rows),
        false_episodes=fa,
        normal_hours=h,
        false_episodes_per_day=fa * 24 / h if h else None,
        alarm_occupancy=(
            sum(r["false_windows"] for r in rows) / normal if normal else None
        ),
    )


def choose(meta, p):
    table = []
    for th in GRID:
        table.append(dict(threshold=float(th), **aggregate(evaluate(meta, p, th))))
    ans = {
        "fixed_05": 0.5,
        "window_q99": float(
            np.quantile(p[meta.label.to_numpy() == 0], 0.99, method="higher")
        )
        + 1e-12,
    }
    for budget in [1, 2, 6]:
        feasible = [
            v
            for v in table
            if v["alarm_occupancy"] <= 0.01 and v["false_episodes_per_day"] <= budget
        ]
        assert feasible
        best = max(
            feasible,
            key=lambda v: (
                v["clean_300"],
                v["detected_300"],
                -v["false_episodes"],
                v["threshold"],
            ),
        )
        ans[f"events_{budget}_day"] = best["threshold"]
    return ans, table
