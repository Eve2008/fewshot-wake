"""Paired differences between threshold rules (same wakes), with 95 % cluster-bootstrap intervals.

Usage: python experiments/paired.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PAIRS = [("bank60-tail", "global-bank"), ("bank60-tail", "deployed"), ("bank60-tail", "oracle-wake"),
         ("bank60-tail", "bank60"), ("bank60-os", "global-bank"), ("bank30-tail", "bank60-tail")]
KEYS = {"mswc": ["seed", "word"], "samespk": ["index"], "esc50": ["seed", "category"]}
POOL = {"mswc": ["shots"], "samespk": ["set", "shots"], "esc50": ["shots"]}
ESC_PAIRS = [("speechbank-tail", "global-bank"), ("speech+soundbank", "speechbank-tail"),
             ("speechbank-tail", "deployed"), ("speechbank-tail", "oracle-wake")]


def boot(diff: pd.Series, clusters: pd.Series, rng) -> tuple[float, float]:
    frame = pd.DataFrame({"d": diff.to_numpy(), "c": clusters.to_numpy()})
    g = frame.groupby("c").d.agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    picks = rng.integers(0, len(s), size=(2000, len(s)))
    means = s[picks].sum(1) / n[picks].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    rng = np.random.default_rng(1)
    for name in ("mswc", "samespk", "esc50"):
        path = common.RESULTS / name / "per_task.csv"
        if not path.is_file():
            continue
        per = pd.read_csv(path)
        rows = []
        for pool_values, frame in per.groupby(POOL[name] + ["method", "condition", "target_fah"]):
            for a, b in (ESC_PAIRS if name == "esc50" else PAIRS):
                x = frame[frame.rule == a].set_index(KEYS[name])
                y = frame[frame.rule == b].set_index(KEYS[name])
                common_idx = x.index.intersection(y.index)
                if len(common_idx) < 3:
                    continue
                x, y = x.loc[common_idx], y.loc[common_idx]
                row = dict(zip(POOL[name] + ["method", "condition", "target_fah"], pool_values))
                row.update({"rule_a": a, "rule_b": b, "wakes": len(common_idx)})
                for col in ("frr", "fah"):
                    d = x[col] - y[col]
                    lo, hi = boot(d, x["cluster"], rng)
                    row[f"d_{col}"] = float(d.mean())
                    row[f"d_{col}_lo"], row[f"d_{col}_hi"] = lo, hi
                row["max_fah_a"], row["max_fah_b"] = float(x.fah.max()), float(y.fah.max())
                rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(common.RESULTS / name / "paired.csv", index=False)
        print(name, len(out))


if __name__ == "__main__":
    main()
