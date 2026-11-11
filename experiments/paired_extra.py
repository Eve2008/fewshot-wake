"""Paired comparisons across enrollment size k and across methods, from per_task.csv.

Enrollment is nested over k (k = 3 is a subset of k = 6, which is a subset of k = 10) and the
positives and negatives of a wake do not change with k, so the k comparison is paired by wake.
Methods are compared on the same wakes (same enrollment recordings, positives and negatives).
Intervals are 95 % cluster-bootstrap intervals (resampling words for MSWC cross-speaker, speakers for
the same-speaker set). Nothing here recomputes scores.

Writes:
  results/mswc/paired_k.csv          d(FRR), d(FA/h) between k values, per method x rule x target x condition
  results/mswc/paired_methods.csv    d(FRR), d(FA/h) between methods, per k x rule x target x condition
  results/samespk/paired_k.csv       main set, speakers whose wake was built at k = 3, 6 and 10
  results/samespk/paired_methods.csv same-speaker sets

Usage: python experiments/paired_extra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

R = common.RESULTS
BOOT = 2000
RULES = ["deployed", "global-bank", "bank60", "bank60-os", "bank60-tail", "oracle-global", "oracle-wake"]
K_PAIRS = [(6, 3), (10, 6), (10, 3)]
METHOD_PAIRS = [("nexa-noaug", "nexa"), ("nexa", "nexa-nonorm"), ("nexa-noaug-bnorm", "nexa-noaug"),
                ("nexa-min", "nexa"), ("nexa-noaug", "emb-only"), ("nexa-noaug", "mel-only"),
                ("nexa-noaug", "prototype"), ("nexa-noaug", "mfcc-dtw"), ("nexa", "mfcc-dtw")]


def boot(diff: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    frame = pd.DataFrame({"d": diff, "c": clusters})
    g = frame.groupby("c").d.agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    picks = rng.integers(0, len(s), size=(BOOT, len(s)))
    means = s[picks].sum(1) / n[picks].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(frame: pd.DataFrame, col: str, a, b, keys: list[str], rng) -> dict | None:
    x = frame[frame[col] == a].set_index(keys)
    y = frame[frame[col] == b].set_index(keys)
    idx = x.index.intersection(y.index)
    if len(idx) < 3:
        return None
    x, y = x.loc[idx], y.loc[idx]
    row = {"a": a, "b": b, "wakes": len(idx), "clusters": int(x.cluster.nunique()),
           "frr_a": float(x.frr.mean()), "frr_b": float(y.frr.mean()),
           "fah_a": float(x.fah.mean()), "fah_b": float(y.fah.mean())}
    for metric in ("frr", "fah"):
        d = (x[metric] - y[metric]).to_numpy()
        lo, hi = boot(d, x.cluster.to_numpy(), rng)
        row[f"d_{metric}"], row[f"d_{metric}_lo"], row[f"d_{metric}_hi"] = float(d.mean()), lo, hi
        row[f"{metric}_a_better"] = int(np.sum(d < 0))
        row[f"{metric}_b_better"] = int(np.sum(d > 0))
    return row


def main() -> None:
    rng = np.random.default_rng(2)
    per = pd.read_csv(R / "mswc" / "per_task.csv")
    per = per[per.rule.isin(RULES)]
    rows = []
    for (method, cond, rule, target), frame in per.groupby(["method", "condition", "rule", "target_fah"]):
        for a, b in K_PAIRS:
            row = compare(frame, "shots", a, b, ["seed", "word"], rng)
            if row:
                rows.append({"method": method, "condition": cond, "rule": rule, "target_fah": target, **row})
    pd.DataFrame(rows).to_csv(R / "mswc" / "paired_k.csv", index=False)
    print("mswc paired_k", len(rows))
    rows = []
    for (k, cond, rule, target), frame in per.groupby(["shots", "condition", "rule", "target_fah"]):
        for a, b in METHOD_PAIRS:
            row = compare(frame, "method", a, b, ["seed", "word"], rng)
            if row:
                rows.append({"shots": k, "condition": cond, "rule": rule, "target_fah": target, **row})
    pd.DataFrame(rows).to_csv(R / "mswc" / "paired_methods.csv", index=False)
    print("mswc paired_methods", len(rows))

    path = R / "samespk" / "per_task.csv"
    if not path.is_file():
        return
    same = pd.read_csv(path)
    same = same[same.rule.isin(RULES)]
    main_set = same[same.set == "main"]
    built = main_set.groupby("index").shots.nunique()
    keep = set(built[built == 3].index)
    main_set = main_set[main_set["index"].isin(keep)]
    rows = []
    for (method, cond, rule, target), frame in main_set.groupby(["method", "condition", "rule", "target_fah"]):
        for a, b in K_PAIRS:
            row = compare(frame, "shots", a, b, ["index"], rng)
            if not row:
                continue
            for name in ("own_clean",):
                col = f"fah_{name}"
                if col in frame:
                    x = frame[frame.shots == a].set_index("index")
                    y = frame[frame.shots == b].set_index("index")
                    idx = x.index.intersection(y.index)
                    hx, hy = x.loc[idx, f"hours_{name}"], y.loc[idx, f"hours_{name}"]
                    row[f"{col}_pooled_a"] = float((x.loc[idx, col] * hx).sum() / hx.sum())
                    row[f"{col}_pooled_b"] = float((y.loc[idx, col] * hy).sum() / hy.sum())
            rows.append({"set": "main", "method": method, "condition": cond, "rule": rule, "target_fah": target, **row})
    pd.DataFrame(rows).to_csv(R / "samespk" / "paired_k.csv", index=False)
    print("samespk paired_k", len(rows), "speakers built at all k:", len(keep))
    rows = []
    for (setname, k, cond, rule, target), frame in same.groupby(["set", "shots", "condition", "rule", "target_fah"]):
        for a, b in METHOD_PAIRS:
            row = compare(frame, "method", a, b, ["index"], rng)
            if row:
                rows.append({"set": setname, "shots": k, "condition": cond, "rule": rule, "target_fah": target, **row})
    pd.DataFrame(rows).to_csv(R / "samespk" / "paired_methods.csv", index=False)
    print("samespk paired_methods", len(rows))


if __name__ == "__main__":
    main()
