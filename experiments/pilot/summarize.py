"""Summarise MSWC wake runs: DET points, FRR at fixed false alarms per hour,
and the operating point each wake's own calibration chose.

Usage: python experiments/summarize.py results/mswc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("seed*_k*/results.json")):
        data = json.loads(path.read_text())
        grid = np.asarray(data["grid"])
        for result in data["results"]:
            for method, entry in result.get("methods", {}).items():
                for condition in ("clean", "noisy"):
                    rows.append({
                        "seed": result["seed"], "shots": result["shots"], "word": result["word"],
                        "method": method, "condition": condition, "grid": grid,
                        "pos": np.asarray(entry[f"pos_min_{condition}"]),
                        "neg": np.asarray(entry[f"neg_events_{condition}"], dtype=float),
                        "hours": result["negative_hours"][condition], "threshold": entry["threshold"],
                    })
    return rows


def det(group: list[dict]) -> pd.DataFrame:
    grid = group[0]["grid"]
    frr = np.asarray([[np.mean(item["pos"] > tau) for tau in grid] for item in group])
    fah = np.asarray([item["neg"] / item["hours"] for item in group])
    return pd.DataFrame({"tau": grid, "frr": frr.mean(axis=0), "fa_per_hour": fah.mean(axis=0)})


def frr_at(table: pd.DataFrame, max_fah: float) -> tuple[float, float]:
    ok = table[table.fa_per_hour <= max_fah]
    if ok.empty:
        return float("nan"), float("nan")
    row = ok.sort_values("frr").iloc[0]
    return float(row.frr), float(row.tau)


def calibrated(group: list[dict]) -> tuple[float, float]:
    frr, fah = [], []
    for item in group:
        if item["threshold"] is None:
            return float("nan"), float("nan")
        index = int(np.searchsorted(item["grid"], item["threshold"], side="right") - 1)
        index = max(0, index)
        frr.append(np.mean(item["pos"] > item["threshold"]))
        fah.append(item["neg"][index] / item["hours"])
    return float(np.mean(frr)), float(np.mean(fah))


def main() -> None:
    root = Path(sys.argv[1])
    rows = load(root)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("grid", "pos", "neg")} for r in rows])
    out = []
    dets = []
    for (shots, method, condition), index in frame.groupby(["shots", "method", "condition"]).groups.items():
        group = [rows[i] for i in index]
        table = det(group)
        table.insert(0, "condition", condition)
        table.insert(0, "method", method)
        table.insert(0, "shots", shots)
        dets.append(table)
        f1, t1 = frr_at(table, 1.0)
        f05, _ = frr_at(table, 0.5)
        f5, _ = frr_at(table, 5.0)
        cf, ca = calibrated(group)
        out.append({"shots": shots, "method": method, "condition": condition, "targets": len(group),
                    "positives": int(sum(len(g["pos"]) for g in group)),
                    "neg_hours": round(float(np.mean([g["hours"] for g in group])), 3),
                    "FRR@0.5FA/h": f05, "FRR@1FA/h": f1, "FRR@5FA/h": f5, "tau@1FA/h": t1,
                    "calib_FRR": cf, "calib_FA/h": ca})
    summary = pd.DataFrame(out).sort_values(["condition", "shots", "FRR@1FA/h"])
    summary.to_csv(root / "summary.csv", index=False)
    pd.concat(dets).to_csv(root / "det.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
