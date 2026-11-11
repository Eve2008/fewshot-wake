"""Evaluate every calibration rule on the saved per-threshold event counts.

Rules (each picks one threshold per wake from common.GRID):
  deployed            NexaRoom's calibrate_wake on 20 s of room audio (snapped down to the grid)
  bank60/30/15        empirical: the largest threshold before the bank's false-alarm rate first
                      exceeds the target (60, 30 or 15 min of the shipped bank)
  bank60+noisy        the same on the bank plus its noisy copy (2 h)
  bank60-os           order-statistic rule: allow at most ceil(target x hours) - 1 bank events, so the expected
                      rate at the chosen threshold equals the target when target x hours >= 1
  bank60-tail         exponential fit of the bank's event-rate tail (2..50 events), solved for the target
  global-bank         one threshold for every wake: chosen on the bank, averaged over the OTHER wakes of
                      the same seed and k (leave-one-wake-out), so it is a fixed shipped constant
  oracle-global       one threshold for all wakes chosen on the evaluation negatives (the pilot's metric;
                      an optimistic reference, not deployable)
  oracle-wake         each wake's own threshold chosen on the evaluation negatives (upper bound)

Writes, per experiment: per_task.csv (every task x method x rule x target x condition),
calibration.csv (means with 95 % cluster-bootstrap intervals), det.csv (pooled DET curves).

Usage: python experiments/summarize.py [mswc|samespk|esc50|esc50_samesrc|fleurs|condnorm|mswc_x|mswc_ssl|mswc_ssl_lc|samespk_v2|samespk_ssl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

G = common.GRID
TARGETS_FAH = (0.5, 1.0, 2.0)
ESC_TARGETS_FAH = (0.5, 1.0, 2.0, 5.0, 10.0)   # sound wakes: also looser operating points
TAIL_SENSITIVITY = ((1, 20), (3, 30), (2, 100), (5, 200))
BOOT = 1000

EXPERIMENTS = {
    "mswc": {
        "dir": common.RESULTS / "mswc",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")], "bank60+noisy": [("bank_clean", None), ("bank_noisy", None)]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"]), "noisy": ("noisy", ["eval_noisy"])},
        "extra": {},
    },
    "samespk": {
        "dir": common.RESULTS / "samespk",
        "key": lambda r: (r["set"], r["shots"], r["index"]),
        "cluster": lambda r: r["speaker"],
        "group": lambda r: (r["set"], r["shots"]),
        "pool": lambda r: (r["set"], r["shots"]),
        "pool_names": ["set", "shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")], "bank60+noisy": [("bank_clean", None), ("bank_noisy", None)]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"]), "noisy": ("noisy", ["eval_noisy"])},
        "extra": {"clean": ["own_clean"]},
    },
    "condnorm": {
        "dir": common.RESULTS / "condnorm",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")], "bank60+noisy": [("bank_clean", None), ("bank_noisy", None)]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"]), "noisy": ("noisy", ["eval_noisy"])},
        "extra": {},
    },
    "fleurs": {
        "dir": common.RESULTS / "fleurs",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["fleurs"])},
        "extra": {"clean": ["eval_clean"]},
    },
    # Added after review: the variants of methods_extra.py on the cross-speaker wakes (results/mswc_x),
    # the self-supervised baseline (results/ssl, clean condition only) and the same-speaker wakes with the
    # speaker's own clips masked out of the seed-1 stream plus the new variants (results/samespk_v2).
    "mswc_x": {
        "dir": common.RESULTS / "mswc_x",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")], "bank60+noisy": [("bank_clean", None), ("bank_noisy", None)]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"]), "noisy": ("noisy", ["eval_noisy"])},
        "extra": {},
    },
    "mswc_ssl": {
        "dir": common.RESULTS / "ssl",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"])},
        "extra": {},
    },
    "mswc_ssl_lc": {
        "dir": common.RESULTS / "ssl_lc",
        "key": lambda r: (r["seed"], r["shots"], r["word"]),
        "cluster": lambda r: r["word"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"])},
        "extra": {},
    },
    "samespk_v2": {
        "dir": common.RESULTS / "samespk_v2",
        "key": lambda r: (r["set"], r["shots"], r["index"]),
        "cluster": lambda r: r["speaker"],
        "group": lambda r: (r["set"], r["shots"]),
        "pool": lambda r: (r["set"], r["shots"]),
        "pool_names": ["set", "shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")], "bank60+noisy": [("bank_clean", None), ("bank_noisy", None)]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"])},
        "extra": {"clean": ["own_clean"]},
    },
    "samespk_ssl": {
        "dir": common.RESULTS / "ssl_samespk",
        "key": lambda r: (r["set"], r["shots"], r["index"]),
        "cluster": lambda r: r["speaker"],
        "group": lambda r: (r["set"], r["shots"]),
        "pool": lambda r: (r["set"], r["shots"]),
        "pool_names": ["set", "shots"],
        "banks": {"bank60": [("bank_clean", None)], "bank30": [("bank_clean", "30min")],
                  "bank15": [("bank_clean", "15min")]},
        "tail_bank": "bank_clean",
        "conditions": {"clean": ("clean", ["eval_clean"])},
        "extra": {"clean": ["own_clean"]},
    },
    "esc50": {
        "dir": common.RESULTS / "esc50",
        "key": lambda r: (r["seed"], r["shots"], r["category"]),
        "cluster": lambda r: r["category"],
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"speechbank": [("speech_bank", None)], "speech+soundbank": [("speech_bank", None), ("sound_bank", None)]},
        "tail_bank": "speech_bank",
        "conditions": {"clean": ("clean", ["sound_eval", "speech_eval"])},
        "extra": {"clean": ["sound_eval", "speech_eval"]},
        "targets": ESC_TARGETS_FAH,
    },
    "esc50_samesrc": {
        "dir": common.RESULTS / "esc50_samesrc",
        "key": lambda r: (r["src_file"], r["repeat"]),
        "cluster": lambda r: f"{r['category']}/{r['src_file']}",
        "group": lambda r: (r["seed"], r["shots"]),
        "pool": lambda r: (r["shots"],),
        "pool_names": ["shots"],
        "banks": {"speechbank": [("speech_bank", None)], "speech+soundbank": [("speech_bank", None), ("sound_bank", None)]},
        "tail_bank": "speech_bank",
        "conditions": {"clean": ("clean", ["sound_eval", "speech_eval"])},
        "extra": {"clean": ["sound_eval", "speech_eval"]},
        "targets": ESC_TARGETS_FAH,
    },
}


def first_crossing(rate: np.ndarray, target: float) -> int:
    """Index of the largest threshold before the rate first exceeds the target."""

    over = np.nonzero(rate > target + 1e-12)[0]
    if over.size == 0:
        return len(rate) - 1
    return max(0, int(over[0]) - 1)


def tail_index(events: np.ndarray, hours: float, target: float, lo: int = 2, hi: int = 50) -> tuple[int, bool]:
    over = np.nonzero(events > hi)[0]
    stop = int(over[0]) if over.size else len(events)
    sel = np.nonzero((events[:stop] >= lo))[0]
    if np.unique(events[sel]).size < 3:
        return first_crossing(events / hours, target), False
    b, a = np.polyfit(G[sel], np.log(events[sel]), 1)
    if b <= 0:
        return first_crossing(events / hours, target), False
    tau = (np.log(target * hours) - a) / b
    return int(np.clip(np.searchsorted(G, tau, side="right") - 1, 0, len(G) - 1)), True


def order_statistic_index(events: np.ndarray, hours: float, target: float) -> int:
    allowed = max(0, int(np.ceil(target * hours - 1e-3)) - 1)   # hours is 1.00004 for the 60-min bank
    return first_crossing(events, allowed)


def load(name: str) -> list[dict]:
    cfg = EXPERIMENTS[name]
    rows = []
    for path in sorted((cfg["dir"] / "tasks").glob("*.json")):
        r = json.loads(path.read_text())
        for method, entry in r.get("methods", {}).items():
            rows.append({"task": r, "method": method, "entry": entry})
    return rows


def curve(item: dict, streams: list[tuple[str, str | None]]) -> np.ndarray:
    r, entry = item["task"], item["entry"]
    events, hours = np.zeros(len(G)), 0.0
    for stream, prefix in streams:
        if prefix is None:
            events = events + np.asarray(entry["neg"][stream], dtype=float)
            hours += r["negative_hours"][stream]
        else:
            events = events + np.asarray(entry["prefix"][stream][prefix], dtype=float)
            hours += r["prefix_hours"][stream][prefix]
    return events / hours


def evaluate(name: str) -> pd.DataFrame:
    cfg = EXPERIMENTS[name]
    items = load(name)
    if not items:
        return pd.DataFrame()
    # Pre-compute bank curves for the leave-one-wake-out global threshold.
    bank_curves = {}
    for item in items:
        group = (item["method"],) + tuple(cfg["group"](item["task"]))
        bank_curves.setdefault(group, []).append((cfg["key"](item["task"]), curve(item, cfg["banks"][next(iter(cfg["banks"]))])))
    # Pooled evaluation curves for the oracle-global threshold.
    pooled = {}
    for item in items:
        for cond, (_pos, streams) in cfg["conditions"].items():
            if any(st not in item["entry"]["neg"] for st in streams):
                continue
            key = (item["method"], cond) + tuple(cfg["pool"](item["task"]))
            pooled.setdefault(key, []).append(curve(item, [(s, None) for s in streams]))
    targets = cfg.get("targets", TARGETS_FAH)
    pooled_index = {key: {t: first_crossing(np.mean(curves, axis=0), t) for t in targets}
                    for key, curves in pooled.items()}

    out = []
    for item in items:
        r, entry, method = item["task"], item["entry"], item["method"]
        key = cfg["key"](r)
        group = (method,) + tuple(cfg["group"](r))
        others = [c for k, c in bank_curves[group] if k != key]
        tail_events = np.asarray(entry["neg"][cfg["tail_bank"]], dtype=float)
        tail_hours = r["negative_hours"][cfg["tail_bank"]]
        base = {"experiment": name, "method": method, "cluster": cfg["cluster"](r),
                **{n: v for n, v in zip(cfg["pool_names"], cfg["pool"](r))}}
        base.update({k: r[k] for k in ("seed", "word", "category", "speaker", "index", "src_file", "repeat") if k in r})
        for target in targets:
            rules = {}
            if entry.get("threshold"):
                rules["deployed"] = int(max(0, np.searchsorted(G, entry["threshold"], side="right") - 1))
            for bank_name, streams in cfg["banks"].items():
                try:
                    rules[bank_name] = first_crossing(curve(item, streams), target)
                except KeyError:
                    pass
            first_bank = next(iter(cfg["banks"]))
            rules[f"{first_bank}-os"] = order_statistic_index(tail_events, tail_hours, target)
            rules[f"{first_bank}-tail"], fitted = tail_index(tail_events, tail_hours, target)
            for bank_name, streams in cfg["banks"].items():
                if bank_name == first_bank:
                    continue
                try:
                    hours = sum(r["negative_hours"][st] if p is None else r["prefix_hours"][st][p] for st, p in streams)
                    rate = curve(item, streams)
                except KeyError:
                    continue
                rules[f"{bank_name}-tail"], _ = tail_index(rate * hours, hours, target)
            for lo, hi in TAIL_SENSITIVITY:
                rules[f"{first_bank}-tail[{lo},{hi}]"], _ = tail_index(tail_events, tail_hours, target, lo, hi)
            if others:
                rules["global-bank"] = first_crossing(np.mean(others, axis=0), target)
            for cond, (pos_name, streams) in cfg["conditions"].items():
                if any(st not in entry["neg"] for st in streams):
                    continue
                pos = np.asarray(entry["pos"][pos_name])
                eval_curve = curve(item, [(s, None) for s in streams])
                cond_rules = dict(rules)
                cond_rules["oracle-wake"] = first_crossing(eval_curve, target)
                cond_rules["oracle-global"] = pooled_index[(method, cond) + tuple(cfg["pool"](r))][target]
                for rule, idx in cond_rules.items():
                    row = {**base, "condition": cond, "rule": rule, "target_fah": target, "tau": float(G[idx]),
                           "frr": float(np.mean(pos > G[idx])), "fah": float(eval_curve[idx]),
                           "positives": int(pos.size), "tail_fitted": bool(fitted) if rule.endswith("-tail") else None}
                    for extra in cfg["extra"].get(cond, []):
                        row[f"fah_{extra}"] = float(curve(item, [(extra, None)])[idx])
                        row[f"hours_{extra}"] = float(r["negative_hours"][extra])
                    out.append(row)
    return pd.DataFrame(out)


def cluster_bootstrap(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> tuple[float, float]:
    clusters = frame.groupby("cluster")[column].agg(["sum", "count"])
    sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
    n = len(sums)
    picks = rng.integers(0, n, size=(BOOT, n))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(per_task: pd.DataFrame, pool_names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    keys = pool_names + ["method", "condition", "rule", "target_fah"]
    for values, frame in per_task.groupby(keys):
        target = values[-1]
        fah = frame.fah.to_numpy()
        row = dict(zip(keys, values))
        lo, hi = cluster_bootstrap(frame, "frr", rng)
        flo, fhi = cluster_bootstrap(frame, "fah", rng)
        row.update({"wakes": len(frame), "clusters": frame.cluster.nunique(),
                    "frr": frame.frr.mean(), "frr_lo": lo, "frr_hi": hi,
                    "fah_mean": fah.mean(), "fah_lo": flo, "fah_hi": fhi, "fah_median": float(np.median(fah)),
                    "fah_p90": float(np.percentile(fah, 90)), "fah_max": float(fah.max()),
                    "share_within_target": float(np.mean(fah <= target + 1e-9)),
                    "share_within_2x": float(np.mean(fah <= 2 * target + 1e-9)),
                    "tau_median": float(frame.tau.median())})
        for col in [c for c in frame.columns if c.startswith("fah_") and c not in row]:
            hours = frame[col.replace("fah_", "hours_")]
            row[col + "_mean"] = frame[col].mean()
            row[col + "_pooled"] = float((frame[col] * hours).sum() / hours.sum()) if hours.sum() > 0 else np.nan
            row[col.replace("fah_", "hours_") + "_total"] = float(hours.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def det(name: str) -> pd.DataFrame:
    cfg = EXPERIMENTS[name]
    items = load(name)
    groups = {}
    for item in items:
        for cond, (pos_name, streams) in cfg["conditions"].items():
            if any(st not in item["entry"]["neg"] for st in streams):
                continue
            key = tuple(cfg["pool"](item["task"])) + (item["method"], cond)
            pos = np.asarray(item["entry"]["pos"][pos_name])
            frr = (pos[None, :] > G[:, None]).mean(axis=1)
            groups.setdefault(key, []).append((frr, curve(item, [(s, None) for s in streams])))
    rows = []
    for key, values in groups.items():
        frr = np.mean([v[0] for v in values], axis=0)
        fah = np.mean([v[1] for v in values], axis=0)
        for tau, f, a in zip(G, frr, fah):
            rows.append(dict(zip(cfg["pool_names"] + ["method", "condition"], key)) | {"tau": tau, "frr": f, "fah": a})
    return pd.DataFrame(rows)


def main() -> None:
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        cfg = EXPERIMENTS[name]
        per_task = evaluate(name)
        if per_task.empty:
            print(f"{name}: no results")
            continue
        per_task.to_csv(cfg["dir"] / "per_task.csv", index=False)
        table = aggregate(per_task, cfg["pool_names"])
        table.to_csv(cfg["dir"] / "calibration.csv", index=False)
        det(name).to_csv(cfg["dir"] / "det.csv", index=False)
        print(f"{name}: {per_task.shape[0]} rows, {table.shape[0]} aggregate rows")


if __name__ == "__main__":
    main()
