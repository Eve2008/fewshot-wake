"""Noise diagnostic, version 2 (after review).

diagnose_noise.py scored the 16 wakes of seed 1, k = 6, on 10 min of synthetic noise at the empirical bank
threshold. This version
  * covers all 144 cross-speaker wakes (3 seeds x k = 3, 6, 10),
  * reports both the empirical bank rule and the bank tail fit (target 1 FA/h),
  * adds the embedding-only variants with noisy and with clean templates, to see whether the burst of false
    alarms on noise needs the 32 log-mel dimensions,
  * adds recorded environmental sound as a second noise type: the 1.75 h ESC-50 evaluation stream (folds 3-5,
    1,200 clips back to back at their recorded level, results/esc50/streams/eval_feat.npy), which the speech
    wakes never saw.
The synthetic noise is generated exactly as in diagnose_noise.py (same recipe as the template augmentation).

Output: results/noise_v2/per_wake.csv, results/noise_v2/summary.csv
Usage: python experiments/diagnose_noise_v2.py [--jobs 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from diagnose_noise import noise_stream  # noqa: E402
from summarize import first_crossing, tail_index  # noqa: E402

OUT = common.RESULTS / "noise_v2"
_NOISE = None


def noise_features():
    global _NOISE
    if _NOISE is None:
        emb, _ = common.models().clip_features(noise_stream().tobytes())
        sound = np.load(common.RESULTS / "esc50" / "streams" / "eval_feat.npy")
        _NOISE = {"synthetic": emb, "esc50": sound}
    return _NOISE


def run(args) -> list[dict]:
    seed, shots, index = args
    from methods import enroll_clips
    from nexawake import fast
    from nexawake.enroll import build_wake

    plan = common.load_json(run_mswc.OUT / f"plan_seed{seed}.json")[index]
    name = f"seed{seed}_k{shots}_w{index:02d}.json"
    task = common.load_json(run_mswc.OUT / "tasks" / name)
    extra = common.load_json(common.RESULTS / "mswc_x" / "tasks" / name)
    room = np.load(run_mswc.stream_dir(seed) / "room_feat.npy")
    enrollments, _ = enroll_clips([common.read(l) for l in plan["enroll"][:shots]],
                                  np.random.default_rng(seed * 100 + index))
    full = build_wake(enrollments, [room], augment=True)
    clean = build_wake(enrollments, [room], augment=False)
    streams = noise_features()
    variants = {}
    if full is not None:
        variants["nexa"] = (lambda x: fast.wake_score(fast.template_scores(x, full.spec.templates, full.normalizer)),
                            task["methods"].get("nexa"))
        cols = slice(0, 96)
        norm = common.fit_normalizer([np.asarray(e.variants[0][0])[:, cols] for e in enrollments] + [room[:, cols]])
        t_aug = [np.asarray(t)[:, cols] for t in full.spec.templates]
        variants["emb-only"] = (lambda x, t=t_aug, n=norm: fast.wake_score(fast.template_scores(np.asarray(x)[:, cols], t, n)),
                                task["methods"].get("emb-only"))
    if clean is not None:
        variants["nexa-noaug"] = (lambda x: fast.wake_score(fast.template_scores(x, clean.spec.templates, clean.normalizer)),
                                  task["methods"].get("nexa-noaug"))
        cols = slice(0, 96)
        norm = common.fit_normalizer([np.asarray(e.variants[0][0])[:, cols] for e in enrollments] + [room[:, cols]])
        t_clean = [np.asarray(t)[:, cols] for t in clean.spec.templates]
        variants["emb-only-noaug"] = (lambda x, t=t_clean, n=norm: fast.wake_score(fast.template_scores(np.asarray(x)[:, cols], t, n)),
                                      extra["methods"].get("emb-only-noaug"))
    rows = []
    for method, (fn, entry) in variants.items():
        if entry is None:
            continue
        bank = np.asarray(entry["neg"]["bank_clean"], dtype=float)
        hours_bank = task["negative_hours"]["bank_clean"]
        rules = {"bank60": first_crossing(bank / hours_bank, 1.0), "bank60-tail": tail_index(bank, hours_bank, 1.0)[0]}
        for stream, feats in streams.items():
            scores = fn(feats)
            ev = common.events(scores)
            hours = feats.shape[0] * common.FRAME_SECONDS / 3600
            for rule, idx in rules.items():
                rows.append({"seed": seed, "shots": shots, "word": plan["word"], "method": method, "stream": stream,
                             "rule": rule, "tau": float(common.GRID[idx]), "fah": float(ev[idx] / hours),
                             "hours": hours,
                             "eval_clean_fah": entry["neg"]["eval_clean"][idx] / task["negative_hours"]["eval_clean"],
                             "eval_noisy_fah": entry["neg"]["eval_noisy"][idx] / task["negative_hours"]["eval_noisy"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    work = [(s, k, i) for s in (1, 2, 3) for k in (3, 6, 10) for i in range(len(common.TARGETS))]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [r for part in pool.map(run, work) for r in part]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "per_wake.csv", index=False)
    summary = table.groupby(["stream", "method", "rule", "shots"]).agg(
        wakes=("fah", "size"), fah_mean=("fah", "mean"), fah_median=("fah", "median"), fah_max=("fah", "max"),
        share_zero=("fah", lambda x: float(np.mean(x == 0))), eval_noisy_fah=("eval_noisy_fah", "mean"),
        eval_clean_fah=("eval_clean_fah", "mean")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))
    common.dump_json(OUT / "info.json", {"synthetic_hours": float(table[table.stream == "synthetic"].hours.iloc[0]),
                                         "esc50_hours": float(table[table.stream == "esc50"].hours.iloc[0])})


if __name__ == "__main__":
    main()
