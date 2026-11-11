"""How much does the event-counting rule differ from the deployed decision layer?

The benchmark counts a false alarm at every frame whose wake score is at or below the threshold, at most one
per 1.2 s, on a matcher that is never reset. The deployed WakeDetector (nexawake/wake.py, unchanged) instead
keeps a candidate while the score ratio falls, fires after two frames without improvement (or when the score
rises above the threshold), then resets the DTW matcher and pauses for 1.2 s.

For the 16 wakes of seed 1 at k = 6, deployed detector ('nexa') and clean templates ('nexa-noaug'), this script
runs the deployed WakeDetector frame by frame over the clean 4-h evaluation stream and over the positive stream,
at the threshold chosen by the bank tail fit for 1 FA/h, and compares FA/h and FRR with the counting rule on the
same thresholds.

Output: results/detector_check/<method>_<rule>_w<idx>.json and results/detector_check/summary.csv
Usage: python experiments/run_detector_check.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from summarize import tail_index  # noqa: E402

SEED, SHOTS = 1, 6
OUT = common.RESULTS / "detector_check"
JOBS = [("nexa", "bank60-tail"), ("nexa-noaug", "bank60-tail")]


def run(args) -> dict:
    method, rule, index = args
    from methods import enroll_clips
    from nexawake.enroll import build_wake
    from nexawake.wake import WakeDetector, WakeSpec

    started = time.time()
    plan = common.load_json(run_mswc.OUT / f"plan_seed{SEED}.json")[index]
    task = common.load_json(run_mswc.OUT / "tasks" / f"seed{SEED}_k{SHOTS}_w{index:02d}.json")
    entry = task["methods"][method]
    folder = run_mswc.stream_dir(SEED)
    room = np.load(folder / "room_feat.npy")
    enrollments, _ = enroll_clips([common.read(l) for l in plan["enroll"][:SHOTS]],
                                  np.random.default_rng(SEED * 100 + index))
    built = build_wake(enrollments, [room], augment=(method == "nexa"))
    if rule == "deployed":
        idx = int(max(0, np.searchsorted(common.GRID, entry["threshold"], side="right") - 1))
    else:
        idx, _fitted = tail_index(np.asarray(entry["neg"]["bank_clean"], float), task["negative_hours"]["bank_clean"], 1.0)
    tau = float(common.GRID[idx])
    spec = WakeSpec("w", "w", list(built.spec.templates), tau, None, None)

    def hits(emb) -> list[int]:
        det = WakeDetector([spec], built.normalizer)
        out = []
        for frame in range(emb.shape[0]):
            hit = det.process(emb[frame])
            if hit is not None:
                out.append(int(hit.frame))       # the candidate frame the deployed hit reports
        return out

    neg = np.load(folder / "neg_clean_feat.npy")
    neg_hits = hits(neg)
    rng_pos = np.random.default_rng(SEED * 100 + index + 50)
    pos, spans = common.stream([common.read(l) for l in plan["positives"]], rng_pos, common.POS_PAD_SECONDS)
    emb, ends = common.models().clip_features(common.to_pcm16(pos).tobytes())
    pos_hits = hits(emb)
    hit_ends = ends[pos_hits] if pos_hits else np.zeros(0, dtype=np.int64)
    detected = [bool(np.any((hit_ends >= a) & (hit_ends <= b + int(0.5 * common.SR)))) for a, b in spans]
    hours = task["negative_hours"]["eval_clean"]
    count_pos = np.asarray(entry["pos"]["clean"])
    return {"method": method, "rule": rule, "index": index, "word": plan["word"], "tau": tau,
            "detector_fa": len(neg_hits), "detector_fah": len(neg_hits) / hours,
            "counting_fa": int(entry["neg"]["eval_clean"][idx]), "counting_fah": entry["neg"]["eval_clean"][idx] / hours,
            "detector_frr": 1.0 - float(np.mean(detected)), "counting_frr": float(np.mean(count_pos > tau)),
            "positive_hits": len(pos_hits), "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    work = []
    for method, rule in JOBS:
        for index in range(len(common.TARGETS)):
            path = OUT / f"{method}_{rule}_w{index:02d}.json"
            if not path.is_file():
                work.append((path, (method, rule, index)))
    print(f"{len(work)} runs", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run, a): p for p, a in work}
        for future in as_completed(futures):
            r = future.result()
            futures[future].write_text(json.dumps(r, ensure_ascii=False))
            print(f"{r['method']} {r['rule']} {r['word']}: FA {r['detector_fa']} vs {r['counting_fa']}, "
                  f"FRR {r['detector_frr']:.3f} vs {r['counting_frr']:.3f} ({r['seconds']}s)", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*_w*.json"))]
    table = pd.DataFrame(rows)
    summary = table.groupby(["method", "rule"]).agg(
        wakes=("index", "count"), detector_fah=("detector_fah", "mean"), counting_fah=("counting_fah", "mean"),
        detector_frr=("detector_frr", "mean"), counting_frr=("counting_frr", "mean"),
        detector_fa_total=("detector_fa", "sum"), counting_fa_total=("counting_fa", "sum")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
