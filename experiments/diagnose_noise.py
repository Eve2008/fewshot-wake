"""Do noise-augmented templates fire on noise alone?

For every target word (seed 1, k = 6), enroll as in run_mswc.py and score 10 min of
pure white + pink noise at the level used for the noisy evaluation stream (10 dB
below the speech level). Reports false alarms per hour on noise alone at the
threshold each wake's clean-bank calibration (1 FA/h) chose.

Usage: python experiments/diagnose_noise.py [--jobs 8]
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from summarize import first_crossing  # noqa: E402

SEED, SHOTS, MINUTES = 1, 6, 10


def noise_stream() -> np.ndarray:
    split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(SEED)]
    sample, _ = common.stream([common.read(link) for link in split["negatives"][:600]], np.random.default_rng(3),
                              common.GAP_SECONDS)
    level = common.signal_level(sample)
    size = int(MINUTES * 60 * common.SR)
    noise = common.floor(np.random.default_rng(4), size).astype(np.float64)
    rng = np.random.default_rng(5)
    seg = 60 * common.SR
    for start in range(0, size, seg):
        end = min(size, start + seg)
        noise[start:end] += common.synthetic_noise(end - start, rng) * level / (10 ** (run_mswc.SNR_DB / 20))
    return common.to_pcm16(noise)


def run(index: int) -> list[dict]:
    from methods import enroll_clips
    from nexawake import fast
    from nexawake.enroll import build_wake

    plan = common.load_json(common.RESULTS / "mswc" / f"plan_seed{SEED}.json")[index]
    task = common.load_json(common.RESULTS / "mswc" / "tasks" / f"seed{SEED}_k{SHOTS}_w{index:02d}.json")
    room = np.load(run_mswc.stream_dir(SEED) / "room_feat.npy")
    clips = [common.read(link) for link in plan["enroll"][:SHOTS]]
    enrollments, _ = enroll_clips(clips, np.random.default_rng(SEED * 100 + index))
    emb, _ends = common.models().clip_features(noise_stream().tobytes())
    hours = emb.shape[0] * common.FRAME_SECONDS / 3600
    rows = []
    for name, augment in (("nexa", True), ("nexa-noaug", False)):
        built = build_wake(enrollments, [room], augment=augment)
        scores = fast.wake_score(fast.template_scores(emb, built.spec.templates, built.normalizer))
        ev = common.events(scores)
        entry = task["methods"][name]
        bank = np.asarray(entry["neg"]["bank_clean"]) / task["negative_hours"]["bank_clean"]
        idx = first_crossing(bank, 1.0)
        rows.append({"word": plan["word"], "method": name, "tau_bank1": float(common.GRID[idx]),
                     "noise_only_fah": float(ev[idx] / hours),
                     "eval_noisy_fah": float(entry["neg"]["eval_noisy"][idx] / task["negative_hours"]["eval_noisy"]),
                     "eval_clean_fah": float(entry["neg"]["eval_clean"][idx] / task["negative_hours"]["eval_clean"]),
                     "noise_min_score": float(np.min(scores[np.isfinite(scores)])), "noise_hours": hours})
    return rows


def main() -> None:
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    with ProcessPoolExecutor(jobs) as pool:
        rows = [r for part in pool.map(run, range(len(common.TARGETS))) for r in part]
    table = pd.DataFrame(rows)
    out = common.RESULTS / "mswc" / "noise_only.csv"
    table.to_csv(out, index=False)
    print(table.groupby("method")[["noise_only_fah", "eval_noisy_fah", "eval_clean_fah"]].agg(["mean", "median"]))


if __name__ == "__main__":
    main()
