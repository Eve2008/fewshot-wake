"""Which words trigger the false alarms that remain after bank calibration?

Seed 1, k = 6, clean templates (nexa-noaug), threshold = bank 60 min tail fit at
1 FA/h (the same rule as summarize.py). Re-scores the clean 4 h evaluation stream,
takes every alarm (refractory 1.2 s) and maps it to the negative clip that was
playing when the alarm frame ended.

Usage: python experiments/diagnose_fa_words.py
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
from summarize import tail_index  # noqa: E402

SEED, SHOTS, TARGET = 1, 6, 1.0


def alarms(scores: np.ndarray, tau: float, refractory: int = common.REFRACTORY_FRAMES) -> list[int]:
    out, last = [], -10**9
    for frame in np.nonzero(scores <= tau)[0]:
        if frame - last >= refractory:
            out.append(int(frame))
            last = frame
    return out


def run(index: int) -> list[dict]:
    from methods import enroll_clips
    from nexawake import fast
    from nexawake.enroll import build_wake

    plan = common.load_json(common.RESULTS / "mswc" / f"plan_seed{SEED}.json")[index]
    task = common.load_json(common.RESULTS / "mswc" / "tasks" / f"seed{SEED}_k{SHOTS}_w{index:02d}.json")
    entry = task["methods"]["nexa-noaug"]
    idx, _ = tail_index(np.asarray(entry["neg"]["bank_clean"], dtype=float), task["negative_hours"]["bank_clean"],
                        TARGET)
    tau = float(common.GRID[idx])
    folder = run_mswc.stream_dir(SEED)
    room = np.load(folder / "room_feat.npy")
    clips = [common.read(link) for link in plan["enroll"][:SHOTS]]
    enrollments, _ = enroll_clips(clips, np.random.default_rng(SEED * 100 + index))
    built = build_wake(enrollments, [room], augment=False)
    emb = np.load(folder / "neg_clean_feat.npy", mmap_mode="r")
    ends = np.load(folder / "neg_clean_ends.npy")
    spans = np.load(folder / "neg_spans.npy")
    scores = fast.wake_score(fast.template_scores(emb, built.spec.templates, built.normalizer))
    split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(SEED)]
    rows = []
    found = alarms(scores, tau)
    assert len(found) == entry["neg"]["eval_clean"][idx], (len(found), entry["neg"]["eval_clean"][idx])
    for frame in found:
        pos = ends[frame]
        # The clip whose audio overlaps the 0.76 s window ending at this frame the most recent.
        clip = int(np.searchsorted(spans[:, 0], pos, side="right") - 1)
        rows.append({"target": plan["word"], "tau": tau, "frame": frame, "score": float(scores[frame]),
                     "negative_word": split["negative_words"][clip] if clip >= 0 else None,
                     "previous_word": split["negative_words"][clip - 1] if clip >= 1 else None})
    if not rows:
        rows.append({"target": plan["word"], "tau": tau, "frame": None, "score": None, "negative_word": None,
                     "previous_word": None})
    return rows


def main() -> None:
    with ProcessPoolExecutor(4) as pool:
        rows = [r for part in pool.map(run, range(len(common.TARGETS))) for r in part]
    table = pd.DataFrame(rows)
    table.to_csv(common.RESULTS / "mswc" / "fa_words_seed1_k6_noaug.csv", index=False)
    print(table.dropna().groupby("target").negative_word.apply(lambda s: ", ".join(s)).to_string())


if __name__ == "__main__":
    main()
