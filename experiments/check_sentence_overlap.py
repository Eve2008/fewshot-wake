"""Control after review: do false alarms come from negative clips cut from a sentence that contains the target?

MSWC clips are 1-s windows cut from Common Voice sentences by forced alignment, so a negative clip cut from a
sentence in which the target word was also said may hold part of the target. For each negative clip we look up
its Common Voice sentence (the clip id in the MSWC link) and whether the target word was extracted from the same
sentence.

1. Same speaker, main set, k = 10, own-voice negatives: clean templates and MFCC + DTW at the bank tail-fit
   threshold (1 FA/h). Scores are recomputed (as in run_samespk.py) and the alarm count is checked against the
   stored count.
2. Cross-speaker, seed 1, k = 6, clean templates, bank tail fit: the 56 alarms of
   results/mswc/fa_words_seed1_k6_noaug.csv, mapped to their clips.

Output: results/manuscript_checks/sentence_overlap_samespk.csv, sentence_overlap_mswc.csv, sentence_overlap.json
Usage: python experiments/check_sentence_overlap.py [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import re
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

OUT = common.RESULTS / "manuscript_checks"
CV = re.compile(r"common_voice_fa_(\d+)")
_SENT = None


def sentence_words() -> dict:
    global _SENT
    if _SENT is None:
        t = pd.read_csv(common.MSWC / "test.csv")
        t["cv"] = t.LINK.str.extract(CV.pattern)[0]
        _SENT = t.groupby("cv").WORD.apply(set).to_dict()
    return _SENT


def alarms(scores, tau, refractory) -> list[int]:
    out, last = [], -10**9
    for f in np.nonzero(np.asarray(scores) <= tau)[0]:
        if f - last >= refractory:
            out.append(int(f))
            last = f
    return out


def run_samespk(task: dict) -> list[dict]:
    from methods import enroll_clips
    from nexawake import fast
    from nexawake.enroll import build_wake

    res = json.loads((common.RESULTS / "samespk" / "tasks" / f"main_k10_s{task['index']:03d}.json").read_text())
    if not res["methods"]:
        return []
    folder = run_mswc.stream_dir(1)
    room = np.load(folder / "room_feat.npy")
    enrollments, audio = enroll_clips([common.read(l) for l in task["enroll"]], np.random.default_rng(8000 + task["index"]))
    own, spans = common.stream([common.read(l) for l in task["own"]], np.random.default_rng(9900 + task["index"]),
                               common.GAP_SECONDS)
    pcm = common.to_pcm16(own)
    emb, ends = common.models().clip_features(pcm.tobytes())
    words = sentence_words()
    shared = np.array([task["word"] in words.get(CV.search(l).group(1), set()) for l in task["own"]])
    starts = np.array([s[0] for s in spans])
    out = []
    built = build_wake(enrollments, [room], augment=False)
    entry = res["methods"]["nexa-noaug"]
    idx, _ = tail_index(np.asarray(entry["neg"]["bank_clean"], float), res["negative_hours"]["bank_clean"], 1.0)
    sc = fast.wake_score(fast.template_scores(emb, built.spec.templates, built.normalizer))
    fa = alarms(sc, float(common.GRID[idx]), common.REFRACTORY_FRAMES)
    ok_clean = len(fa) == entry["neg"]["own_clean"][idx]
    for f in fa:
        c = max(0, int(np.searchsorted(starts, ends[f], side="right") - 1))
        out.append({"index": task["index"], "method": "nexa-noaug", "clip_from_target_sentence": bool(shared[c]),
                    "count_matches_stored": ok_clean})
    entry = res["methods"]["mfcc-dtw"]
    idx, _ = tail_index(np.asarray(entry["neg"]["bank_clean"], float), res["negative_hours"]["bank_clean"], 1.0)
    templates = []
    for padded, onset, offset in audio:
        feats = common.mfcc(padded)
        templates.append(feats[max(0, onset // common.MFCC_HOP): max(onset // common.MFCC_HOP + 3, offset // common.MFCC_HOP)])
    norm = common.fit_normalizer([common.mfcc(p) for p, _, _ in audio] + [np.load(folder / "room_mfcc.npy")])
    sc = fast.wake_score(fast.template_scores(common.mfcc(pcm.astype(np.float32)), templates, norm))
    fa = alarms(sc, float(common.GRID[idx]), common.MFCC_REFRACTORY)
    ok_mfcc = len(fa) == entry["neg"]["own_clean"][idx]
    for f in fa:
        c = max(0, int(np.searchsorted(starts, f * common.MFCC_HOP, side="right") - 1))
        out.append({"index": task["index"], "method": "mfcc-dtw", "clip_from_target_sentence": bool(shared[c]),
                    "count_matches_stored": ok_mfcc})
    out.append({"index": task["index"], "method": "_share", "share_from_target_sentence": float(shared.mean()),
                "own_clips": int(len(shared)), "own_from_target_sentence": int(shared.sum())})
    return out


def mswc_check() -> tuple[pd.DataFrame, dict]:
    split = common.load_json(common.RESULTS / "splits.json")["seeds"]["1"]
    links = split["negatives"]
    words = sentence_words()
    folder = run_mswc.stream_dir(1)
    ends = np.load(folder / "neg_clean_ends.npy")
    starts = np.load(folder / "neg_spans.npy")[:, 0]
    fa = pd.read_csv(common.RESULTS / "mswc" / "fa_words_seed1_k6_noaug.csv").dropna(subset=["frame"])
    rows = []
    for r in fa.itertuples():
        c = max(0, int(np.searchsorted(starts, ends[int(r.frame)], side="right") - 1))
        rows.append({"target": r.target, "negative_word": r.negative_word,
                     "clip_from_target_sentence": r.target in words.get(CV.search(links[c]).group(1), set())})
    per_target = {t: int(sum(t in words.get(CV.search(l).group(1), set()) for l in links)) for t in common.TARGETS}
    return pd.DataFrame(rows), {"negatives_from_a_target_sentence_by_target": per_target,
                                "negatives_from_any_target_sentence": int(sum(
                                    any(t in words.get(CV.search(l).group(1), set()) for t in common.TARGETS)
                                    for l in links))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = common.load_json(common.RESULTS / "samespk" / "plan.json")
    tasks = [t for t in plan if t["set"] == "main" and t["shots"] == 10]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [r for part in pool.map(run_samespk, tasks) for r in part]
    same = pd.DataFrame(rows)
    same.to_csv(OUT / "sentence_overlap_samespk.csv", index=False)
    cross, info = mswc_check()
    cross.to_csv(OUT / "sentence_overlap_mswc.csv", index=False)
    alarms_only = same[same.method != "_share"]
    share = same[same.method == "_share"]
    summary = {
        "samespk_main_k10": {
            "own_clips": int(share.own_clips.sum()), "own_clips_from_target_sentence": int(share.own_from_target_sentence.sum()),
            "alarms": {m: {"total": int(len(g)), "from_target_sentence": int(g.clip_from_target_sentence.sum()),
                           "counts_match_stored": bool(g.count_matches_stored.all())}
                       for m, g in alarms_only.groupby("method")}},
        "mswc_seed1_k6_noaug": {"alarms": int(len(cross)), "from_target_sentence": int(cross.clip_from_target_sentence.sum()),
                                **info},
    }
    common.dump_json(OUT / "sentence_overlap.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
