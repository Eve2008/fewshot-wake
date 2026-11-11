"""Exploratory: does a background normaliser that follows the current acoustic condition
keep the bank-calibrated thresholds valid when stationary noise appears?

Same tasks as run_mswc.py (seed 1, all 16 words, k = 3, 6, 10). Templates are
normalised with the bank normaliser (clean enrollments + clean bank, as
nexa-noaug-bnorm); every query stream is normalised with the mean/std of its own
background: the clean bank, the noisy bank, the clean and the noisy 4 h evaluation
streams. Positives use the statistics of the evaluation stream of the same
condition, never their own. With stationary noise this approximates a long-window
online normaliser after it has converged; it is an upper bound for such a scheme,
not a deployed component.

Usage: python experiments/run_condnorm.py [--jobs 8]
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

import common  # noqa: E402
import run_mswc  # noqa: E402

OUT = common.RESULTS / "condnorm"
SEED = 1


def scores(seq, templates, tnorm, qnorm) -> np.ndarray:
    from nexawake import fast

    packed, lengths = fast.pack(templates, tnorm)
    queries = qnorm.apply(np.asarray(seq, dtype=np.float32)).astype(np.float64)
    per = fast._sequence_scores(np.ascontiguousarray(queries), np.ascontiguousarray(packed), lengths)
    return fast.wake_score(per)


def run_task(payload) -> dict:
    shots, task = payload
    from methods import enroll_clips
    from nexawake.enroll import build_wake
    from nexawake.wake import Normalizer

    started = time.time()
    folder = run_mswc.stream_dir(SEED)
    negs = run_mswc.negatives(SEED)
    stats = {name: Normalizer.fit([np.asarray(neg.emb, dtype=np.float32)]) for name, neg in negs.items()}
    rng_pos = np.random.default_rng(SEED * 100 + task["index"] + 50)
    pos, spans = common.stream([common.read(link) for link in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
    pos_noisy = common.add_noise(pos, run_mswc.SNR_DB, seed=SEED * 100 + task["index"] + 7)
    m = common.models()
    pos_feat = {"clean": m.clip_features(common.to_pcm16(pos).tobytes()),
                "noisy": m.clip_features(common.to_pcm16(pos_noisy).tobytes())}
    room = np.load(folder / "room_feat.npy")
    clips = [common.read(link) for link in task["enroll"][:shots]]
    enrollments, _ = enroll_clips(clips, np.random.default_rng(SEED * 100 + task["index"]))
    bank_clean = np.asarray(run_mswc.bank()["clean"][0], dtype=np.float32)
    result = {"word": task["word"], "seed": SEED, "shots": shots, "enrolled": len(enrollments),
              "negative_hours": {n: s.hours() for n, s in negs.items()},
              "prefix_hours": {n: {p: v / 3600.0 for p, v in s.prefixes.items()} for n, s in negs.items()},
              "methods": {}}
    for name, augment in (("nexa-noaug-cn", False), ("nexa-cn", True)):
        built = build_wake(enrollments, [room], augment=augment)
        if built is None:
            continue
        clean = [np.asarray(e.variants[0][0]) for e in enrollments]
        tnorm = Normalizer.fit(clean + [bank_clean])
        entry = {"threshold": None, "pos": {}, "neg": {}, "prefix": {}}
        for cond, qname in (("clean", "eval_clean"), ("noisy", "eval_noisy")):
            emb, ends = pos_feat[cond]
            entry["pos"][cond] = common.window_min(scores(emb, built.spec.templates, tnorm, stats[qname]), ends, spans)
        for sname, neg in negs.items():
            s = scores(neg.emb, built.spec.templates, tnorm, stats[sname])
            entry["neg"][sname] = common.events(s).tolist()
            if neg.prefixes:
                entry["prefix"][sname] = {p: common.events(s[: int(sec / common.FRAME_SECONDS)]).tolist()
                                          for p, sec in neg.prefixes.items()}
        result["methods"][name] = entry
    result["seconds"] = round(time.time() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    tasks = common.load_json(common.RESULTS / "mswc" / f"plan_seed{SEED}.json")
    out = OUT / "tasks"
    out.mkdir(parents=True, exist_ok=True)
    work = [(out / f"seed{SEED}_k{k}_w{t['index']:02d}.json", (k, t)) for k in (10, 6, 3) for t in tasks]
    work = [(p, w) for p, w in work if not p.is_file()]
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, w): p for p, w in work}
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            futures[future].write_text(json.dumps(r, ensure_ascii=False))
            print(f"[{done}/{len(work)}] k {r['shots']} {r['word']} {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
