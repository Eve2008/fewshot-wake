"""mHuBERT-147 + DTW on the same-speaker main set (k = 3, 6, 10), as run_samespk_v2.py for the other variants.

Same wakes as results/samespk_v2 (same enrollment links, padding noise, positives and own-voice streams, same
random seeds). Negatives: the seed-1 evaluation stream with the speaker's own clips masked (SSL features of
that stream from run_ssl.py), the speaker's own other words, and the bank. Feature settings, templates and
scoring are those of run_ssl.py (layer chosen in results/ssl/layer_selection.json, PCA fitted on the bank).

Output: results/ssl_samespk/feats/s<idx>.npz, results/ssl_samespk/tasks/main_k<k>_s<idx>.json
Usage: python experiments/run_ssl_samespk.py [--procs 3] [--steps feats,score]
"""

from __future__ import annotations

import argparse
import multiprocessing
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
import run_ssl  # noqa: E402
from common import SR  # noqa: E402

OUT = common.RESULTS / "ssl_samespk"
SRC = common.RESULTS / "samespk"


def main_tasks() -> list[dict]:
    plan = common.load_json(SRC / "plan.json")
    return [t for t in plan if t["set"] == "main" and t["shots"] == 10]


def step_feats(pool) -> None:
    layer = run_ssl.chosen_layer()
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    for task in main_tasks():
        path = OUT / "feats" / f"s{task['index']:03d}.npz"
        if path.is_file():
            continue
        i = task["index"]
        pos, spans = common.stream([common.read(l) for l in task["positives"]], np.random.default_rng(9000 + i),
                                   common.POS_PAD_SECONDS)
        own, own_spans = common.stream([common.read(l) for l in task["own"]], np.random.default_rng(9900 + i),
                                       common.GAP_SECONDS)
        pf, pe = run_ssl.features(pos, layer, pool, OUT / "scratch")
        of, oe = run_ssl.features(own, layer, pool, OUT / "scratch")
        enroll = run_ssl.padded_enrollment([common.read(l) for l in task["enroll"]], np.random.default_rng(8000 + i))
        items = list(pool.map(run_ssl._enroll_part, [(p, layer) for p, _b in enroll]))
        arrays = {"pos_feat": pf, "pos_ends": pe, "pos_spans": np.asarray(spans, dtype=np.int64),
                  "own_feat": of, "own_ends": oe}
        for j, ((_p, b), f) in enumerate(zip(enroll, items)):
            arrays[f"enroll{j}"] = f
            arrays[f"bounds{j}"] = np.asarray(b if b else (-1, -1), dtype=np.int64)
        np.savez(path, **arrays)
        print(f"speaker {i}: features saved", flush=True)


def score(args) -> list[tuple[str, dict]]:
    task, = args
    from methods import mask_frames
    from nexawake import fast

    i = task["index"]
    data = np.load(OUT / "feats" / f"s{i:03d}.npz")
    bank = np.load(run_ssl.OUT / "bank_feat.npy").astype(np.float32)
    folder = run_ssl.OUT / "streams" / "seed1"
    neg = np.load(folder / "neg_clean_feat.npy").astype(np.float32)
    neg_ends = np.load(folder / "neg_clean_ends.npy")
    room = np.load(folder / "room_feat.npy").astype(np.float32)
    split = common.load_json(common.RESULTS / "splits.json")["seeds"]["1"]
    spans = np.load(run_mswc.stream_dir(1) / "neg_spans.npy")
    mask = [(int(a), int(b)) for (a, b), spk in zip(spans, split["negative_speakers"]) if spk == task["speaker"]]
    keep = mask_frames(mask, neg_ends)
    hop_hours = run_ssl.HOP / SR / 3600
    out = []
    for k in (3, 6, 10):
        name = f"main_k{k}_s{i:03d}.json"
        original = common.load_json(common.RESULTS / "samespk_v2" / "tasks" / name)
        enroll = [(data[f"enroll{j}"], tuple(data[f"bounds{j}"])) for j in range(k)]
        accepted = [(f, b) for f, b in enroll if b[0] >= 0]
        r = {k2: original[k2] for k2 in ("set", "index", "speaker", "word", "repeats", "gender", "shots")}
        r.update({"enrolled": len(accepted), "positives": {"clean": int(len(data["pos_spans"]))},
                  "negative_hours": {"eval_clean": float(keep.sum()) * hop_hours,
                                     "bank_clean": bank.shape[0] * hop_hours,
                                     "own_clean": data["own_feat"].shape[0] * hop_hours},
                  "prefix_hours": {"eval_clean": {}, "bank_clean": {"15min": 0.25, "30min": 0.5}, "own_clean": {}},
                  "masked_clips": len(mask), "methods": {}})
        assert r["enrolled"] == original["enrolled"], (name, r["enrolled"], original["enrolled"])
        if len(accepted) >= 3:
            temps = [run_ssl.template(f, b[0], b[1]) for f, b in accepted]
            norm = common.fit_normalizer([f for f, _b in accepted] + [room])
            fn = lambda x: fast.wake_score(fast.template_scores(x, temps, norm))  # noqa: E731
            entry = {"threshold": None, "calibration": None, "pos": {}, "neg": {}, "prefix": {}}
            entry["pos"]["clean"] = common.window_min(fn(data["pos_feat"]), data["pos_ends"],
                                                      [tuple(s) for s in data["pos_spans"]])
            scores = np.where(keep, fn(neg), np.inf)
            entry["neg"]["eval_clean"] = common.events(scores, run_ssl.REFRACTORY).tolist()
            entry["neg"]["own_clean"] = common.events(fn(data["own_feat"]), run_ssl.REFRACTORY).tolist()
            bank_scores = fn(bank)
            entry["neg"]["bank_clean"] = common.events(bank_scores, run_ssl.REFRACTORY).tolist()
            entry["prefix"]["bank_clean"] = {p: common.events(bank_scores[: int(s * SR / run_ssl.HOP)],
                                                              run_ssl.REFRACTORY).tolist()
                                             for p, s in run_mswc.BANK_PREFIXES.items()}
            r["methods"]["ssl-dtw"] = entry
        out.append((name, r))
    return out


def step_score(procs: int) -> None:
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)
    work = [(t,) for t in main_tasks() if not (OUT / "tasks" / f"main_k10_s{t['index']:03d}.json").is_file()]
    with ProcessPoolExecutor(procs) as pool:
        for results in pool.map(score, work):
            for name, r in results:
                (OUT / "tasks" / name).write_text(json.dumps(r, ensure_ascii=False))
            print(f"scored speaker {results[0][1]['index']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--steps", default="feats,score")
    args = parser.parse_args()
    steps = args.steps.split(",")
    if "feats" in steps:
        # 'spawn': the main process may run the model itself (a stream shorter than one 30-s chunk), and
        # workers forked after that deadlock in torch's thread pool.
        with ProcessPoolExecutor(args.procs, mp_context=multiprocessing.get_context("spawn")) as pool:
            step_feats(pool)
    if "score" in steps:
        step_score(args.procs * 2)


if __name__ == "__main__":
    main()
