"""Non-speech wakes, same-source enrollment (ESC-50): the analogue of same-speaker enrollment.

ESC-50 cuts up to 8 five-second clips ("takes") from one Freesound recording; all
takes of a recording are in the same fold. A user who records their own knock or
clap is closer to this case (same source, room and microphone) than to the
cross-recording protocol of run_esc50.py. Every recording with >= 4 takes is
used: enroll k = 3 takes (the loudest 1.5 s of each), test on its remaining
takes (a hit anywhere in the 5 s clip counts). Up to three distinct enrollment
subsets per recording (repetitions r = 1..3, chosen by a seeded permutation).

Negatives and banks are those of run_esc50.py: folds 3-5 of the other classes
plus the 4 h speech stream for evaluation; the 60-min speech bank plus folds 1-2
of the other classes as the bank. All clips of the wake's class are masked.

Usage: python experiments/run_esc50_samesrc.py [--jobs 2] [--methods nexa,...]
A task file that exists but lacks some of the requested methods is completed.
"""

from __future__ import annotations

import argparse
import itertools
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
import run_esc50  # noqa: E402
import run_mswc  # noqa: E402

OUT = common.RESULTS / "esc50_samesrc"
SHOTS = 3
MIN_TAKES = 4
REPEATS = 3
METHODS = ["nexa", "nexa-min", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "prototype", "mfcc-dtw"]


def plan() -> list[dict]:
    meta, _ = run_esc50.load_clips()
    tasks = []
    for (category, src), rows in meta.groupby(["category", "src_file"]):
        if len(rows) < MIN_TAKES:
            continue
        rows = rows.sort_values("take")
        index = rows.index.tolist()
        rng = np.random.default_rng(int(src) % (2**31) + 17)
        subsets = [tuple(sorted(c)) for c in itertools.combinations(index, SHOTS)]
        order = rng.permutation(len(subsets))
        for rep, pick in enumerate(order[:REPEATS], 1):
            enroll = list(subsets[pick])
            tasks.append({"category": category, "src_file": int(src), "repeat": rep, "fold": int(rows.fold.iloc[0]),
                          "enroll": [int(i) for i in enroll],
                          "positives": [int(i) for i in index if i not in enroll],
                          "takes": len(index)})
    return tasks


def run_task(payload: tuple) -> dict:
    task, methods = payload
    from methods import NegStream, Positives, evaluate_wake

    methods = list(methods)
    if not set(methods) & {"nexa", "nexa-min", "emb-only", "mel-only", "prototype"}:
        methods.append("prototype")   # evaluate_wake stops before MFCC-DTW without the deployed build (cheap)
    started = time.time()
    meta, clips = run_esc50.load_clips()
    folder = run_esc50.OUT / "streams"
    category = task["category"]
    pos, spans = common.stream([np.asarray(clips[i]) for i in task["positives"]],
                               np.random.default_rng(task["src_file"] % 1000 + 77), common.POS_PAD_SECONDS)
    negs = {}
    for name in ("eval", "bank"):
        index = common.load_json(folder / f"{name}_index.json")
        mask = [tuple(s) for s, c in zip(index["spans"], index["category"]) if c == category]
        negs[f"sound_{name}"] = NegStream(np.load(folder / f"{name}_feat.npy", mmap_mode="r"),
                                          np.load(folder / f"{name}_ends.npy"),
                                          np.load(folder / f"{name}_mfcc.npy", mmap_mode="r"), mask=mask)
    speech = run_mswc.negatives(1)
    negs["speech_eval"] = speech["eval_clean"]
    negs["speech_bank"] = speech["bank_clean"]
    negs["speech_bank"].prefixes = {}
    seed_folder = run_mswc.stream_dir(1)
    room = {"emb": np.load(seed_folder / "room_feat.npy"), "mfcc": np.load(seed_folder / "room_mfcc.npy")}
    enroll = [run_esc50.loudest(np.asarray(clips[i]), run_esc50.ENROLL_SECONDS) for i in task["enroll"]]
    result = evaluate_wake(enroll, {"clean": Positives(pos, spans)}, negs, room, run_mswc.bank()["clean"][0],
                           np.random.default_rng(task["src_file"] % 1000 + 5 + task["repeat"]), methods=methods)
    result.update({k: task[k] for k in ("category", "src_file", "repeat", "fold", "takes")})
    result.update({"shots": SHOTS, "seed": task["repeat"], "enroll_files": meta.filename[task["enroll"]].tolist(),
                   "positive_files": meta.filename[task["positives"]].tolist(),
                   "seconds": round(time.time() - started, 1)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    wanted = [m for m in METHODS if m in args.methods.split(",")]
    tasks = plan()
    common.dump_json(OUT / "plan.json", tasks)
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for task in tasks:
        path = tasks_dir / f"{task['category']}_src{task['src_file']}_r{task['repeat']}.json"
        if not path.is_file():
            work.append((path, (task, wanted)))
        else:
            missing = [m for m in wanted if m not in json.loads(path.read_text())["methods"]]
            if missing:
                work.append((path, (task, missing)))
    sources = {(t["category"], t["src_file"]) for t in tasks}
    print(f"{len(tasks)} tasks from {len(sources)} recordings in {len({c for c, _ in sources})} classes; "
          f"{len(work)} to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, payload): path for path, payload in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = futures[future]
            if path.is_file():
                old = json.loads(path.read_text())
                assert old["enroll_files"] == result["enroll_files"] and old["enrolled"] == result["enrolled"]
                for name, entry in result["methods"].items():
                    if entry.get("threshold") is not None and name in old["methods"]:
                        assert abs(entry["threshold"] - old["methods"][name]["threshold"]) < 1e-9
                    old["methods"].setdefault(name, entry)
                old["seconds"] = round(old["seconds"] + result["seconds"], 1)
                result = old
            path.write_text(json.dumps(result, ensure_ascii=False))
            print(f"[{done}/{len(work)}] {result['category']} src {result['src_file']} r {result['repeat']}: "
                  f"{len(result['methods'])} methods, {result['seconds']}s (elapsed {time.time() - started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
