"""Non-speech wakes: every ESC-50 class used as a few-shot wake.

ESC-50 (Piczak 2015; 2,000 five-second clips, 50 classes, 5 folds; clips cut from
the same Freesound recording share a fold). Read from the Hugging Face parquet
mirror ashraq/esc50 and resampled to 16 kHz; each clip is peak-normalised to 0.3.

For class c, seed s and k in {3, 6}: enroll k of the 8 fold-1 clips of c (the
loudest 1.5 s of each, as a user would record a single clap or knock), test on
the 32 clips of c in folds 2-5 (a hit anywhere in the 5 s clip counts).
Negatives: folds 3-5 of the other 49 classes (1.6 h), and the 4 h seed-1 speech
stream of the MSWC benchmark. Calibration banks: the speech bank alone, and the
speech bank plus folds 1-2 of the other classes (sound bank). Clips of class c
are masked out of every negative stream when c is the wake.

Usage: python experiments/run_esc50.py [--jobs 8] [--shots 6,3] [--methods nexa,...]
A task file that exists but lacks some of the requested methods is completed: only the
missing methods are scored (same clips and seeds, so the same templates) and merged in.
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
import soundfile as sf  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from common import SR  # noqa: E402

ESC = common.DATA / "esc50" / "extracted"
OUT = common.RESULTS / "esc50"
PEAK = 0.3 * 32767
ENROLL_SECONDS = 1.5
METHODS = ["nexa", "nexa-min", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "mel-only", "prototype", "mfcc-dtw"]
SEEDS = (1, 2, 3)
SHOTS = (3, 6)


def load_clips() -> tuple[pd.DataFrame, np.ndarray]:
    cache = common.DATA / "esc50" / "clips16k.npy"   # derived cache, not a result
    meta = pd.read_csv(ESC / "meta.csv").sort_values("filename").reset_index(drop=True)
    if cache.is_file():
        return meta, np.load(cache, mmap_mode="r")
    from scipy.signal import resample_poly

    clips = []
    for name in meta.filename:
        data, rate = sf.read(ESC / "audio" / name, dtype="float64")
        assert rate == 44100
        y = resample_poly(data, 160, 441)
        peak = np.max(np.abs(y))
        clips.append((y / max(peak, 1e-9) * PEAK).astype(np.float32))
    arr = np.stack(clips)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return meta, arr


def prepare(jobs: int) -> None:
    if (OUT / "streams" / "info.json").is_file():
        return
    meta, clips = load_clips()
    folder = OUT / "streams"
    folder.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, folds in (("eval", (3, 4, 5)), ("bank", (1, 2))):
        rows = meta[meta.fold.isin(folds)].sample(frac=1.0, random_state=5)
        values, spans = common.stream([np.asarray(clips[i]) for i in rows.index], np.random.default_rng(55),
                                      common.GAP_SECONDS)
        pcm = common.to_pcm16(values)
        emb, ends = common.features(pcm, jobs=jobs, scratch=folder)
        np.save(folder / f"{name}_feat.npy", emb)
        np.save(folder / f"{name}_ends.npy", ends)
        np.save(folder / f"{name}_mfcc.npy", common.mfcc_long(pcm, jobs=jobs, scratch=folder))
        common.dump_json(folder / f"{name}_index.json", {"rows": rows.index.tolist(),
                                                         "category": rows.category.tolist(), "spans": spans})
        info[name] = {"clips": len(rows), "seconds": pcm.size / SR, "frames": int(emb.shape[0])}
    common.dump_json(folder / "info.json", info)
    print("ESC-50 streams ready", info, flush=True)


def loudest(clip: np.ndarray, seconds: float) -> np.ndarray:
    width = int(seconds * SR)
    energy = np.convolve(clip.astype(np.float64) ** 2, np.ones(320), mode="valid")[::160]
    best, best_start = -1.0, 0
    steps = int(width / 160)
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    for start in range(0, max(1, energy.size - steps)):
        total = csum[min(energy.size, start + steps)] - csum[start]
        if total > best:
            best, best_start = total, start
    a = best_start * 160
    return clip[a: a + width]


def run_task(payload: tuple) -> dict:
    category, seed, shots = payload[:3]
    methods = list(payload[3]) if len(payload) > 3 else list(METHODS)
    if not set(methods) & {"nexa", "nexa-min", "emb-only", "mel-only", "prototype"}:
        methods.append("prototype")   # evaluate_wake stops before MFCC-DTW without the deployed build (cheap)
    from methods import NegStream, Positives, evaluate_wake

    started = time.time()
    meta, clips = load_clips()
    folder = OUT / "streams"
    own = meta[meta.category == category]
    fold1 = own[own.fold == 1].index.tolist()
    rng = np.random.default_rng(seed * 1000 + int(own.target.iloc[0]))
    enroll_idx = list(rng.permutation(fold1)[:shots])
    pos_idx = own[own.fold != 1].index.tolist()
    pos, spans = common.stream([np.asarray(clips[i]) for i in pos_idx], np.random.default_rng(seed + 77),
                               common.POS_PAD_SECONDS)
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
    enroll = [loudest(np.asarray(clips[i]), ENROLL_SECONDS) for i in enroll_idx]
    result = evaluate_wake(enroll, {"clean": Positives(pos, spans)}, negs, room, run_mswc.bank()["clean"][0],
                           np.random.default_rng(seed * 1000 + 5), methods=list(methods))
    result.update({"category": category, "seed": seed, "shots": shots,
                   "enroll_files": meta.filename[enroll_idx].tolist(), "seconds": round(time.time() - started, 1)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--categories", default="")
    parser.add_argument("--shots", default=",".join(str(k) for k in sorted(SHOTS, reverse=True)))
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    wanted = [m for m in METHODS if m in args.methods.split(",")]
    prepare(args.jobs)
    meta, _ = load_clips()
    categories = sorted(meta.category.unique())
    if args.categories:
        categories = args.categories.split(",")
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for shots in [int(k) for k in args.shots.split(",")]:
        for seed in SEEDS:
            for category in categories:
                path = tasks_dir / f"{category}_seed{seed}_k{shots}.json"
                if not path.is_file():
                    work.append((path, (category, seed, shots, wanted)))
                else:
                    missing = [m for m in wanted if m not in json.loads(path.read_text())["methods"]]
                    if missing:
                        work.append((path, (category, seed, shots, missing)))
    print(f"{len(work)} tasks to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, payload): path for path, payload in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = futures[future]
            if path.is_file():   # merge newly scored methods into the existing task file
                old = json.loads(path.read_text())
                assert old["enroll_files"] == result["enroll_files"] and old["enrolled"] == result["enrolled"]
                for name, entry in result["methods"].items():
                    if entry.get("threshold") is not None and name in old["methods"]:
                        assert abs(entry["threshold"] - old["methods"][name]["threshold"]) < 1e-9
                    old["methods"].setdefault(name, entry)
                old["seconds"] = round(old["seconds"] + result["seconds"], 1)
                result = old
            path.write_text(json.dumps(result, ensure_ascii=False))
            print(f"[{done}/{len(work)}] {result['category']} seed {result['seed']} k {result['shots']}: "
                  f"{len(result['methods'])} methods, {result['seconds']}s (elapsed {time.time() - started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
