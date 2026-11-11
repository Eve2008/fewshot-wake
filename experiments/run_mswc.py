"""Speaker-independent few-shot wake detection on Persian MSWC (test split).

For each seed (1-3), shot count k (3, 6, 10) and each of 16 target words: enroll k
recordings by k different speakers, then score 60 recordings of the word by 60
other speakers, 4 h of other words by other speakers (clean, and with white+pink
noise at 10 dB SNR), and the shipped 60-min negative bank (clean and noisy).
Enrollment speakers are nested (k=3 is a subset of k=6, which is a subset of k=10)
and the 60 positives are the same for every k, so the k comparison is paired.

All speaker/word disjointness is fixed in results/splits.json (build_bank.py).
Per-threshold event counts are saved, so every calibration rule is evaluated
afterwards by summarize.py without rerunning the matchers.

Usage: python experiments/run_mswc.py [--jobs 8] [--shots 3,6,10] [--seeds 1,2,3]
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
from common import SR  # noqa: E402

OUT = common.RESULTS / "mswc"
ENROLL_POOL = 10
MAX_POSITIVES = 60
SNR_DB = 10.0
BANK_PREFIXES = {"15min": 900.0, "30min": 1800.0}


def stream_dir(seed: int) -> Path:
    return OUT / "streams" / f"seed{seed}"


def prepare(seed: int, jobs: int) -> None:
    folder = stream_dir(seed)
    if (folder / "info.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(seed)]
    started = time.time()
    rng = np.random.default_rng(seed + 1000)
    room, _ = common.stream([common.read(link) for link in split["room"]], rng, common.GAP_SECONDS)
    room_pcm = common.to_pcm16(room)
    emb, ends = common.models().clip_features(room_pcm.tobytes())
    np.save(folder / "room_feat.npy", emb)
    np.save(folder / "room_mfcc.npy", common.mfcc(room_pcm.astype(np.float32)))
    clean, spans = common.stream([common.read(link) for link in split["negatives"]], rng, common.GAP_SECONDS)
    np.save(folder / "neg_spans.npy", np.asarray(spans, dtype=np.int64))
    noisy = common.add_noise(clean, SNR_DB, seed=seed + 2000)
    info = {"room_seconds": room.size / SR, "negative_clips": len(split["negatives"])}
    for name, audio in (("clean", clean), ("noisy", noisy)):
        pcm = common.to_pcm16(audio)
        emb, ends = common.features(pcm, jobs=jobs, scratch=folder)
        np.save(folder / f"neg_{name}_feat.npy", emb)
        np.save(folder / f"neg_{name}_ends.npy", ends)
        np.save(folder / f"neg_{name}_mfcc.npy", common.mfcc_long(pcm, jobs=jobs, scratch=folder))
        info[f"{name}_seconds"] = pcm.size / SR
        info[f"{name}_frames"] = int(emb.shape[0])
    info["seconds_build"] = round(time.time() - started, 1)
    common.dump_json(folder / "info.json", info)
    print(f"seed {seed}: streams ready {info}", flush=True)


def plan(seed: int) -> list[dict]:
    table = pd.read_csv(common.MSWC / "test.csv")
    split = common.load_json(common.RESULTS / "splits.json")
    bank_speakers = set(split["bank_speakers"])
    tasks = []
    for index, word in enumerate(common.TARGETS):
        rng = np.random.default_rng(seed * 100 + index)
        rows = table[(table.WORD == word) & ~table.SPEAKER.isin(bank_speakers)]
        speakers = rows.SPEAKER.drop_duplicates().tolist()
        rng.shuffle(speakers)
        pick = lambda s: rows[rows.SPEAKER == s].LINK.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0]  # noqa: E731
        enroll = [pick(s) for s in speakers[:ENROLL_POOL]]
        positives = [pick(s) for s in speakers[ENROLL_POOL: ENROLL_POOL + MAX_POSITIVES]]
        tasks.append({"word": word, "index": index, "enroll": enroll, "positives": positives,
                      "enroll_speakers": speakers[:ENROLL_POOL],
                      "positive_speakers": speakers[ENROLL_POOL: ENROLL_POOL + MAX_POSITIVES]})
    return tasks


_BANK = None


def bank() -> dict:
    global _BANK
    if _BANK is None:
        folder = common.RESULTS / "bank"
        ends = np.load(folder / "ends.npy")
        _BANK = {name: (np.load(folder / f"{name}_feat.npy").astype(np.float32), ends,
                        np.load(folder / f"{name}_mfcc.npy").astype(np.float32)) for name in ("clean", "noisy")}
    return _BANK


def negatives(seed: int) -> dict:
    from methods import NegStream

    folder = stream_dir(seed)
    out = {}
    for name in ("clean", "noisy"):
        out[f"eval_{name}"] = NegStream(np.load(folder / f"neg_{name}_feat.npy", mmap_mode="r"),
                                        np.load(folder / f"neg_{name}_ends.npy"),
                                        np.load(folder / f"neg_{name}_mfcc.npy", mmap_mode="r"))
    for name, (emb, ends, mf) in bank().items():
        out[f"bank_{name}"] = NegStream(emb, ends, mf, prefixes=BANK_PREFIXES if name == "clean" else {})
    return out


def run_task(payload: tuple) -> dict:
    seed, shots, task = payload
    from methods import Positives, evaluate_wake

    started = time.time()
    folder = stream_dir(seed)
    rng_pos = np.random.default_rng(seed * 100 + task["index"] + 50)
    pos, spans = common.stream([common.read(link) for link in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
    pos_noisy = common.add_noise(pos, SNR_DB, seed=seed * 100 + task["index"] + 7)
    positives = {"clean": Positives(pos, spans), "noisy": Positives(pos_noisy, spans)}
    room = {"emb": np.load(folder / "room_feat.npy"), "mfcc": np.load(folder / "room_mfcc.npy")}
    clips = [common.read(link) for link in task["enroll"][:shots]]
    result = evaluate_wake(clips, positives, negatives(seed), room, bank()["clean"][0],
                           np.random.default_rng(seed * 100 + task["index"]))
    result.update({"word": task["word"], "seed": seed, "shots": shots, "seconds": round(time.time() - started, 1)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--shots", default="10,6,3")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--words", default="")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    shots = [int(s) for s in args.shots.split(",")]
    for seed in seeds:
        prepare(seed, args.jobs)
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for seed in seeds:
        tasks = plan(seed)
        common.dump_json(OUT / f"plan_seed{seed}.json", tasks)
        for k in shots:
            for task in tasks:
                if args.words and str(task["index"]) not in args.words.split(","):
                    continue
                path = tasks_dir / f"seed{seed}_k{k}_w{task['index']:02d}.json"
                if not path.is_file():
                    work.append((path, (seed, k, task)))
    print(f"{len(work)} tasks to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, payload): path for path, payload in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            futures[future].write_text(json.dumps(result, ensure_ascii=False))
            print(f"[{done}/{len(work)}] seed {result['seed']} k {result['shots']} {result['word']}: "
                  f"{len(result['methods'])} methods, {result['seconds']}s (elapsed {time.time() - started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
