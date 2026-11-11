"""Out-of-domain negatives: continuous read Persian speech from FLEURS (fa_ir, test split).

The bank and the MSWC evaluation negatives are both isolated one-second words. Here
the same wakes (cross-speaker, k = 6, 3 seeds, 16 words) and the same bank
thresholds are tested on FLEURS sentences (Conneau et al., 2023), concatenated
with 0.25 s gaps. Utterances whose transcription contains the target word as a
substring (after mapping Arabic kaf/yeh to the Persian letters and removing
ZWNJ) are masked out for that wake, so no true occurrence counts as a false alarm.

The enrollment, templates and normaliser are rebuilt exactly as in run_mswc.py
(same clips, same random seeds); the output merges the FLEURS event counts into a
copy of each main-run task file.

Usage: python experiments/run_fleurs.py [--jobs 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from common import SR  # noqa: E402

SRC = common.DATA / "fleurs_fa"
OUT = common.RESULTS / "fleurs"
SHOTS = 6
METHODS = ["nexa", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "mfcc-dtw"]


def norm(text: str) -> str:
    return (text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").replace("‌", "")
            .replace("ۀ", "ه").replace("أ", "ا").replace("إ", "ا"))


def utterances() -> list[dict]:
    rows = []
    with open(SRC / "test.tsv", newline="", encoding="utf-8") as handle:
        for parts in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(parts) < 4:
                continue
            path = SRC / "test" / parts[1]
            if path.is_file():
                rows.append({"file": parts[1], "raw": parts[2], "text": norm(parts[2] + " " + parts[3])})
    return rows


def prepare(jobs: int) -> None:
    folder = OUT / "streams"
    if (folder / "info.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    rows = utterances()
    clips = []
    for row in rows:
        data, rate = sf.read(SRC / "test" / row["file"], dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        assert rate == SR, rate
        clips.append(data * 32767.0)
    values, spans = common.stream(clips, np.random.default_rng(66), common.GAP_SECONDS)
    pcm = common.to_pcm16(values)
    emb, ends = common.features(pcm, jobs=jobs, scratch=folder)
    np.save(folder / "feat.npy", emb)
    np.save(folder / "ends.npy", ends)
    np.save(folder / "mfcc.npy", common.mfcc_long(pcm, jobs=jobs, scratch=folder))
    common.dump_json(folder / "index.json", {"spans": spans, "text": [r["text"] for r in rows],
                                             "file": [r["file"] for r in rows]})
    info = {"utterances": len(rows), "seconds": pcm.size / SR, "frames": int(emb.shape[0]),
            "speech_seconds": float(sum(c.size for c in clips) / SR)}
    common.dump_json(folder / "info.json", info)
    print("FLEURS stream ready", info, flush=True)


def run_task(payload) -> dict:
    seed, task = payload
    from methods import NegStream, evaluate_wake

    started = time.time()
    folder = OUT / "streams"
    index = common.load_json(folder / "index.json")
    target = norm(task["word"])
    mask = [tuple(s) for s, text in zip(index["spans"], index["text"]) if target in text]
    neg = NegStream(np.load(folder / "feat.npy", mmap_mode="r"), np.load(folder / "ends.npy"),
                    np.load(folder / "mfcc.npy", mmap_mode="r"), mask=mask)
    room_dir = run_mswc.stream_dir(seed)
    room = {"emb": np.load(room_dir / "room_feat.npy"), "mfcc": np.load(room_dir / "room_mfcc.npy")}
    clips = [common.read(link) for link in task["enroll"][:SHOTS]]
    result = evaluate_wake(clips, {}, {"fleurs": neg}, room, run_mswc.bank()["clean"][0],
                           np.random.default_rng(seed * 100 + task["index"]), methods=METHODS)
    main = common.load_json(common.RESULTS / "mswc" / "tasks" / f"seed{seed}_k{SHOTS}_w{task['index']:02d}.json")
    for method in list(main["methods"]):
        if method not in result["methods"]:
            del main["methods"][method]
            continue
        new = result["methods"][method]
        if new["threshold"] is not None:   # same enrollment must give the same deployed threshold
            assert abs(new["threshold"] - main["methods"][method]["threshold"]) < 1e-9, method
        main["methods"][method]["neg"]["fleurs"] = new["neg"]["fleurs"]
    main["negative_hours"]["fleurs"] = result["negative_hours"]["fleurs"]
    main["fleurs_masked_utterances"] = len(mask)
    main["seconds_fleurs"] = round(time.time() - started, 1)
    return main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    prepare(args.jobs)
    out = OUT / "tasks"
    out.mkdir(parents=True, exist_ok=True)
    work = []
    for seed in (1, 2, 3):
        for task in common.load_json(common.RESULTS / "mswc" / f"plan_seed{seed}.json"):
            path = out / f"seed{seed}_k{SHOTS}_w{task['index']:02d}.json"
            if not path.is_file():
                work.append((path, (seed, task)))
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, w): p for p, w in work}
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            futures[future].write_text(json.dumps(r, ensure_ascii=False))
            print(f"[{done}/{len(work)}] seed {r['seed']} {r['word']} masked {r['fleurs_masked_utterances']} "
                  f"{r['seconds_fleurs']}s", flush=True)


if __name__ == "__main__":
    main()
