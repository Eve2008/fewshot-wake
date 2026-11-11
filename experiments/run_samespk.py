"""Same-speaker few-shot wake detection on Persian MSWC (test split).

The deployed use case: the owner enrolls a word in their own voice and later says
it again. MSWC words are cut from Common Voice sentences, so a speaker who read
many sentences may say a frequent word many times. We take every evaluation
speaker (never in the bank) with a word repeated >= 15 times, one word per
speaker (the most repeated), enroll k = 3, 6 or 10 of those recordings (nested),
and test on up to 20 other recordings of the same word by the same speaker. A
'broad' set at k = 3 uses every speaker with >= 8 repeats.

Negatives: the seed-1 evaluation stream of the cross-speaker benchmark (4 h of
other words by other speakers, clean and noisy), the shipped bank, and the
speaker's own voice saying other words (up to 400 recordings, words outside the
bank vocabulary).

Usage: python experiments/run_samespk.py [--jobs 8]
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

OUT = common.RESULTS / "samespk"
MAIN_MIN = 15
BROAD_MIN = 8
ENROLL_POOL = 10
MAX_POSITIVES = 20
OWN_MAX = 400
METHODS = ["nexa", "nexa-min", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "prototype", "mfcc-dtw"]
NEG_SEED = 1


def plan() -> list[dict]:
    table = pd.read_csv(common.MSWC / "test.csv")
    split = common.load_json(common.RESULTS / "splits.json")
    bank_speakers, bank_words = set(split["bank_speakers"]), set(split["bank_words"])
    same_words = set(split["same_words"])
    rows = table[~table.SPEAKER.isin(bank_speakers)]
    counts = rows[rows.WORD.isin(same_words)].groupby(["SPEAKER", "WORD"]).size().reset_index(name="n")
    counts = counts.sort_values(["SPEAKER", "n", "WORD"], ascending=[True, False, True])
    best = counts.drop_duplicates("SPEAKER")
    tasks = []
    for index, (speaker, word, n) in enumerate(best[best.n >= BROAD_MIN].itertuples(index=False)):
        rng = np.random.default_rng(7000 + index)
        links = rows[(rows.SPEAKER == speaker) & (rows.WORD == word)].LINK.tolist()
        rng.shuffle(links)
        own = rows[(rows.SPEAKER == speaker) & (rows.WORD != word) & ~rows.WORD.isin(bank_words)].LINK.tolist()
        rng.shuffle(own)
        gender = table[table.SPEAKER == speaker].GENDER.iloc[0]
        if n >= MAIN_MIN:
            for k in (3, 6, 10):
                tasks.append({"set": "main", "index": index, "speaker": speaker, "word": word, "repeats": int(n),
                              "gender": gender, "shots": k, "enroll": links[:k],
                              "positives": links[ENROLL_POOL: ENROLL_POOL + MAX_POSITIVES], "own": own[:OWN_MAX]})
        tasks.append({"set": "broad", "index": index, "speaker": speaker, "word": word, "repeats": int(n),
                      "gender": gender, "shots": 3, "enroll": links[:3],
                      "positives": links[3: 3 + MAX_POSITIVES], "own": own[:OWN_MAX]})
    return tasks


def run_task(task: dict) -> dict:
    from methods import NegStream, Positives, evaluate_wake

    started = time.time()
    seed = NEG_SEED
    folder = run_mswc.stream_dir(seed)
    rng_pos = np.random.default_rng(9000 + task["index"])
    pos, spans = common.stream([common.read(link) for link in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
    pos_noisy = common.add_noise(pos, run_mswc.SNR_DB, seed=9500 + task["index"])
    positives = {"clean": Positives(pos, spans), "noisy": Positives(pos_noisy, spans)}
    negs = run_mswc.negatives(seed)
    if task["set"] == "broad":
        negs.pop("eval_noisy")
    own, _ = common.stream([common.read(link) for link in task["own"]], np.random.default_rng(9900 + task["index"]),
                           common.GAP_SECONDS)
    own_pcm = common.to_pcm16(own)
    emb, ends = common.models().clip_features(own_pcm.tobytes())
    negs["own_clean"] = NegStream(emb, ends, common.mfcc(own_pcm.astype(np.float32)))
    room = {"emb": np.load(folder / "room_feat.npy"), "mfcc": np.load(folder / "room_mfcc.npy")}
    clips = [common.read(link) for link in task["enroll"]]
    result = evaluate_wake(clips, positives, negs, room, run_mswc.bank()["clean"][0],
                           np.random.default_rng(8000 + task["index"]), methods=METHODS)
    result.update({k: task[k] for k in ("set", "index", "speaker", "word", "repeats", "gender", "shots")})
    result["own_clips"] = len(task["own"])
    result["seconds"] = round(time.time() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tasks = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    common.dump_json(OUT / "plan.json", tasks)
    main_n = len({t["index"] for t in tasks if t["set"] == "main"})
    broad_n = len({t["index"] for t in tasks if t["set"] == "broad"})
    print(f"main set: {main_n} speakers x 3 k; broad set: {broad_n} speakers at k=3", flush=True)
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    # Main set first (all k), then the broad set.
    tasks.sort(key=lambda t: (t["set"] != "main", -t["shots"], t["index"]))
    work = [(tasks_dir / f"{t['set']}_k{t['shots']}_s{t['index']:03d}.json", t) for t in tasks]
    work = [(p, t) for p, t in work if not p.is_file()]
    if args.limit:
        work = work[: args.limit]
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, t): p for p, t in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            futures[future].write_text(json.dumps(result, ensure_ascii=False))
            print(f"[{done}/{len(work)}] {result['set']} k {result['shots']} s{result['index']}: "
                  f"{len(result['methods'])} methods, {result['seconds']}s (elapsed {time.time() - started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
