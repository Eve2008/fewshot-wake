"""Cross-speaker MSWC: the variants of methods_extra.py on exactly the wakes of run_mswc.py.

Positives, enrollment clips, random seeds and negative streams are those of run_mswc.run_task, so every
task here pairs with results/mswc/tasks/<same name>.json. Output: results/mswc_x/tasks/*.json, one file per
task (resumable). The heads' training negatives (results/headneg/) are built once: 1,000 clips of bank
speakers saying bank words that are not in the shipped bank (disjoint from the bank and from every
evaluation clip).

Usage: python experiments/run_extra.py [--jobs 8] [--seeds 1,2,3] [--shots 10,6,3]
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

OUT = common.RESULTS / "mswc_x"
HEADNEG = common.RESULTS / "headneg"
HEADNEG_CLIPS = 1000


def prepare_headneg(jobs: int = 8) -> None:
    if (HEADNEG / "info.json").is_file():
        return
    HEADNEG.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(common.MSWC / "test.csv")
    split = common.load_json(common.RESULTS / "splits.json")
    pool = table[table.SPEAKER.isin(set(split["bank_speakers"])) & table.WORD.isin(set(split["bank_words"]))]
    used = set(split["bank"]) | {link for v in split["seeds"].values() for link in v["room"] + v["negatives"]}
    pool = pool[~pool.LINK.isin(used)]
    pick = pool.sample(n=HEADNEG_CLIPS, random_state=12)
    values, _spans = common.stream([common.read(link) for link in pick.LINK], np.random.default_rng(13),
                                   common.GAP_SECONDS)
    pcm = common.to_pcm16(values)
    emb, _ends = common.features(pcm, jobs=jobs, scratch=HEADNEG)
    np.save(HEADNEG / "feat.npy", emb.astype(np.float32))
    info = {"clips": HEADNEG_CLIPS, "seconds": values.size / common.SR, "frames": int(emb.shape[0]),
            "speakers": int(pick.SPEAKER.nunique()), "words": int(pick.WORD.nunique()),
            "pool_clips_available": int(len(pool)), "links": pick.LINK.tolist()}
    assert not (set(pick.LINK) & used)
    common.dump_json(HEADNEG / "info.json", info)
    print(f"head negatives ready: {info['seconds'] / 60:.1f} min, {info['frames']} frames", flush=True)


_HEADNEG = None


def headneg() -> np.ndarray:
    global _HEADNEG
    if _HEADNEG is None:
        _HEADNEG = np.load(HEADNEG / "feat.npy")
    return _HEADNEG


def run_task(payload: tuple) -> dict:
    seed, shots, task, methods = payload
    from methods import Positives
    from methods_extra import evaluate_wake_extra

    started = time.time()
    folder = run_mswc.stream_dir(seed)
    rng_pos = np.random.default_rng(seed * 100 + task["index"] + 50)
    pos, spans = common.stream([common.read(link) for link in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
    pos_noisy = common.add_noise(pos, run_mswc.SNR_DB, seed=seed * 100 + task["index"] + 7)
    positives = {"clean": Positives(pos, spans), "noisy": Positives(pos_noisy, spans)}
    room = {"emb": np.load(folder / "room_feat.npy")}
    clips = [common.read(link) for link in task["enroll"][:shots]]
    kwargs = {"methods": methods} if methods else {}
    result = evaluate_wake_extra(clips, positives, run_mswc.negatives(seed), room, headneg(),
                                 np.random.default_rng(seed * 100 + task["index"]), **kwargs)
    result.update({"word": task["word"], "seed": seed, "shots": shots, "seconds": round(time.time() - started, 1)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--shots", default="10,6,3")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--methods", default="", help="only these methods, merged into the existing task files")
    args = parser.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    prepare_headneg(args.jobs)
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        tasks = common.load_json(run_mswc.OUT / f"plan_seed{seed}.json")
        for k in [int(s) for s in args.shots.split(",")]:
            for task in tasks:
                path = tasks_dir / f"seed{seed}_k{k}_w{task['index']:02d}.json"
                if methods:
                    if path.is_file() and not set(methods) <= set(json.loads(path.read_text())["methods"]):
                        work.append((path, (seed, k, task, methods)))
                elif not path.is_file():
                    work.append((path, (seed, k, task, None)))
    print(f"{len(work)} tasks to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, payload): path for path, payload in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = futures[future]
            original = json.loads((run_mswc.OUT / "tasks" / path.name).read_text())
            assert result["enrolled"] == original["enrolled"], (path.name, result["enrolled"], original["enrolled"])
            if methods:                    # merge the new methods into the existing file
                merged = json.loads(path.read_text())
                merged["methods"].update(result["methods"])
                merged["head_info"] = {**merged.get("head_info", {}), **result.get("head_info", {})}
                result = {**merged, "seconds": result["seconds"]}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False))
            tmp.replace(path)
            print(f"[{done}/{len(work)}] seed {result['seed']} k {result['shots']} {result['word']}: "
                  f"{len(result['methods'])} methods, {result['seconds']}s (elapsed {time.time() - started:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
