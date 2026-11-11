"""Same-speaker wakes, version 2: the speaker's own clips are masked out of the 4-h seed-1 stream.

In run_samespk.py the 'other speakers' negatives were the seed-1 evaluation stream as it is, and every
same-speaker speaker has clips in that stream (1,844 clips for the 66 main-set speakers). Event counts cannot
be split afterwards, so this script rescores the stream with the enrolled speaker's own clips masked
(methods.mask_frames: the clip plus 2 s after it), for every original variant, and also runs the variants of
methods_extra.py on all same-speaker negatives (masked seed-1 stream, bank, own other words).

Enrollment, positives, own-voice stream and random seeds are those of run_samespk.run_task, so the wakes are
the same wakes; the script checks that the recomputed positive scores equal the stored ones.

Output: results/samespk_v2/tasks/<same names>.json = the original task with
  * negative_hours['eval_clean'] and every method's neg['eval_clean'] replaced by the masked stream,
  * the noisy condition (eval_noisy, noisy positives) removed (it is not used in the paper),
  * the methods of methods_extra.py added,
  * 'masked_clips' and 'masked_hours' recorded.

Usage: python experiments/run_samespk_v2.py [--jobs 8]
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
import run_extra  # noqa: E402
import run_mswc  # noqa: E402
import run_samespk  # noqa: E402

SRC = common.RESULTS / "samespk"
OUT = common.RESULTS / "samespk_v2"

_STREAM = None


def seed_stream():
    global _STREAM
    if _STREAM is None:
        folder = run_mswc.stream_dir(run_samespk.NEG_SEED)
        split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(run_samespk.NEG_SEED)]
        _STREAM = {"emb": np.load(folder / "neg_clean_feat.npy", mmap_mode="r"),
                   "ends": np.load(folder / "neg_clean_ends.npy"),
                   "mfcc": np.load(folder / "neg_clean_mfcc.npy", mmap_mode="r"),
                   "spans": np.load(folder / "neg_spans.npy"),
                   "speakers": split["negative_speakers"]}
    return _STREAM


def run_task(task: dict) -> dict:
    from methods import NegStream, Positives, evaluate_wake
    from methods_extra import evaluate_wake_extra

    started = time.time()
    folder = run_mswc.stream_dir(run_samespk.NEG_SEED)
    s = seed_stream()
    mask = [(int(a), int(b)) for (a, b), spk in zip(s["spans"], s["speakers"]) if spk == task["speaker"]]
    masked = NegStream(s["emb"], s["ends"], s["mfcc"], mask=mask)
    rng_pos = np.random.default_rng(9000 + task["index"])
    pos, spans = common.stream([common.read(link) for link in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
    positives = {"clean": Positives(pos, spans)}
    room = {"emb": np.load(folder / "room_feat.npy"), "mfcc": np.load(folder / "room_mfcc.npy")}
    clips = [common.read(link) for link in task["enroll"]]
    old = evaluate_wake(clips, positives, {"eval_clean": masked}, room, run_mswc.bank()["clean"][0],
                        np.random.default_rng(8000 + task["index"]), methods=run_samespk.METHODS)
    own, _ = common.stream([common.read(link) for link in task["own"]], np.random.default_rng(9900 + task["index"]),
                           common.GAP_SECONDS)
    own_pcm = common.to_pcm16(own)
    emb, ends = common.models().clip_features(own_pcm.tobytes())
    negs = run_mswc.negatives(run_samespk.NEG_SEED)
    negs.pop("eval_noisy")
    negs["eval_clean"] = masked
    negs["own_clean"] = NegStream(emb, ends)
    new = evaluate_wake_extra(clips, positives, negs, room, run_extra.headneg(),
                              np.random.default_rng(8000 + task["index"]))
    return {"old": old, "new": new, "masked_clips": len(mask), "seconds": round(time.time() - started, 1)}


def assemble(path: Path, result: dict) -> dict:
    original = json.loads((SRC / "tasks" / path.name).read_text())
    old, new = result["old"], result["new"]
    assert old["enrolled"] == original["enrolled"] == new["enrolled"], path.name
    out = dict(original)
    out["negative_hours"] = {k: v for k, v in original["negative_hours"].items() if k != "eval_noisy"}
    out["prefix_hours"] = {k: v for k, v in original["prefix_hours"].items() if k != "eval_noisy"}
    out["positives"] = {"clean": original["positives"]["clean"]}
    out["negative_hours"]["eval_clean"] = old["negative_hours"]["eval_clean"]
    out["masked_clips"] = result["masked_clips"]
    out["masked_hours"] = original["negative_hours"]["eval_clean"] - old["negative_hours"]["eval_clean"]
    methods = {}
    for name, entry in original["methods"].items():
        fresh = old["methods"][name]
        assert np.allclose(fresh["pos"]["clean"], entry["pos"]["clean"]), (path.name, name)
        e = dict(entry)
        e["pos"] = {"clean": entry["pos"]["clean"]}
        e["neg"] = {k: v for k, v in entry["neg"].items() if k != "eval_noisy"}
        e["neg"]["eval_clean"] = fresh["neg"]["eval_clean"]
        e["prefix"] = {k: v for k, v in entry.get("prefix", {}).items() if k != "eval_noisy"}
        methods[name] = e
    for name, entry in new["methods"].items():
        assert name not in methods
        methods[name] = entry
    out["methods"] = methods
    out["head_info"] = new.get("head_info", {})
    out["seconds_v2"] = result["seconds"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tasks = common.load_json(SRC / "plan.json")
    tasks.sort(key=lambda t: (t["set"] != "main", -t["shots"], t["index"]))
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for t in tasks:
        path = tasks_dir / f"{t['set']}_k{t['shots']}_s{t['index']:03d}.json"
        if path.is_file():
            continue
        original = json.loads((SRC / "tasks" / path.name).read_text())
        if not original["methods"]:          # wake could not be built: copy as it is
            path.write_text(json.dumps(original, ensure_ascii=False))
            continue
        work.append((path, t))
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work)} tasks to run", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, t): p for p, t in work}
        for done, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            merged = assemble(path, future.result())
            path.write_text(json.dumps(merged, ensure_ascii=False))
            print(f"[{done}/{len(work)}] {path.name}: masked {merged['masked_clips']} clips "
                  f"({merged['masked_hours'] * 60:.1f} min), {len(merged['methods'])} methods, "
                  f"{merged['seconds_v2']}s (elapsed {time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
