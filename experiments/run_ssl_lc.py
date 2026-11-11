"""Limited-context control for the mHuBERT-147 + DTW baseline (added after review), seed 1 only.

run_ssl.py computes the SSL features offline over 30-s windows with 1.28 s of context on each side, which a
streaming detector cannot do. Here every 0.32-s block of 40-ms frames is computed from its own short window:
the block itself, the 0.96 s before it and 5 ms after it (the 25-ms receptive field of the last 20-ms frame).
A frame therefore sees 0.96-1.28 s of past audio and at most 0.32 s of future audio (0.16 s on average), as a
detector that runs the model every 0.32 s on a 1.28-s buffer would. Each window is normalised on its own, as
the model's feature extractor normalises one input.

Everything else is run_ssl.py: layer 6 (results/ssl/layer_selection.json), 20-ms frames averaged in pairs,
128-d PCA (refitted here on the limited-context bank features), templates from onset to offset, normaliser on
the enrollment clips plus the room audio, mean of the two best template distances, 1.2 s refractory.
Same enrollment clips, padding noise, positives and negative streams as run_ssl.py (same random seeds).

Steps (resumable; results written incrementally):
  bank     results/ssl_lc/bank_feat.npy (fp16), bank_ends.npy, pca.npz
  streams  results/ssl_lc/streams/seed1/{room,neg_clean}_{feat,ends}.npy
  tasks    results/ssl_lc/feats/seed1_w<idx>.npz
  score    results/ssl_lc/tasks/seed1_k<k>_w<idx>.json (method 'ssl-dtw-lc'; format of results/ssl/tasks)

Usage: SSL_THREADS=2 python experiments/run_ssl_lc.py [--procs 3] [--steps bank,streams,tasks,score]
"""

from __future__ import annotations

import argparse
import multiprocessing
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
import run_ssl  # noqa: E402
from common import SR  # noqa: E402

OUT = common.RESULTS / "ssl_lc"
HOP20, HOP, RECEPTIVE = run_ssl.HOP20, run_ssl.HOP, run_ssl.RECEPTIVE
BLOCK = 8 * HOP                 # 0.32 s of output frames per model call
LEFT = 24 * HOP                 # 0.96 s of past audio
TAIL = RECEPTIVE - HOP20        # 80 samples: the receptive field of the last 20-ms frame of a block
BATCH = 24
SEED = 1
METHOD = "ssl-dtw-lc"

_PCA = None


def pca():
    global _PCA
    if _PCA is None and (OUT / "pca.npz").is_file():
        data = np.load(OUT / "pca.npz")
        _PCA = (data["mean"], data["components"])
    return _PCA


def _hidden_batch(windows: list[np.ndarray], layer: int) -> list[np.ndarray]:
    import torch

    out = []
    by_len = {}
    for i, w in enumerate(windows):
        by_len.setdefault(w.size, []).append(i)
    result = [None] * len(windows)
    for _n, idx in by_len.items():
        for j in range(0, len(idx), BATCH):
            part = idx[j: j + BATCH]
            x = np.stack([windows[i] for i in part]).astype(np.float32)
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-7)
            with torch.inference_mode():
                hs = run_ssl.model(layer)(torch.from_numpy(x), output_hidden_states=True).hidden_states[layer]
            for i, h in zip(part, hs.numpy()):
                result[i] = h
    out.extend(result)
    return out


def _part(args):
    """Frames of [start, end) (sample indices, start a multiple of 640), each block from its own window."""

    path, start, end, total, layer, project = args
    pcm = np.load(path, mmap_mode="r")
    blocks = [(s, min(end, s + BLOCK)) for s in range(start, end, BLOCK)]
    windows, offsets = [], []
    for s, e in blocks:
        a = max(0, s - LEFT)
        b = min(total, e + TAIL)
        windows.append(np.asarray(pcm[a:b], dtype=np.float32))
        offsets.append(a)
    hs = _hidden_batch(windows, layer)
    frames = []
    for (s, e), a, h in zip(blocks, offsets, hs):
        g = np.arange(h.shape[0]) + a // HOP20
        frames.append(h[(g >= s // HOP20) & (g < e // HOP20)])
    h = np.vstack(frames)
    h = h[: h.shape[0] - h.shape[0] % 2]
    pooled = 0.5 * (h[0::2] + h[1::2])
    if project:
        mean, comp = pca()
        pooled = (pooled - mean) @ comp.T
    return pooled.astype(np.float32)


def features(values: np.ndarray, layer: int, pool, scratch: Path, project: bool = True):
    """As run_ssl.features, with limited-context frames; same frame grid and end samples."""

    pcm = common.to_pcm16(values)
    total = pcm.size
    step = int(run_ssl.CHUNK_SECONDS * SR) // HOP * HOP
    usable = total - (total - RECEPTIVE) % HOP if total > RECEPTIVE else 0
    bounds = list(range(0, usable, step))
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"pcm_{os.getpid()}_{id(values)}.npy"
    np.save(path, pcm)
    jobs = [(str(path), s, min(usable, s + step), total, layer, project) for s in bounds]
    try:
        parts = list(pool.map(_part, jobs)) if pool is not None and len(jobs) > 1 else [_part(j) for j in jobs]
    finally:
        path.unlink(missing_ok=True)
    feats = np.vstack(parts) if parts else np.zeros((0, run_ssl.PCA_DIM if project else 768), np.float32)
    ends = np.arange(feats.shape[0], dtype=np.int64) * HOP + HOP20 + RECEPTIVE
    keep = ends <= total
    return feats[keep], ends[keep]


def _enroll_part(args):
    padded, layer = args
    feats, _ends = features(padded, layer, None, OUT / "scratch")
    return feats


def step_bank(pool) -> None:
    if (OUT / "pca.npz").is_file() and (OUT / "bank_feat.npy").is_file():
        return
    layer = run_ssl.chosen_layer()
    split = common.load_json(common.RESULTS / "splits.json")
    values, _spans = common.stream([common.read(link) for link in split["bank"]], np.random.default_rng(10),
                                   common.GAP_SECONDS)
    started = time.time()
    raw, ends = features(values, layer, pool, OUT / "scratch", project=False)
    mean = raw.mean(axis=0)
    _u, s, vt = np.linalg.svd(raw[::2] - mean, full_matrices=False)
    comp = vt[: run_ssl.PCA_DIM]
    kept = float((s[: run_ssl.PCA_DIM] ** 2).sum() / (s ** 2).sum())
    np.savez(OUT / "pca.npz", mean=mean, components=comp, explained=kept)
    global _PCA
    _PCA = (mean, comp)
    np.save(OUT / "bank_feat.npy", ((raw - mean) @ comp.T).astype(np.float16))
    np.save(OUT / "bank_ends.npy", ends)
    print(f"bank: {raw.shape[0]} frames, PCA keeps {kept:.3f}, {time.time() - started:.0f}s", flush=True)


def step_streams(pool) -> None:
    layer = run_ssl.chosen_layer()
    folder = OUT / "streams" / f"seed{SEED}"
    if (folder / "neg_clean_feat.npy").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(SEED)]
    rng = np.random.default_rng(SEED + 1000)              # as run_mswc.prepare and run_ssl.step_streams
    room, _ = common.stream([common.read(link) for link in split["room"]], rng, common.GAP_SECONDS)
    clean, _spans = common.stream([common.read(link) for link in split["negatives"]], rng, common.GAP_SECONDS)
    f, e = features(room, layer, pool, folder)
    np.save(folder / "room_feat.npy", f.astype(np.float16))
    np.save(folder / "room_ends.npy", e)
    started = time.time()
    f, e = features(clean, layer, pool, folder)
    np.save(folder / "neg_clean_ends.npy", e)
    np.save(folder / "neg_clean_feat.npy", f.astype(np.float16))
    print(f"seed {SEED}: stream {f.shape[0]} frames in {time.time() - started:.0f}s", flush=True)


def step_tasks(pool) -> None:
    layer = run_ssl.chosen_layer()
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    for task in common.load_json(run_mswc.OUT / f"plan_seed{SEED}.json"):
        path = OUT / "feats" / f"seed{SEED}_w{task['index']:02d}.npz"
        if path.is_file():
            continue
        rng_pos = np.random.default_rng(SEED * 100 + task["index"] + 50)      # as run_ssl.step_tasks
        pos, spans = common.stream([common.read(l) for l in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
        pf, pe = features(pos, layer, pool, OUT / "scratch")
        enroll = run_ssl.padded_enrollment([common.read(l) for l in task["enroll"]],
                                           np.random.default_rng(SEED * 100 + task["index"]))
        arrays = {"pos_feat": pf, "pos_ends": pe, "pos_spans": np.asarray(spans, dtype=np.int64)}
        items = list(pool.map(_enroll_part, [(p, layer) for p, _b in enroll]))
        for i, ((_p, b), f) in enumerate(zip(enroll, items)):
            arrays[f"enroll{i}"] = f
            arrays[f"bounds{i}"] = np.asarray(b if b else (-1, -1), dtype=np.int64)
        np.savez(path, **arrays)
        print(f"seed {SEED} {task['word']}: features saved", flush=True)


def score_task(args) -> dict:
    shots, index, word = args
    from nexawake import fast

    data = np.load(OUT / "feats" / f"seed{SEED}_w{index:02d}.npz")
    bank = np.load(OUT / "bank_feat.npy").astype(np.float32)
    folder = OUT / "streams" / f"seed{SEED}"
    neg = np.load(folder / "neg_clean_feat.npy").astype(np.float32)
    room = np.load(folder / "room_feat.npy").astype(np.float32)
    enroll = [(data[f"enroll{i}"], tuple(data[f"bounds{i}"])) for i in range(shots)]
    accepted = [(f, b) for f, b in enroll if b[0] >= 0]
    out = {"word": word, "seed": SEED, "shots": shots, "enrolled": len(accepted), "positives": {"clean": 60},
           "negative_hours": {"eval_clean": neg.shape[0] * HOP / SR / 3600, "bank_clean": bank.shape[0] * HOP / SR / 3600},
           "prefix_hours": {"eval_clean": {}, "bank_clean": {"15min": 0.25, "30min": 0.5}}, "methods": {},
           "context": {"left_s": LEFT / SR, "block_s": BLOCK / SR, "lookahead_max_s": (BLOCK + TAIL) / SR}}
    if len(accepted) < 3:
        return out
    temps = [run_ssl.template(f, b[0], b[1]) for f, b in accepted]
    norm = common.fit_normalizer([f for f, _b in accepted] + [room])
    fn = lambda x: fast.wake_score(fast.template_scores(x, temps, norm))  # noqa: E731
    spans = [tuple(s) for s in data["pos_spans"]]
    entry = {"threshold": None, "calibration": None, "pos": {}, "neg": {}, "prefix": {}}
    entry["pos"]["clean"] = common.window_min(fn(data["pos_feat"]), data["pos_ends"], spans)
    entry["neg"]["eval_clean"] = common.events(fn(neg), run_ssl.REFRACTORY).tolist()
    bank_scores = fn(bank)
    entry["neg"]["bank_clean"] = common.events(bank_scores, run_ssl.REFRACTORY).tolist()
    per_second = SR / HOP
    entry["prefix"]["bank_clean"] = {p: common.events(bank_scores[: int(s * per_second)], run_ssl.REFRACTORY).tolist()
                                     for p, s in run_mswc.BANK_PREFIXES.items()}
    out["methods"][METHOD] = entry
    out["template_frames"] = [int(len(t)) for t in temps]
    return out


def step_score(procs: int) -> None:
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)
    work = []
    for task in common.load_json(run_mswc.OUT / f"plan_seed{SEED}.json"):
        for k in (3, 6, 10):
            path = OUT / "tasks" / f"seed{SEED}_k{k}_w{task['index']:02d}.json"
            if not path.is_file():
                work.append((path, (k, task["index"], task["word"])))
    with ProcessPoolExecutor(procs) as pool:
        for (path, _a), result in zip(work, pool.map(score_task, [a for _p, a in work])):
            original = json.loads((run_mswc.OUT / "tasks" / path.name).read_text())
            assert result["enrolled"] == original["enrolled"], (path.name, result["enrolled"], original["enrolled"])
            path.write_text(json.dumps(result, ensure_ascii=False))
            print(f"scored {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--steps", default="bank,streams,tasks,score")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    steps = args.steps.split(",")
    # 'spawn': the main process may run the model itself (a stream shorter than one 30-s chunk), and
    # workers forked after that deadlock in torch's thread pool.
    with ProcessPoolExecutor(args.procs, mp_context=multiprocessing.get_context("spawn")) as pool:
        if "bank" in steps:
            step_bank(pool)
        if "streams" in steps:
            step_streams(pool)
        if "tasks" in steps:
            step_tasks(pool)
    if "score" in steps:
        step_score(args.procs * 2)


if __name__ == "__main__":
    main()
