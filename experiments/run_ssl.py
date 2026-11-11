"""Baseline after review: frame features of a multilingual self-supervised model with the same subsequence DTW.

Model: mHuBERT-147 (Zanon Boito et al., 2024; Hugging Face 'utter-project/mHuBERT-147', a 95M-parameter HuBERT
base model trained on 147 languages). Frames: the hidden state of one transformer layer, 20 ms frames averaged
in pairs to 40 ms, projected to 128 dimensions by a PCA fitted on the shipped bank only. Features are computed
offline over 30-s windows with 1.28 s of context on each side, so the baseline sees more context than a
streaming detector would (an advantage for the baseline).

The layer is chosen before any evaluation data are touched ('select' step): on 150 words of bank speakers
saying bank words that are neither in the bank nor in any evaluation material, each word said by two
different speakers, we compute the subsequence-DTW distance between the template of one recording and the
padded other recording (same word) and between templates and recordings of different words, and keep the
layer with the highest ROC AUC.

Enrollment, templates and scoring follow the MFCC + DTW variant of methods.py: the padded enrollment clip
(same random noise floor as methods.enroll_clips), the deployed energy rule for onset and offset, template =
frames whose centre lies between onset and offset (at least 3), normaliser on the enrollment clips plus the
room audio, wake score = mean of the two best template distances, 1.2 s refractory (30 frames).

Steps (each resumable, results written incrementally):
  select   results/ssl/layer_selection.json
  bank     results/ssl/bank_feat.npy (fp16), bank_ends.npy, pca.npz
  streams  results/ssl/streams/seed{s}/{room,neg_clean}_{feat,ends}.npy
  tasks    results/ssl/feats/seed{s}_w{idx}.npz (positives and the 10 padded enrollment clips)
  score    results/ssl/tasks/seed{s}_k{k}_w{idx}.json (method 'ssl-dtw'; same format as results/mswc/tasks)

Usage: python experiments/run_ssl.py [--procs 4] [--threads 2] [--steps select,bank,streams,tasks,score]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import run_mswc  # noqa: E402
from common import SR  # noqa: E402

OUT = common.RESULTS / "ssl"
MODEL = Path("/home/ali/.cache/huggingface/hub/models--utter-project--mHuBERT-147/snapshots/"
             "7ad3fc0bc5106c58c9c13526abccad527150d135")
HOP20 = 320
HOP = 640                     # 40 ms pooled frames
RECEPTIVE = 400
CHUNK_SECONDS = 30.0
CONTEXT = 2 * HOP * 16        # 1.28 s, a multiple of 640 samples
PCA_DIM = 128
REFRACTORY = 30               # 1.2 s at 40 ms
CANDIDATE_LAYERS = (4, 6, 8, 9, 10, 12)
THREADS = int(os.environ.get("SSL_THREADS", "2"))

_MODEL = None
_PCA = None


def model(layers: int | None = None):
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import HubertModel

        torch.set_num_threads(THREADS)
        m = HubertModel.from_pretrained(str(MODEL))
        m.eval()
        if layers is not None:
            m.encoder.layers = m.encoder.layers[:layers]
        _MODEL = m
    return _MODEL


def pca():
    global _PCA
    if _PCA is None and (OUT / "pca.npz").is_file():
        data = np.load(OUT / "pca.npz")
        _PCA = (data["mean"], data["components"])
    return _PCA


def chosen_layer() -> int:
    return int(common.load_json(OUT / "layer_selection.json")["layer"])


def hidden(values: np.ndarray, layer: int) -> np.ndarray:
    """Hidden states (frames x 768) of one layer at 20 ms for audio in int16 scale."""

    import torch

    x = np.asarray(values, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.inference_mode():
        out = model(layer)(torch.from_numpy(x)[None], output_hidden_states=True)
    return out.hidden_states[layer][0].numpy()


def _part(args):
    path, start, end, total, layer, project = args
    pcm = np.load(path, mmap_mode="r")
    a = max(0, start - CONTEXT)
    b = min(total, end + CONTEXT)
    h = hidden(np.asarray(pcm[a:b], dtype=np.float32), layer)
    g = np.arange(h.shape[0]) + a // HOP20
    h = h[(g >= start // HOP20) & (g < end // HOP20)]
    h = h[: h.shape[0] - h.shape[0] % 2]
    pooled = 0.5 * (h[0::2] + h[1::2])
    if project:
        mean, comp = pca()
        pooled = (pooled - mean) @ comp.T
    return pooled.astype(np.float32)


def features(values: np.ndarray, layer: int, pool: ProcessPoolExecutor | None, scratch: Path,
             project: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """40-ms SSL features of an audio array (int16 scale) and the end sample of each frame."""

    pcm = common.to_pcm16(values)
    total = pcm.size
    step = int(CHUNK_SECONDS * SR) // HOP * HOP
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
    feats = np.vstack(parts) if parts else np.zeros((0, PCA_DIM if project else 768), np.float32)
    ends = np.arange(feats.shape[0], dtype=np.int64) * HOP + HOP20 + RECEPTIVE
    keep = ends <= total
    return feats[keep], ends[keep]


def padded_enrollment(clips: list[np.ndarray], rng: np.random.Generator):
    """Same padding and energy rule as methods.enroll_clips (and the same random draws)."""

    from nexawake.enroll import frame_rms, sound_bounds

    out = []
    for clip in clips:
        padded = np.concatenate([common.floor(rng, SR), clip + common.floor(rng, clip.size),
                                 common.floor(rng, int(0.45 * SR))])
        noise = float(np.percentile(frame_rms(padded), 10))
        out.append((padded, sound_bounds(padded, noise)))
    return out


def template(feats: np.ndarray, onset: int, offset: int) -> np.ndarray:
    centres = np.arange(feats.shape[0]) * HOP + 360          # centre of the two 20-ms frames (25 ms windows)
    idx = np.nonzero((centres >= onset) & (centres <= offset))[0]
    if idx.size < 3:
        first = int(idx[0]) if idx.size else int(np.searchsorted(centres, onset))
        idx = np.arange(first, min(feats.shape[0], first + 3))
    return feats[idx]


# ------------------------------------------------------------------ steps
def step_select(pool) -> None:
    path = OUT / "layer_selection.json"
    if path.is_file():
        return
    from nexawake import fast

    table = pd.read_csv(common.MSWC / "test.csv")
    split = common.load_json(common.RESULTS / "splits.json")
    headneg = set(common.load_json(common.RESULTS / "headneg" / "info.json")["links"])
    used = set(split["bank"]) | headneg | {l for v in split["seeds"].values() for l in v["room"] + v["negatives"]}
    rows = table[table.SPEAKER.isin(set(split["bank_speakers"])) & table.WORD.isin(set(split["bank_words"]))
                 & ~table.LINK.isin(used)]
    rng = np.random.default_rng(21)
    words = [w for w, g in rows.groupby("WORD") if g.SPEAKER.nunique() >= 2]
    rng.shuffle(words)
    words = words[:150]
    pairs = []
    for w in words:
        g = rows[rows.WORD == w].drop_duplicates("SPEAKER").sample(n=2, random_state=int(rng.integers(1 << 30)))
        pairs.append(g.LINK.tolist())
    enroll = padded_enrollment([common.read(l) for pair in pairs for l in pair], np.random.default_rng(22))
    layers = {}
    for padded, _b in enroll:
        import torch

        x = np.asarray(padded, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        with torch.inference_mode():
            hs = model()(torch.from_numpy(x)[None], output_hidden_states=True).hidden_states
        for layer in CANDIDATE_LAYERS:
            h = hs[layer][0].numpy()
            h = h[: h.shape[0] - h.shape[0] % 2]
            layers.setdefault(layer, []).append(0.5 * (h[0::2] + h[1::2]))
    from sklearn.metrics import roc_auc_score

    result = {"words": len(words), "recordings": len(enroll), "auc": {}}
    n = len(words)
    for layer, feats in layers.items():
        norm = common.fit_normalizer(feats)
        temps = [template(f, b[0], b[1]) if b else None for f, b in zip(feats, (e[1] for e in enroll))]
        same, diff = [], []
        for i in range(n):
            a, b = 2 * i, 2 * i + 1
            if temps[a] is None:
                continue
            score = lambda t, j: float(np.min(fast.template_scores(feats[j], [t], norm)[:, 0]))  # noqa: E731
            same.append(score(temps[a], b))
            for j in rng.choice([x for x in range(n) if x != i], size=4, replace=False):
                diff.append(score(temps[a], 2 * int(j) + 1))
        y = np.r_[np.ones(len(same)), np.zeros(len(diff))]
        s = -np.r_[same, diff]
        s[~np.isfinite(s)] = -10.0
        result["auc"][str(layer)] = float(roc_auc_score(y, s))
        print(f"layer {layer}: AUC {result['auc'][str(layer)]:.3f} ({len(same)} same, {len(diff)} different)",
              flush=True)
    result["layer"] = int(max(result["auc"], key=lambda k: result["auc"][k]))
    common.dump_json(path, result)


def step_bank(pool) -> None:
    if (OUT / "pca.npz").is_file() and (OUT / "bank_feat.npy").is_file():
        return
    layer = chosen_layer()
    split = common.load_json(common.RESULTS / "splits.json")
    values, _spans = common.stream([common.read(link) for link in split["bank"]], np.random.default_rng(10),
                                   common.GAP_SECONDS)
    raw, ends = features(values, layer, pool, OUT / "scratch", project=False)
    mean = raw.mean(axis=0)
    _u, _s, vt = np.linalg.svd(raw[::2] - mean, full_matrices=False)
    comp = vt[:PCA_DIM]
    np.savez(OUT / "pca.npz", mean=mean, components=comp, explained=(_s[:PCA_DIM] ** 2).sum() / (_s ** 2).sum())
    global _PCA
    _PCA = (mean, comp)
    np.save(OUT / "bank_feat.npy", ((raw - mean) @ comp.T).astype(np.float16))
    np.save(OUT / "bank_ends.npy", ends)
    print(f"bank: {raw.shape[0]} frames, PCA {PCA_DIM} dims keep {(_s[:PCA_DIM] ** 2).sum() / (_s ** 2).sum():.3f}",
          flush=True)


def step_streams(pool, seeds) -> None:
    layer = chosen_layer()
    for seed in seeds:
        folder = OUT / "streams" / f"seed{seed}"
        if (folder / "neg_clean_feat.npy").is_file():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        split = common.load_json(common.RESULTS / "splits.json")["seeds"][str(seed)]
        rng = np.random.default_rng(seed + 1000)              # as run_mswc.prepare
        room, _ = common.stream([common.read(link) for link in split["room"]], rng, common.GAP_SECONDS)
        clean, _spans = common.stream([common.read(link) for link in split["negatives"]], rng, common.GAP_SECONDS)
        f, e = features(room, layer, pool, folder)
        np.save(folder / "room_feat.npy", f.astype(np.float16))
        np.save(folder / "room_ends.npy", e)
        started = time.time()
        f, e = features(clean, layer, pool, folder)
        np.save(folder / "neg_clean_ends.npy", e)
        np.save(folder / "neg_clean_feat.npy", f.astype(np.float16))
        print(f"seed {seed}: stream {f.shape[0]} frames in {time.time() - started:.0f}s", flush=True)


def step_tasks(pool, seeds) -> None:
    layer = chosen_layer()
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for task in common.load_json(run_mswc.OUT / f"plan_seed{seed}.json"):
            path = OUT / "feats" / f"seed{seed}_w{task['index']:02d}.npz"
            if path.is_file():
                continue
            rng_pos = np.random.default_rng(seed * 100 + task["index"] + 50)      # as run_mswc.run_task
            pos, spans = common.stream([common.read(l) for l in task["positives"]], rng_pos, common.POS_PAD_SECONDS)
            pf, pe = features(pos, layer, pool, OUT / "scratch")
            enroll = padded_enrollment([common.read(l) for l in task["enroll"]],
                                       np.random.default_rng(seed * 100 + task["index"]))
            arrays = {"pos_feat": pf, "pos_ends": pe, "pos_spans": np.asarray(spans, dtype=np.int64)}
            items = list(pool.map(_enroll_part, [(p, layer) for p, _b in enroll]))
            for i, ((_p, b), f) in enumerate(zip(enroll, items)):
                arrays[f"enroll{i}"] = f
                arrays[f"bounds{i}"] = np.asarray(b if b else (-1, -1), dtype=np.int64)
            np.savez(path, **arrays)
            print(f"seed {seed} {task['word']}: features saved", flush=True)


def _enroll_part(args):
    padded, layer = args
    feats, _ends = features(padded, layer, None, OUT / "scratch")
    return feats


def score_task(args) -> dict:
    seed, shots, index, word = args
    from nexawake import fast

    data = np.load(OUT / "feats" / f"seed{seed}_w{index:02d}.npz")
    bank = np.load(OUT / "bank_feat.npy").astype(np.float32)
    folder = OUT / "streams" / f"seed{seed}"
    neg = np.load(folder / "neg_clean_feat.npy").astype(np.float32)
    room = np.load(folder / "room_feat.npy").astype(np.float32)
    enroll = [(data[f"enroll{i}"], tuple(data[f"bounds{i}"])) for i in range(shots)]
    accepted = [(f, b) for f, b in enroll if b[0] >= 0]
    out = {"word": word, "seed": seed, "shots": shots, "enrolled": len(accepted), "positives": {"clean": 60},
           "negative_hours": {"eval_clean": neg.shape[0] * HOP / SR / 3600, "bank_clean": bank.shape[0] * HOP / SR / 3600},
           "prefix_hours": {"eval_clean": {}, "bank_clean": {"15min": 0.25, "30min": 0.5}}, "methods": {}}
    if len(accepted) < 3:
        return out
    temps = [template(f, b[0], b[1]) for f, b in accepted]
    norm = common.fit_normalizer([f for f, _b in accepted] + [room])
    fn = lambda x: fast.wake_score(fast.template_scores(x, temps, norm))  # noqa: E731
    spans = [tuple(s) for s in data["pos_spans"]]
    entry = {"threshold": None, "calibration": None, "pos": {}, "neg": {}, "prefix": {}}
    entry["pos"]["clean"] = common.window_min(fn(data["pos_feat"]), data["pos_ends"], spans)
    entry["neg"]["eval_clean"] = common.events(fn(neg), REFRACTORY).tolist()
    bank_scores = fn(bank)
    entry["neg"]["bank_clean"] = common.events(bank_scores, REFRACTORY).tolist()
    per_second = SR / HOP
    entry["prefix"]["bank_clean"] = {p: common.events(bank_scores[: int(s * per_second)], REFRACTORY).tolist()
                                     for p, s in run_mswc.BANK_PREFIXES.items()}
    out["methods"]["ssl-dtw"] = entry
    out["template_frames"] = [int(len(t)) for t in temps]
    return out


def step_score(seeds, shots, procs) -> None:
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)
    work = []
    for seed in seeds:
        for task in common.load_json(run_mswc.OUT / f"plan_seed{seed}.json"):
            for k in shots:
                path = OUT / "tasks" / f"seed{seed}_k{k}_w{task['index']:02d}.json"
                if not path.is_file():
                    work.append((path, (seed, k, task["index"], task["word"])))
    with ProcessPoolExecutor(procs) as pool:
        for (path, _a), result in zip(work, pool.map(score_task, [a for _p, a in work])):
            original = json.loads((run_mswc.OUT / "tasks" / path.name).read_text())
            assert result["enrolled"] == original["enrolled"], (path.name, result["enrolled"], original["enrolled"])
            path.write_text(json.dumps(result, ensure_ascii=False))
            print(f"scored {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--steps", default="select,bank,streams,tasks,score")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--shots", default="3,6,10")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    steps = args.steps.split(",")
    if "select" in steps:
        step_select(None)
    with ProcessPoolExecutor(args.procs) as pool:
        if "bank" in steps:
            step_bank(pool)
        if "streams" in steps:
            step_streams(pool, seeds)
        if "tasks" in steps:
            step_tasks(pool, seeds)
    if "score" in steps:
        step_score(seeds, [int(s) for s in args.shots.split(",")], args.procs * 2)


if __name__ == "__main__":
    main()
