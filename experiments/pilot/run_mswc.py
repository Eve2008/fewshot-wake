"""Few-shot wake detection on Persian MSWC (test split).

For each seed and target word: enroll k recordings, calibrate on 20 s of
other speech, then stream (a) held-out recordings of the target by speakers
never enrolled and (b) a ~30 min stream of other words through each method.
Scores are kept per frame, so false alarms per hour and false-reject rates
can be computed for any threshold afterwards (summarize.py).

Usage: python experiments/run_mswc.py --mswc DATA/mswc --models DATA/models --out results/mswc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

TARGETS = ["برای", "خیلی", "باید", "نیست", "اینجا", "دوست", "وقتی", "چیزی", "آنها", "وجود", "حالا", "الان",
           "خونه", "درست", "خواهم", "کردم"]
SR = 16000
GAP_SECONDS = 0.25
POS_PAD_SECONDS = 1.0
FLOOR_RMS = 20.0
NEGATIVE_CLIPS = 1800
ROOM_CLIPS = 20
MAX_POSITIVES = 60

_MODELS = None


def models(directory: str):
    global _MODELS
    if _MODELS is None:
        from nexawake.features import EmbeddingModels

        _MODELS = EmbeddingModels(Path(directory))
        _MODELS.load()
    return _MODELS


def wav_path(mswc: Path, link: str) -> Path:
    word, name = link.split("/", 1)
    return mswc / "audio" / f"{word}_{name.replace('.opus', '.wav')}"


def read(mswc: Path, link: str) -> np.ndarray:
    data, rate = sf.read(wav_path(mswc, link), dtype="float32")
    assert rate == SR, rate
    return data * 32767.0


def floor(rng: np.random.Generator, seconds: float) -> np.ndarray:
    return rng.normal(0, FLOOR_RMS, int(seconds * SR)).astype(np.float32)


def stream(clips: list[np.ndarray], rng: np.random.Generator, pad: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    parts, spans, cursor = [floor(rng, pad)], [], int(pad * SR)
    for clip in clips:
        spans.append((cursor, cursor + clip.size))
        parts += [clip + floor(rng, clip.size / SR)[: clip.size], floor(rng, pad)]
        cursor += clip.size + int(pad * SR)
    return np.concatenate(parts), spans


def add_noise(values: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    levels = values[: values.size - values.size % 320].reshape(-1, 320)
    signal = float(np.percentile(np.sqrt(np.mean(levels * levels, axis=1)), 95))
    white = rng.normal(0, 1, values.size)
    pink = np.cumsum(rng.normal(0, 1, values.size))
    pink -= np.convolve(pink, np.ones(64) / 64, mode="same")
    noise = 0.5 * white / white.std() + 0.5 * pink / max(pink.std(), 1e-6)
    return values + noise * signal / (10 ** (snr_db / 20)) / noise.std()


def features(model_dir: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from nexawake.enroll import to_pcm

    return models(model_dir).clip_features(to_pcm(values))


def mfcc(values: np.ndarray) -> np.ndarray:
    import librosa

    m = librosa.feature.mfcc(y=values / 32767.0, sr=SR, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40)
    full = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).T
    return full.astype(np.float32)


def plan(mswc: Path, seed: int, shots: int) -> dict:
    table = pd.read_csv(mswc / "test.csv")
    rng = np.random.default_rng(seed)
    tasks = []
    for word in TARGETS:
        rows = table[table.WORD == word]
        speakers = rows.SPEAKER.drop_duplicates().sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        enrolled = list(speakers[:shots])
        enroll_links = [rows[rows.SPEAKER == s].LINK.iloc[0] for s in enrolled]
        held = rows[~rows.SPEAKER.isin(enrolled)].drop_duplicates("SPEAKER")
        pos_links = held.LINK.sample(n=min(MAX_POSITIVES, len(held)), random_state=int(rng.integers(1 << 31))).tolist()
        tasks.append({"word": word, "enroll": enroll_links, "positives": pos_links})
    others = table[~table.WORD.isin(TARGETS)]
    picked = others.sample(n=NEGATIVE_CLIPS + ROOM_CLIPS, random_state=seed)
    return {"seed": seed, "shots": shots, "tasks": tasks,
            "room": picked.LINK.iloc[:ROOM_CLIPS].tolist(), "negatives": picked.LINK.iloc[ROOM_CLIPS:].tolist(),
            "negative_words": picked.WORD.iloc[ROOM_CLIPS:].tolist()}


def prepare_streams(args, spec: dict) -> dict:
    """Shared per-seed audio: room calibration and the long negative stream (clean and noisy)."""

    mswc = Path(args.mswc)
    rng = np.random.default_rng(spec["seed"] + 1000)
    room, _ = stream([read(mswc, link) for link in spec["room"]], rng, GAP_SECONDS)
    negative, spans = stream([read(mswc, link) for link in spec["negatives"]], rng, GAP_SECONDS)
    noisy = add_noise(negative, args.snr, np.random.default_rng(spec["seed"] + 2000))
    return {"room": room, "negative": negative, "negative_noisy": noisy, "negative_spans": spans}


def run_task(payload: tuple) -> dict:
    args, spec, task_index = payload
    from nexawake import fast
    from nexawake.enroll import build_wake, enroll, frame_rms
    from nexawake.features import FRAME_SECONDS
    from nexawake.wake import Normalizer

    mswc = Path(args.mswc)
    cache = Path(args.out) / f"seed{spec['seed']}_k{spec['shots']}"
    shared = np.load(cache / "streams.npz")
    room_feat = np.load(cache / "room_feat.npz")["emb"]
    neg = {name: np.load(cache / f"{name}_feat.npz") for name in ("negative", "negative_noisy")}
    neg_mfcc = {name: np.load(cache / f"{name}_mfcc.npy") for name in ("negative", "negative_noisy")}
    task = spec["tasks"][task_index]
    rng = np.random.default_rng(spec["seed"] * 100 + task_index)
    m = models(args.models)
    started = time.time()

    enrollments, enroll_audio = [], []
    for index, link in enumerate(task["enroll"]):
        clip = read(mswc, link)
        padded = np.concatenate([floor(rng, 1.0), clip + floor(rng, clip.size / SR)[: clip.size], floor(rng, 0.45)])
        noise = float(np.percentile(frame_rms(padded), 10))
        item = enroll(m, f"s{index}", padded, noise)
        if item is not None:
            enrollments.append(item)
            enroll_audio.append((padded, item.onset, item.offset))
    positives, pos_spans = stream([read(mswc, link) for link in task["positives"]], rng, POS_PAD_SECONDS)
    pos_noisy = add_noise(positives, args.snr, np.random.default_rng(spec["seed"] * 100 + task_index + 7))
    pos_feat = {"clean": m.clip_features(_pcm(positives)), "noisy": m.clip_features(_pcm(pos_noisy))}

    result = {"word": task["word"], "seed": spec["seed"], "shots": spec["shots"], "enrolled": len(enrollments),
              "positives": len(pos_spans), "methods": {}}
    if len(enrollments) < 3:
        return result

    def windows(ends: np.ndarray, spans: list[tuple[int, int]]) -> list[np.ndarray]:
        # A detection may come while the word is heard or within 0.5 s after it ends.
        return [np.where((ends >= a) & (ends <= b + int(0.5 * SR)))[0] for a, b in spans]

    def evaluate(name: str, score_fn, threshold: float | None, calib: dict | None) -> None:
        entry = {"threshold": threshold, "calibration": calib}
        for condition in ("clean", "noisy"):
            emb, ends = pos_feat[condition]
            pos_scores = score_fn(emb, positive=True, condition=condition)
            idx = windows(ends, pos_spans)
            entry[f"pos_min_{condition}"] = [float(np.min(pos_scores[i])) if i.size else float("inf") for i in idx]
            neg_key = "negative" if condition == "clean" else "negative_noisy"
            neg_scores = score_fn(neg[neg_key]["emb"], positive=False, condition=condition)
            entry[f"neg_events_{condition}"] = events(neg_scores, args.grid)
        result["methods"][name] = entry

    neg_hours = {c: neg[k]["emb"].shape[0] * FRAME_SECONDS / 3600 for c, k in
                 (("clean", "negative"), ("noisy", "negative_noisy"))}
    result["negative_hours"] = neg_hours

    # ---- NexaRoom as deployed, and ablations through the same build code
    variants = {
        "nexa": dict(augment=True, normalize=True),
        "nexa-noaug": dict(augment=False, normalize=True),
        "nexa-nonorm": dict(augment=True, normalize=False),
    }
    room_seq = [room_feat]
    built_full = None
    for name, options in variants.items():
        built = build_wake(enrollments, room_seq, **options)
        if built is None:
            continue
        if name == "nexa":
            built_full = built
        templates = built.spec.templates

        def score_fn(emb, *, positive, condition, templates=templates, normalizer=built.normalizer):
            return fast.wake_score(fast.template_scores(emb, templates, normalizer))

        evaluate(name, score_fn, built.spec.threshold, built.calibration.to_dict())
    if built_full is None:
        return result

    # ---- deployed templates, other scoring rules / feature subsets
    templates = built_full.spec.templates
    normalizer = built_full.normalizer

    def best1(emb, *, positive, condition):
        return fast.wake_score(fast.template_scores(emb, templates, normalizer), best=1)

    evaluate("nexa-min", best1, None, None)

    for name, cols in (("emb-only", slice(0, 96)), ("mel-only", slice(96, 128))):
        sub_t = [np.asarray(t)[:, cols] for t in templates]
        sub_norm = fit_normalizer([np.asarray(e.variants[0][0])[:, cols] for e in enrollments] + [room_feat[:, cols]])

        def sub_fn(emb, *, positive, condition, sub_t=sub_t, sub_norm=sub_norm, cols=cols):
            return fast.wake_score(fast.template_scores(np.asarray(emb)[:, cols], sub_t, sub_norm))

        evaluate(name, sub_fn, None, None)

    # Prototype: mean of all normalised template frames vs mean of the last L frames.
    proto = normalizer.apply(np.vstack([np.asarray(t) for t in templates])).mean(axis=0)
    proto /= max(np.linalg.norm(proto), 1e-6)
    width = int(np.median([len(t) for t in templates]))

    def proto_fn(emb, *, positive, condition):
        values = normalizer.apply(np.asarray(emb, dtype=np.float32))
        kernel = np.ones(width) / width
        smooth = np.vstack([np.convolve(values[:, d], kernel, mode="full")[: values.shape[0]]
                            for d in range(values.shape[1])]).T
        smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-6)
        return 1.0 - smooth @ proto

    evaluate("prototype", proto_fn, None, None)

    # ---- classic baseline: MFCC + deltas, CMVN, the same DTW matcher at 10 ms frames
    mf_templates = []
    for padded, onset, offset in enroll_audio:
        feats = mfcc(padded)
        mf_templates.append(feats[max(0, onset // 160): max(onset // 160 + 3, offset // 160)])
    mf_norm = fit_normalizer([mfcc(p) for p, _, _ in enroll_audio] + [mfcc(shared["room"])], unit=False)
    pos_mfcc = {"clean": mfcc(positives), "noisy": mfcc(pos_noisy)}

    def mfcc_fn(emb, *, positive, condition):
        seq = pos_mfcc[condition] if positive else neg_mfcc["negative" if condition == "clean" else "negative_noisy"]
        return fast.wake_score(fast.template_scores(seq, mf_templates, mf_norm))

    # Re-map frame positions to 10 ms for the positive windows.
    entry = {"threshold": None, "calibration": None}
    for condition in ("clean", "noisy"):
        seq_scores = mfcc_fn(None, positive=True, condition=condition)
        ends = (np.arange(seq_scores.shape[0]) + 1) * 160
        idx = windows(ends, pos_spans)
        entry[f"pos_min_{condition}"] = [float(np.min(seq_scores[i])) if i.size else float("inf") for i in idx]
        neg_scores = mfcc_fn(None, positive=False, condition=condition)
        entry[f"neg_events_{condition}"] = events(neg_scores, args.grid, refractory_frames=120)
    result["methods"]["mfcc-dtw"] = entry
    result["seconds"] = round(time.time() - started, 1)
    return result


class fit_normalizer:  # noqa: N801 - mirrors Normalizer for any dimension
    def __init__(self, frames: list[np.ndarray], unit: bool = True) -> None:
        data = np.vstack([np.asarray(f, dtype=np.float32) for f in frames if len(f)])
        std = data.std(axis=0)
        self.mean = data.mean(axis=0)
        self.std = np.maximum(std, max(float(np.median(std)) * 0.25, 1e-3))
        self.unit = unit

    def apply(self, values):
        data = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        return data / np.maximum(norms, 1e-6)


def _pcm(values: np.ndarray) -> bytes:
    return np.clip(values, -32768, 32767).astype("<i2").tobytes()


def events(scores: np.ndarray, grid: list[float], refractory_frames: int = 15) -> list[int]:
    """False alarms at each threshold: frames at or below it, at most one per refractory window."""

    finite = np.where(np.isfinite(scores), scores, np.inf).astype(np.float64)
    return [int(v) for v in _events(finite, np.asarray(grid, dtype=np.float64), refractory_frames)]


@__import__("numba").njit(cache=True)
def _events(scores, grid, refractory):
    out = np.zeros(grid.size, dtype=np.int64)
    for g in range(grid.size):
        tau = grid[g]
        last = -10**9
        count = 0
        for frame in range(scores.size):
            if scores[frame] <= tau and frame - last >= refractory:
                count += 1
                last = frame
        out[g] = count
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mswc", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--shots", default="6")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    args.grid = [round(float(v), 4) for v in np.arange(0.02, 1.2001, 0.005)]
    out = Path(args.out)
    for shots in [int(s) for s in args.shots.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            cache = out / f"seed{seed}_k{shots}"
            cache.mkdir(parents=True, exist_ok=True)
            spec = plan(Path(args.mswc), seed, shots)
            (cache / "plan.json").write_text(json.dumps(spec, ensure_ascii=False))
            if not (cache / "negative_noisy_mfcc.npy").is_file():
                started = time.time()
                streams = prepare_streams(args, spec)
                np.savez(cache / "streams.npz", room=streams["room"])
                for name in ("room", "negative", "negative_noisy"):
                    emb, ends = features(args.models, streams[name])
                    np.savez(cache / f"{name}_feat.npz", emb=emb, ends=ends)
                    if name != "room":
                        np.save(cache / f"{name}_mfcc.npy", mfcc(streams[name]))
                print(f"seed {seed} k {shots}: streams ready ({time.time() - started:.0f}s)", flush=True)
            with ProcessPoolExecutor(args.jobs) as pool:
                results = list(pool.map(run_task, [(args, spec, i) for i in range(len(spec["tasks"]))]))
            with (cache / "results.json").open("w") as handle:
                json.dump({"grid": args.grid, "results": results}, handle, ensure_ascii=False)
            done = sum(1 for r in results if r["methods"])
            print(f"seed {seed} k {shots}: {done}/{len(results)} targets evaluated", flush=True)


if __name__ == "__main__":
    main()
