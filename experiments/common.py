"""Shared helpers for the wake experiments: audio streams, noise, chunk-parallel
feature extraction (bit-identical to the single-stream path), MFCC baseline,
false-alarm event counting and the fixed speaker/word partition.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numba  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/ali/Desktop/Articles/data")
MSWC = DATA / "mswc"
MODELS = DATA / "models"
RESULTS = ROOT / "results"

SR = 16000
CHUNK = 1280                      # one feature frame per 80 ms
FRAME_SECONDS = CHUNK / SR
GAP_SECONDS = 0.25
POS_PAD_SECONDS = 1.0
FLOOR_RMS = 20.0
REFRACTORY_FRAMES = 15            # 1.2 s at 80 ms, as REFRACTORY_SECONDS in wake.py
MFCC_HOP = 160
MFCC_REFRACTORY = 120             # 1.2 s at 10 ms
# Threshold grid: 0.0005 steps below 0.1 (mel-only and no-normaliser scores are small), 0.002 above.
GRID = np.unique(np.round(np.concatenate([np.arange(0.0, 0.1, 0.0005), np.arange(0.1, 1.5 + 1e-9, 0.002)]), 4))

TARGETS = ["برای", "خیلی", "باید", "نیست", "اینجا", "دوست", "وقتی", "چیزی", "آنها", "وجود", "حالا", "الان",
           "خونه", "درست", "خواهم", "کردم"]

_MODELS = None


def models():
    global _MODELS
    if _MODELS is None:
        from nexawake.features import EmbeddingModels

        _MODELS = EmbeddingModels(MODELS)
        _MODELS.load()
    return _MODELS


# ------------------------------------------------------------------ audio
def wav_path(link: str) -> Path:
    word, name = link.split("/", 1)
    return MSWC / "audio" / f"{word}_{name.replace('.opus', '.wav')}"


def read(link: str) -> np.ndarray:
    data, rate = sf.read(wav_path(link), dtype="float32")
    assert rate == SR, rate
    return data * 32767.0


def floor(rng: np.random.Generator, samples: int) -> np.ndarray:
    return rng.normal(0, FLOOR_RMS, int(samples)).astype(np.float32)


def stream(clips: list[np.ndarray], rng: np.random.Generator, pad: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Concatenate clips with ``pad`` seconds of a low noise floor between them."""

    gap = int(pad * SR)
    total = gap + sum(c.size + gap for c in clips)
    out = floor(rng, total)
    spans, cursor = [], gap
    for clip in clips:
        out[cursor: cursor + clip.size] += clip.astype(np.float32)
        spans.append((cursor, cursor + clip.size))
        cursor += clip.size + gap
    return out, spans


def signal_level(values: np.ndarray) -> float:
    usable = values[: values.size - values.size % 320].reshape(-1, 320).astype(np.float64)
    return float(np.percentile(np.sqrt(np.mean(usable * usable, axis=1)), 95))


def synthetic_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    """Half white, half band-limited 'pink-like' noise, as in the pilot; unit std."""

    white = rng.normal(0, 1, size)
    pink = np.cumsum(rng.normal(0, 1, size))
    pink -= np.convolve(pink, np.ones(64) / 64, mode="same")
    noise = 0.5 * white / white.std() + 0.5 * pink / max(pink.std(), 1e-6)
    return noise / noise.std()


def add_noise(values: np.ndarray, snr_db: float, seed: int, segment_seconds: float = 60.0) -> np.ndarray:
    """White + pink noise at a fixed SNR relative to the 95th percentile frame level.

    Generated per 60 s segment to bound memory on multi-hour streams.
    """

    level = signal_level(values)
    scale = level / (10 ** (snr_db / 20))
    out = values.astype(np.float32).copy()
    seg = int(segment_seconds * SR)
    rng = np.random.default_rng(seed)
    for start in range(0, values.size, seg):
        end = min(values.size, start + seg)
        out[start:end] += (synthetic_noise(end - start, rng) * scale).astype(np.float32)
    return out


def to_pcm16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -32768, 32767).astype("<i2")


# ------------------------------------------------- chunk-parallel features
WARMUP_CHUNKS = 16   # > 76 mel frames / 8 per chunk; the stream state is then identical


def _feature_part(args):
    path, start, end = args
    pcm = np.load(path, mmap_mode="r")
    begin = max(0, start - WARMUP_CHUNKS * CHUNK)
    piece = np.asarray(pcm[begin:end]).astype("<i2").tobytes()
    emb, ends = models().clip_features(piece)
    ends = ends + begin
    keep = ends > start
    return emb[keep], ends[keep]


def features(pcm: np.ndarray, jobs: int = 8, scratch: Path | None = None, part_seconds: float = 600.0):
    """Neural features of a long int16 stream, split into chunk-aligned parts with warm-up.

    Equal to ``EmbeddingModels.clip_features`` on the whole stream (tests/test_parallel_features.py).
    """

    pcm = np.asarray(pcm, dtype="<i2")
    if pcm.size < 2 * part_seconds * SR or jobs <= 1:
        return models().clip_features(pcm.tobytes())
    scratch = Path(scratch or os.environ.get("WAKE_SCRATCH", "/tmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"pcm_{os.getpid()}_{id(pcm)}.npy"
    np.save(path, pcm)
    step = int(part_seconds * SR) // CHUNK * CHUNK
    usable = pcm.size - pcm.size % CHUNK
    bounds = [(str(path), s, min(usable, s + step)) for s in range(0, usable, step)]
    try:
        with ProcessPoolExecutor(jobs) as pool:
            parts = list(pool.map(_feature_part, bounds))
    finally:
        path.unlink(missing_ok=True)
    emb = np.vstack([p[0] for p in parts]).astype(np.float32)
    ends = np.concatenate([p[1] for p in parts]).astype(np.int64)
    return emb, ends


def mfcc(values: np.ndarray) -> np.ndarray:
    import librosa

    y = np.asarray(values, dtype=np.float32) / 32767.0
    # Log-mel without librosa's default 80 dB floor below the global maximum, so that
    # every frame depends only on its own samples (needed for chunked extraction).
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=MFCC_HOP, n_mels=40)
    m = librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=None), n_mfcc=13)
    full = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).T
    return full.astype(np.float32)


MFCC_MARGIN = 10 * MFCC_HOP


def _mfcc_part(args):
    path, start, end, total = args
    pcm = np.load(path, mmap_mode="r")
    begin = max(0, start - MFCC_MARGIN)
    stop = min(total, end + MFCC_MARGIN)
    piece = np.asarray(pcm[begin:stop]).astype(np.float32)
    feats = mfcc(piece)
    first = (start - begin) // MFCC_HOP
    count = (end - start) // MFCC_HOP if end < total else feats.shape[0] - first
    return feats[first: first + count]


def mfcc_long(pcm: np.ndarray, jobs: int = 8, scratch: Path | None = None, part_seconds: float = 600.0):
    """MFCC+deltas of a long stream in parallel parts with a 10-frame margin (equal to ``mfcc``)."""

    pcm = np.asarray(pcm, dtype="<i2")
    if pcm.size < 2 * part_seconds * SR or jobs <= 1:
        return mfcc(pcm.astype(np.float32))
    scratch = Path(scratch or os.environ.get("WAKE_SCRATCH", "/tmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"pcm_mfcc_{os.getpid()}_{id(pcm)}.npy"
    np.save(path, pcm)
    step = int(part_seconds * SR) // MFCC_HOP * MFCC_HOP
    bounds = [(str(path), s, min(pcm.size, s + step), pcm.size) for s in range(0, pcm.size, step)]
    try:
        with ProcessPoolExecutor(jobs) as pool:
            parts = list(pool.map(_mfcc_part, bounds))
    finally:
        path.unlink(missing_ok=True)
    return np.vstack(parts).astype(np.float32)


# ------------------------------------------------------------ evaluation
@numba.njit(cache=True)
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


def events(scores: np.ndarray, refractory: int = REFRACTORY_FRAMES, grid: np.ndarray = GRID) -> np.ndarray:
    """False alarms at each grid threshold: frames at or below it, at most one per refractory window."""

    finite = np.where(np.isfinite(scores), scores, np.inf).astype(np.float64)
    return _events(finite, np.asarray(grid, dtype=np.float64), refractory)


def window_min(scores: np.ndarray, ends: np.ndarray, spans: list[tuple[int, int]], after: float = 0.5) -> list[float]:
    """Best score while the target is heard or within ``after`` seconds after it ends."""

    out = []
    for a, b in spans:
        idx = np.where((ends >= a) & (ends <= b + int(after * SR)))[0]
        out.append(float(np.min(scores[idx])) if idx.size else float("inf"))
    return out


class fit_normalizer:  # noqa: N801 - mirrors wake.Normalizer for any dimension
    def __init__(self, frames: list[np.ndarray]) -> None:
        data = np.vstack([np.asarray(f, dtype=np.float32) for f in frames if len(f)])
        std = data.std(axis=0)
        self.mean = data.mean(axis=0)
        self.std = np.maximum(std, max(float(np.median(std)) * 0.25, 1e-3))

    def apply(self, values):
        data = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        return data / np.maximum(norms, 1e-6)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def dump_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False))
