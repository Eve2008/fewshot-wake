"""Enrollment and calibration exactly as NexaRoom's VoiceService builds them,
without the storage, microphone and threading layers.

``clip_variants`` reproduces ``VoiceService._clip_variants``: the clean clip,
two noisy copies (white + pink noise at 15 and 8 dB SNR, seeded by the clip
content) and three copies shifted by 20, 40 and 60 ms. ``build_wake``
reproduces the wake part of ``VoiceService._build``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np

from .features import CHUNK_SAMPLES, EmbeddingModels
from .wake import MIN_TEMPLATES, Normalizer, WakeCalibration, WakeSpec, calibrate_wake, select_template

AUGMENT_SNR_DB = (15.0, 8.0)
HUB_FRAME_SAMPLES = 320          # 20 ms microphone frames
SAMPLE_RATE = 16000


def frame_rms(values: np.ndarray) -> np.ndarray:
    usable = values[: values.size - values.size % HUB_FRAME_SAMPLES]
    if not usable.size:
        return np.zeros(0)
    frames = usable.reshape(-1, HUB_FRAME_SAMPLES).astype(np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1))


def onset_offset_levels(noise: float) -> tuple[float, float]:
    noise = max(float(noise), 20.0)
    return max(noise * 3.2, noise + 260.0, 220.0), max(noise * 2.0, noise + 120.0, 150.0)


def sound_bounds(values: np.ndarray, noise: float) -> tuple[int, int] | None:
    """NexaRoom's recorder rule: the loudest contiguous region, in samples."""

    levels = frame_rms(values)
    if not levels.size:
        return None
    onset_level, offset_level = onset_offset_levels(noise)
    loud = np.where(levels >= onset_level)[0]
    if not loud.size:
        return None
    start, end = int(loud[0]), int(loud[-1])
    while start > 0 and levels[start - 1] >= offset_level:
        start -= 1
    while end + 1 < levels.size and levels[end + 1] >= offset_level:
        end += 1
    return start * HUB_FRAME_SAMPLES, (end + 1) * HUB_FRAME_SAMPLES


def to_pcm(values: np.ndarray) -> bytes:
    return np.clip(values, -32768, 32767).astype("<i2").tobytes()


def clip_variants(models: EmbeddingModels, values: np.ndarray, *, augment: bool) -> list[tuple[Any, Any]]:
    pcm = to_pcm(values)
    key = sha256(pcm).hexdigest()[:24]
    variants = [models.clip_features(pcm)]
    if augment:
        data = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        levels = data.reshape(-1)[: data.size - data.size % 320].reshape(-1, 320)
        signal = float(np.percentile(np.sqrt(np.mean(levels * levels, axis=1)), 95)) if levels.size else 1000.0
        rng = np.random.default_rng(int(key[:8], 16))
        for snr in AUGMENT_SNR_DB:
            white = rng.normal(0, 1, data.size)
            pink = np.cumsum(rng.normal(0, 1, data.size))
            pink -= np.convolve(pink, np.ones(64) / 64, mode="same")
            pink /= max(float(np.std(pink)), 1e-6)
            noise = 0.5 * white + 0.5 * pink
            noise *= signal / (10 ** (snr / 20)) / max(float(np.std(noise)), 1e-6)
            variants.append(models.clip_features(to_pcm(data + noise)))
        for shift in range(HUB_FRAME_SAMPLES, CHUNK_SAMPLES, HUB_FRAME_SAMPLES):
            variants.append(models.clip_features(bytes(shift * 2) + pcm))
    return variants


@dataclass
class Enrollment:
    """Feature variants of one recorded wake sample, plus its sound bounds."""

    group: str
    variants: list[tuple[Any, Any]]
    onset: int
    offset: int


def enroll(models: EmbeddingModels, group: str, values: np.ndarray, noise: float) -> Enrollment | None:
    bounds = sound_bounds(values, noise)
    if bounds is None:
        return None
    return Enrollment(group, clip_variants(models, values, augment=True), bounds[0], bounds[1])


@dataclass
class BuiltWake:
    spec: WakeSpec
    calibration: WakeCalibration
    normalizer: Normalizer


def build_wake(
    enrollments: Sequence[Enrollment],
    negative_sequences: Sequence[Any],
    *,
    sensitivity: str = "medium",
    room_seconds: float = 20.0,
    augment: bool = True,
    normalize: bool = True,
    name: str = "wake",
) -> BuiltWake | None:
    templates: list[tuple[str, Any]] = []
    samples: list[tuple[str, Any, bool]] = []
    clean: list[Any] = []
    template_variants = 1 + len(AUGMENT_SNR_DB) if augment else 1
    for item in enrollments:
        for index, (emb, ends) in enumerate(item.variants):
            if not augment and index > 0:
                continue
            if index < template_variants:
                templates.append((item.group, select_template(emb, ends, item.onset, item.offset)))
            samples.append((item.group, emb, index == 0))
            if index == 0:
                clean.append(emb)
    if len({group for group, _ in templates}) < MIN_TEMPLATES:
        return None
    normalizer = Normalizer.fit(clean + list(negative_sequences)) if normalize else Normalizer.identity()
    calibration, _ = calibrate_wake(samples, templates, list(negative_sequences), normalizer, sensitivity,
                                    has_room_negatives=room_seconds >= 8)
    if calibration.threshold <= 0:
        return None
    spec = WakeSpec(name, name, [t for _g, t in templates], calibration.threshold,
                    calibration.confidence_low, calibration.confidence_high)
    return BuiltWake(spec, calibration, normalizer)
