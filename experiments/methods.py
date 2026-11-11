"""Scoring of one enrolled wake by every method, on positives and on any number of
negative streams. Shared by the MSWC cross-speaker, same-speaker and ESC-50 runs.

For each method the result holds:
  pos[condition]  best score inside each positive window (lower = closer),
  neg[stream]     false-alarm events at every threshold of common.GRID,
  prefix[stream]  the same for leading parts of a stream (bank-size ablation),
  threshold       the deployed calibration's threshold (NexaRoom variants only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import common
from common import FRAME_SECONDS, MFCC_HOP, MFCC_REFRACTORY, REFRACTORY_FRAMES, SR, fit_normalizer

ALL_METHODS = ["nexa", "nexa-noaug", "nexa-nonorm", "nexa-min", "nexa-noaug-bnorm",
               "emb-only", "mel-only", "prototype", "mfcc-dtw"]
MASK_AFTER_SECONDS = 2.0


@dataclass
class NegStream:
    emb: np.ndarray
    ends: np.ndarray
    mfcc: np.ndarray | None = None
    prefixes: dict[str, float] = field(default_factory=dict)       # name -> seconds from the start
    mask: list[tuple[int, int]] = field(default_factory=list)       # sample spans to ignore

    def hours(self) -> float:
        total = self.emb.shape[0] * FRAME_SECONDS
        return (total - masked_seconds(self.mask, self.ends)) / 3600.0


def masked_seconds(mask, ends) -> float:
    if not mask:
        return 0.0
    keep = mask_frames(mask, ends)
    return float(np.sum(~keep)) * FRAME_SECONDS


def mask_frames(mask, positions) -> np.ndarray:
    """True for frames whose end position is outside every masked span (+2 s)."""

    keep = np.ones(positions.size, dtype=bool)
    for a, b in mask:
        lo = np.searchsorted(positions, a, side="left")
        hi = np.searchsorted(positions, b + int(MASK_AFTER_SECONDS * SR), side="right")
        keep[lo:hi] = False
    return keep


@dataclass
class Positives:
    values: np.ndarray
    spans: list[tuple[int, int]]


def enroll_clips(clips: list[np.ndarray], rng: np.random.Generator):
    """NexaRoom enrollment of raw clips: 1 s pre-roll, energy bounds, noisy and shifted copies."""

    from nexawake.enroll import enroll, frame_rms

    m = common.models()
    enrollments, audio = [], []
    for index, clip in enumerate(clips):
        padded = np.concatenate([common.floor(rng, SR), clip + common.floor(rng, clip.size),
                                 common.floor(rng, int(0.45 * SR))])
        noise = float(np.percentile(frame_rms(padded), 10))
        item = enroll(m, f"s{index}", padded, noise)
        if item is not None:
            enrollments.append(item)
            audio.append((padded, item.onset, item.offset))
    return enrollments, audio


def evaluate_wake(clips: list[np.ndarray], positives: dict[str, Positives], negatives: dict[str, NegStream],
                  room: dict, bank_feat: np.ndarray | None, rng: np.random.Generator,
                  methods: list[str] = ALL_METHODS) -> dict:
    from nexawake import fast
    from nexawake.enroll import build_wake
    from nexawake.wake import Normalizer

    m = common.models()
    enrollments, enroll_audio = enroll_clips(clips, rng)
    out = {"enrolled": len(enrollments), "positives": {c: len(p.spans) for c, p in positives.items()},
           "negative_hours": {name: neg.hours() for name, neg in negatives.items()},
           "prefix_hours": {name: {p: s / 3600.0 for p, s in neg.prefixes.items()} for name, neg in negatives.items()},
           "methods": {}}
    if len(enrollments) < 3:
        return out

    pos_feat = {c: m.clip_features(common.to_pcm16(p.values).tobytes()) for c, p in positives.items()}
    need_mfcc = "mfcc-dtw" in methods
    pos_mfcc = {c: common.mfcc(common.to_pcm16(p.values).astype(np.float32)) for c, p in positives.items()} \
        if need_mfcc else {}

    def run(name: str, emb_fn, threshold=None, calibration=None, mfcc_fn=None) -> None:
        entry = {"threshold": threshold, "calibration": calibration, "pos": {}, "neg": {}, "prefix": {}}
        for cond, pos in positives.items():
            if mfcc_fn is None:
                emb, ends = pos_feat[cond]
                scores = emb_fn(emb)
            else:
                scores = mfcc_fn(pos_mfcc[cond])
                ends = (np.arange(scores.shape[0]) + 1) * MFCC_HOP
            entry["pos"][cond] = common.window_min(scores, ends, pos.spans)
        for stream_name, neg in negatives.items():
            if mfcc_fn is None:
                scores = emb_fn(neg.emb)
                positions = neg.ends
                refractory, per_second = REFRACTORY_FRAMES, 1.0 / FRAME_SECONDS
            else:
                scores = mfcc_fn(neg.mfcc)
                positions = np.arange(scores.shape[0]) * MFCC_HOP
                refractory, per_second = MFCC_REFRACTORY, SR / MFCC_HOP
            scores = np.asarray(scores, dtype=np.float64)
            if neg.mask:
                scores = np.where(mask_frames(neg.mask, positions), scores, np.inf)
            entry["neg"][stream_name] = common.events(scores, refractory).tolist()
            if neg.prefixes:
                entry["prefix"][stream_name] = {
                    p: common.events(scores[: int(seconds * per_second)], refractory).tolist()
                    for p, seconds in neg.prefixes.items()}
        out["methods"][name] = entry

    room_emb = room["emb"]
    builds = {}
    for name, options in (("nexa", dict(augment=True, normalize=True)),
                          ("nexa-noaug", dict(augment=False, normalize=True)),
                          ("nexa-nonorm", dict(augment=True, normalize=False))):
        needed = {"nexa": {"nexa", "nexa-min", "emb-only", "mel-only", "prototype"},
                  "nexa-noaug": {"nexa-noaug", "nexa-noaug-bnorm"},
                  "nexa-nonorm": {"nexa-nonorm"}}[name]
        if not needed & set(methods):
            continue
        builds[name] = build_wake(enrollments, [room_emb], **options)

    full = builds.get("nexa")
    if full is not None and ({"nexa", "nexa-min"} & set(methods)):
        cache: dict[int, np.ndarray] = {}

        def per_template(emb, templates=full.spec.templates, normalizer=full.normalizer):
            key = id(emb)          # every input array stays alive while the cache is used
            if key not in cache:
                cache[key] = fast.template_scores(emb, templates, normalizer)
            return cache[key]

        if "nexa" in methods:
            run("nexa", lambda emb: fast.wake_score(per_template(emb)), full.spec.threshold,
                full.calibration.to_dict())
        if "nexa-min" in methods:
            run("nexa-min", lambda emb: fast.wake_score(per_template(emb), best=1))
        cache.clear()
    for name in ("nexa-noaug", "nexa-nonorm"):
        built = builds.get(name)
        if built is None or name not in methods:
            continue
        run(name, lambda emb, b=built: fast.wake_score(fast.template_scores(emb, b.spec.templates, b.normalizer)),
            built.spec.threshold, built.calibration.to_dict())
    noaug = builds.get("nexa-noaug")
    if noaug is not None and "nexa-noaug-bnorm" in methods and bank_feat is not None:
        clean = [np.asarray(e.variants[0][0]) for e in enrollments]
        bnorm = Normalizer.fit(clean + [np.asarray(bank_feat, dtype=np.float32)])
        run("nexa-noaug-bnorm",
            lambda emb: fast.wake_score(fast.template_scores(emb, noaug.spec.templates, bnorm)))

    if full is None:
        return out
    templates = full.spec.templates
    for name, cols in (("emb-only", slice(0, 96)), ("mel-only", slice(96, 128))):
        if name not in methods:
            continue
        sub_t = [np.asarray(t)[:, cols] for t in templates]
        sub_norm = fit_normalizer([np.asarray(e.variants[0][0])[:, cols] for e in enrollments]
                                  + [np.asarray(room_emb)[:, cols]])
        run(name, lambda emb, sub_t=sub_t, sub_norm=sub_norm, cols=cols:
            fast.wake_score(fast.template_scores(np.asarray(emb)[:, cols], sub_t, sub_norm)))

    if "prototype" in methods:
        normalizer = full.normalizer
        proto = normalizer.apply(np.vstack([np.asarray(t) for t in templates])).mean(axis=0)
        proto /= max(np.linalg.norm(proto), 1e-6)
        width = int(np.median([len(t) for t in templates]))

        def proto_fn(emb):
            values = normalizer.apply(np.asarray(emb, dtype=np.float32)).astype(np.float64)
            csum = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), axis=0)
            idx = np.arange(1, values.shape[0] + 1)
            smooth = (csum[idx] - csum[np.maximum(idx - width, 0)]) / width
            smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-6)
            return 1.0 - smooth @ proto

        run("prototype", proto_fn)

    if need_mfcc:
        mf_templates = []
        for padded, onset, offset in enroll_audio:
            feats = common.mfcc(padded)
            mf_templates.append(feats[max(0, onset // MFCC_HOP): max(onset // MFCC_HOP + 3, offset // MFCC_HOP)])
        mf_norm = fit_normalizer([common.mfcc(p) for p, _, _ in enroll_audio] + [room["mfcc"]])
        run("mfcc-dtw", None, mfcc_fn=lambda seq: fast.wake_score(fast.template_scores(seq, mf_templates, mf_norm)))
    return out
