"""Variants added after review (methods.py is left unchanged so that earlier results stay reproducible).

The first three complete the 2 x 2 design of the feature and alignment ablations: in methods.py,
'emb-only', 'mel-only' and 'prototype' inherit the deployed noise-augmented templates (3 per recording).
Here the same variants are built from the clean templates (1 per recording), as 'nexa-noaug' is:

  emb-only-noaug    96 embedding dimensions only, clean templates, own normaliser
  mel-only-noaug    32 log-mel dimensions only, clean templates, own normaliser
  prototype-noaug   mean of the normalised clean template frames, no DTW (as 'prototype')

Two discriminative few-shot heads on the same 96-d embedding (in the spirit of Lin et al., 2020, and of
openWakeWord's custom verifier models):

  emb-head          logistic regression, positives = frames of the clean templates, negatives = every frame
                    of a separate 20.8-min negative stream (1,000 clips of bank speakers saying bank words that
                    are NOT in the shipped bank; results/headneg/). Score = 1 / (1 + exp(z / 4)), z = logit.
  emb-head-aug      the same with the frames of the noise-augmented templates as positives (3 per recording)

The heads never see the bank or the evaluation negatives, so the bank rules and the oracle rules apply as
for every other variant. Enrollment, positives and negatives are reproduced with the same random seeds as
methods.evaluate_wake, so the wakes are the same wakes.
"""

from __future__ import annotations

import numpy as np

import common
from common import FRAME_SECONDS, REFRACTORY_FRAMES, fit_normalizer
from methods import enroll_clips, mask_frames

EXTRA_METHODS = ["emb-only-noaug", "mel-only-noaug", "prototype-noaug", "emb-head", "emb-head-aug",
                 "emb-head-lo", "emb-head-aug-lo"]
HEAD_C = 1.0
HEAD_TEMPERATURE = 4.0
LOGIT_SCALE = 60.0
# emb-head-lo, emb-head-aug-lo: the same two heads with the score on a log-odds scale, 0.75 - z / 60 clipped to
# [0, 1.5]. The probability scale compresses the tail near 0 (thresholds at 1 FA/h fall at 0.004-0.04, close to
# the grid resolution), which breaks the exponential tail fit; frame ranking, and so every oracle result, is the
# same on both scales up to grid resolution. Added after the first run of the probability-scale heads.


def _prototype_fn(templates, normalizer):
    proto = normalizer.apply(np.vstack([np.asarray(t) for t in templates])).mean(axis=0)
    proto /= max(np.linalg.norm(proto), 1e-6)
    width = int(np.median([len(t) for t in templates]))

    def fn(emb):
        values = normalizer.apply(np.asarray(emb, dtype=np.float32)).astype(np.float64)
        csum = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), axis=0)
        idx = np.arange(1, values.shape[0] + 1)
        smooth = (csum[idx] - csum[np.maximum(idx - width, 0)]) / width
        smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-6)
        return 1.0 - smooth @ proto

    return fn


def _head_fn(pos_frames: np.ndarray, neg_frames: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    x = np.vstack([pos_frames, neg_frames]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos_frames)), np.zeros(len(neg_frames))])
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    clf = LogisticRegression(C=HEAD_C, class_weight="balanced", max_iter=5000)
    clf.fit((x - mean) / std, y)
    w, b = clf.coef_[0], float(clf.intercept_[0])

    def logit(emb):
        return ((np.asarray(emb, dtype=np.float64)[:, :96] - mean) / std) @ w + b

    def fn(emb):
        return 1.0 / (1.0 + np.exp(np.clip(logit(emb) / HEAD_TEMPERATURE, -50, 50)))

    def fn_lo(emb):
        return np.clip(0.75 - logit(emb) / LOGIT_SCALE, 0.0, 1.5)

    return fn, fn_lo, {"train_pos": int(len(pos_frames)), "train_neg": int(len(neg_frames)),
                "train_acc_pos": float(np.mean(clf.predict((pos_frames - mean) / std) == 1))}


def evaluate_wake_extra(clips, positives, negatives, room, head_neg, rng, methods=EXTRA_METHODS) -> dict:
    from nexawake import fast
    from nexawake.enroll import build_wake

    m = common.models()
    enrollments, _audio = enroll_clips(clips, rng)
    out = {"enrolled": len(enrollments), "positives": {c: len(p.spans) for c, p in positives.items()},
           "negative_hours": {name: neg.hours() for name, neg in negatives.items()},
           "prefix_hours": {name: {p: s / 3600.0 for p, s in neg.prefixes.items()} for name, neg in negatives.items()},
           "methods": {}, "head_info": {}}
    if len(enrollments) < 3:
        return out
    pos_feat = {c: m.clip_features(common.to_pcm16(p.values).tobytes()) for c, p in positives.items()}

    def run(name, fn):
        entry = {"threshold": None, "calibration": None, "pos": {}, "neg": {}, "prefix": {}}
        for cond, pos in positives.items():
            emb, ends = pos_feat[cond]
            entry["pos"][cond] = common.window_min(fn(emb), ends, pos.spans)
        for stream_name, neg in negatives.items():
            scores = np.asarray(fn(neg.emb), dtype=np.float64)
            if neg.mask:
                scores = np.where(mask_frames(neg.mask, neg.ends), scores, np.inf)
            entry["neg"][stream_name] = common.events(scores, REFRACTORY_FRAMES).tolist()
            if neg.prefixes:
                entry["prefix"][stream_name] = {
                    p: common.events(scores[: int(seconds / FRAME_SECONDS)], REFRACTORY_FRAMES).tolist()
                    for p, seconds in neg.prefixes.items()}
        out["methods"][name] = entry

    room_emb = np.asarray(room["emb"])
    noaug = build_wake(enrollments, [room_emb], augment=False, normalize=True)
    full = build_wake(enrollments, [room_emb], augment=True, normalize=True) \
        if {"emb-head-aug", "emb-head-aug-lo"} & set(methods) else None
    if noaug is None:
        return out
    clean_templates = [np.asarray(t) for t in noaug.spec.templates]
    out["templates"] = len(clean_templates)
    for name, cols in (("emb-only-noaug", slice(0, 96)), ("mel-only-noaug", slice(96, 128))):
        if name not in methods:
            continue
        sub_t = [t[:, cols] for t in clean_templates]
        sub_norm = fit_normalizer([np.asarray(e.variants[0][0])[:, cols] for e in enrollments] + [room_emb[:, cols]])
        run(name, lambda emb, sub_t=sub_t, sub_norm=sub_norm, cols=cols:
            fast.wake_score(fast.template_scores(np.asarray(emb)[:, cols], sub_t, sub_norm)))
    if "prototype-noaug" in methods:
        run("prototype-noaug", _prototype_fn(clean_templates, noaug.normalizer))
    if head_neg is not None and ({"emb-head", "emb-head-lo"} & set(methods)):
        fn, fn_lo, info = _head_fn(np.vstack([t[:, :96] for t in clean_templates]), np.asarray(head_neg)[:, :96])
        out["head_info"]["emb-head"] = info
        if "emb-head" in methods:
            run("emb-head", fn)
        if "emb-head-lo" in methods:
            run("emb-head-lo", fn_lo)
    if head_neg is not None and full is not None and ({"emb-head-aug", "emb-head-aug-lo"} & set(methods)):
        aug_templates = [np.asarray(t) for t in full.spec.templates]
        fn, fn_lo, info = _head_fn(np.vstack([t[:, :96] for t in aug_templates]), np.asarray(head_neg)[:, :96])
        out["head_info"]["emb-head-aug"] = info
        if "emb-head-aug" in methods:
            run("emb-head-aug", fn)
        if "emb-head-aug-lo" in methods:
            run("emb-head-aug-lo", fn_lo)
    return out
