"""A compiled version of ``TemplateMatcher.step`` for whole sequences.

Same recurrence, same tie-breaking and the same length rule as the deployed
matcher (tests/test_fast.py checks equality on random data). It exists only
to make the evaluation fast enough to run thousands of trials; the device
itself runs the original code.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, fastmath=False)
def _sequence_scores(queries, templates, lengths):
    n_frames = queries.shape[0]
    count, max_len, dim = templates.shape
    out = np.full((n_frames, count), np.inf, dtype=np.float64)
    prev_cost = np.full((count, max_len), np.inf)
    prev_weight = np.ones((count, max_len))
    prev_start = np.zeros((count, max_len), dtype=np.int64)
    cost = np.empty((count, max_len))
    weight = np.empty((count, max_len))
    start = np.empty((count, max_len), dtype=np.int64)
    local = np.empty(max_len)
    for t in range(n_frames):
        for n in range(count):
            length = lengths[n]
            for k in range(max_len):
                if k < length:
                    dot = 0.0
                    for d in range(dim):
                        dot += templates[n, k, d] * queries[t, d]
                    value = 1.0 - dot
                    local[k] = value if value > 0.0 else 0.0
                else:
                    local[k] = np.inf
            for k in range(max_len):
                c = local[k]
                if k == 0:
                    best_c, best_w, best_s = c, 1.0, t
                    best_avg = best_c / best_w
                    oc, ow, os_ = prev_cost[n, 0] + c, prev_weight[n, 0] + 1.0, prev_start[n, 0]
                    avg = oc / ow
                    if avg < best_avg:
                        best_c, best_w, best_s, best_avg = oc, ow, os_, avg
                else:
                    best_c, best_w, best_s = prev_cost[n, k] + c, prev_weight[n, k] + 1.0, prev_start[n, k]
                    best_avg = best_c / best_w
                    oc, ow, os_ = prev_cost[n, k - 1] + 2.0 * c, prev_weight[n, k - 1] + 2.0, prev_start[n, k - 1]
                    avg = oc / ow
                    if avg < best_avg:
                        best_c, best_w, best_s, best_avg = oc, ow, os_, avg
                    oc, ow, os_ = cost[n, k - 1] + c, weight[n, k - 1] + 1.0, start[n, k - 1]
                    avg = oc / ow
                    if avg < best_avg:
                        best_c, best_w, best_s, best_avg = oc, ow, os_, avg
                cost[n, k] = best_c
                weight[n, k] = best_w
                start[n, k] = best_s
            last = length - 1
            used = t - start[n, last] + 1
            score = cost[n, last] / weight[n, last]
            if used * 2 >= length and used <= length * 2 + 1 and np.isfinite(score):
                out[t, n] = score
        for n in range(count):
            for k in range(max_len):
                prev_cost[n, k] = cost[n, k]
                prev_weight[n, k] = weight[n, k]
                prev_start[n, k] = start[n, k]
    return out


def pack(templates, normalizer) -> tuple[np.ndarray, np.ndarray]:
    arrays = [normalizer.apply(np.asarray(t, dtype=np.float32)) for t in templates if len(t)]
    lengths = np.asarray([a.shape[0] for a in arrays], dtype=np.int64)
    packed = np.zeros((len(arrays), int(lengths.max()), arrays[0].shape[1]), dtype=np.float64)
    for row, array in enumerate(arrays):
        packed[row, : array.shape[0]] = array
    return packed, lengths


def template_scores(sequence, templates, normalizer) -> np.ndarray:
    """Per-frame distance to every template: shape (frames, templates)."""

    packed, lengths = pack(templates, normalizer)
    queries = normalizer.apply(np.asarray(sequence, dtype=np.float32)).astype(np.float64)
    return _sequence_scores(np.ascontiguousarray(queries), np.ascontiguousarray(packed), lengths)


def wake_score(per_template: np.ndarray, best: int = 2) -> np.ndarray:
    """Deployed rule: mean of the two best finite template distances per frame."""

    ordered = np.sort(per_template, axis=1)[:, :best]
    finite = np.isfinite(ordered)
    total = np.where(finite, ordered, 0.0).sum(axis=1)
    count = finite.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.inf)
