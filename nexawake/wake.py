# Copied unchanged from NexaRoom (nexaroom/voice/wake.py, commit be63f3e, October 2026).
# Only the import of FEATURE_DIM/FRAME_SECONDS is package-relative, as in the original.
"""Few-shot, multi-wake detection by query-by-example matching.

Each wake (a word, a phrase, or any repeatable sound) is described by a few
recorded examples. Every example becomes a template: the sequence of neural
embeddings (see ``features.py``) covering the sound. Live audio is matched
against every template with an incremental, free-start dynamic time warping
(subsequence DTW) on cosine distances, so a wake can be said faster or slower
than during enrollment and can start at any time.

Decision thresholds are calibrated per wake from data instead of being fixed:
leave-one-out distances between a wake's own examples (positives) and the
closest matches found in ordinary room audio, the other wakes' examples and
enrolled speech (negatives).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

from .features import FEATURE_DIM as EMBEDDING_DIM, FRAME_SECONDS

SENSITIVITY_POSITION = {"low": 0.3, "medium": 0.5, "high": 0.7}
MIN_TEMPLATES = 3
MIN_TEMPLATE_FRAMES = 3
MAX_TEMPLATE_FRAMES = 30
REFRACTORY_SECONDS = 1.2
HOLD_FRAMES = 2


@dataclass(slots=True)
class WakeSpec:
    id: str
    name: str
    templates: list[Any]
    threshold: float | None = None
    confidence_low: float | None = None   # score mapped to 0 % confidence
    confidence_high: float | None = None  # score mapped to 100 % confidence


@dataclass(slots=True)
class WakeHit:
    wake_id: str
    name: str
    score: float
    threshold: float
    confidence: float
    frame: int
    runner_up: dict[str, Any] | None = None
    end_offset_seconds: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.wake_id,
            "name": self.name,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "confidence": round(self.confidence, 3),
            "runner_up": self.runner_up,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass(slots=True)
class Normalizer:
    mean: Any
    std: Any

    @classmethod
    def identity(cls) -> "Normalizer":
        import numpy as np  # type: ignore[import-not-found]

        return cls(np.zeros(EMBEDDING_DIM, dtype=np.float32), np.ones(EMBEDDING_DIM, dtype=np.float32))

    @classmethod
    def fit(cls, frames: Sequence[Any]) -> "Normalizer":
        import numpy as np  # type: ignore[import-not-found]

        stacked = [np.asarray(item, dtype=np.float32).reshape(-1, EMBEDDING_DIM) for item in frames if len(item)]
        if not stacked:
            return cls.identity()
        data = np.vstack(stacked)
        if data.shape[0] < 20:
            return cls(data.mean(axis=0), np.ones(EMBEDDING_DIM, dtype=np.float32))
        std = data.std(axis=0)
        floor = max(float(np.median(std)) * 0.25, 1e-3)
        return cls(data.mean(axis=0).astype(np.float32), np.maximum(std, floor).astype(np.float32))

    def apply(self, values: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        data = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        return data / np.maximum(norms, 1e-6)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": [round(float(v), 6) for v in self.mean], "std": [round(float(v), 6) for v in self.std]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Normalizer":
        import numpy as np  # type: ignore[import-not-found]

        if not data:
            return cls.identity()
        mean = np.asarray(data.get("mean", []), dtype=np.float32)
        std = np.asarray(data.get("std", []), dtype=np.float32)
        if mean.shape != (EMBEDDING_DIM,) or std.shape != (EMBEDDING_DIM,) or not np.all(std > 0):
            return cls.identity()
        return cls(mean, std)


def select_template(embeddings: Any, ends: Any, onset: int, offset: int, sample_rate: int = 16000) -> Any:
    """Pick the embeddings that describe the sound between onset and offset.

    Each embedding summarises the ~0.76 s that end at ``ends[i]``. Frames are
    kept from shortly after the onset (the window already contains the start
    of the sound) to shortly after the offset (the window still contains the
    end of it).
    """

    import numpy as np  # type: ignore[import-not-found]

    values = np.asarray(embeddings, dtype=np.float32)
    positions = np.asarray(ends)
    if values.shape[0] == 0:
        return values
    first = onset + int(0.12 * sample_rate)
    last = offset + int(0.20 * sample_rate)
    index = np.where((positions >= first) & (positions <= last))[0]
    if index.size < MIN_TEMPLATE_FRAMES:
        center = int(np.argmin(np.abs(positions - (onset + offset) / 2 - 0.3 * sample_rate)))
        start = max(0, center - 1)
        index = np.arange(start, min(values.shape[0], start + MIN_TEMPLATE_FRAMES))
    if index.size > MAX_TEMPLATE_FRAMES:
        index = index[:MAX_TEMPLATE_FRAMES]
    return values[index]


class TemplateMatcher:
    """Incremental free-start DTW of a live embedding stream against templates."""

    def __init__(self, wakes: Sequence[WakeSpec], normalizer: Normalizer) -> None:
        import numpy as np  # type: ignore[import-not-found]

        self._np = np
        self.wakes = [wake for wake in wakes if len(wake.templates) > 0]
        self.normalizer = normalizer
        templates: list[Any] = []
        owners: list[int] = []
        for index, wake in enumerate(self.wakes):
            for template in wake.templates:
                array = np.asarray(template, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
                if array.shape[0] >= 1:
                    templates.append(normalizer.apply(array))
                    owners.append(index)
        self.count = len(templates)
        self.owners = np.asarray(owners, dtype=np.int64)
        self.lengths = np.asarray([item.shape[0] for item in templates], dtype=np.int64)
        self.max_length = int(self.lengths.max()) if self.count else 0
        packed = np.zeros((self.count, max(1, self.max_length), EMBEDDING_DIM), dtype=np.float32)
        for row, template in enumerate(templates):
            packed[row, : template.shape[0]] = template
        self.templates = packed
        self.valid = np.arange(max(1, self.max_length))[None, :] < self.lengths[:, None]
        self.reset()

    def reset(self) -> None:
        np = self._np
        shape = (self.count, max(1, self.max_length))
        self._cost = np.full(shape, np.inf, dtype=np.float64)
        self._weight = np.ones(shape, dtype=np.float64)
        self._start = np.zeros(shape, dtype=np.int64)
        self.frame = 0

    def step(self, embedding: Any) -> Any:
        """Consume one embedding; return the best normalized distance per template."""

        np = self._np
        if self.count == 0:
            self.frame += 1
            return np.zeros(0)
        query = self.normalizer.apply(np.asarray(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM))[0]
        local = 1.0 - self.templates @ query            # (n, L) cosine distance in [0, 2]
        local = np.where(self.valid, np.maximum(local, 0.0), np.inf)
        prev_cost, prev_weight, prev_start = self._cost, self._weight, self._start
        cost = np.full_like(prev_cost, np.inf)
        weight = np.ones_like(prev_weight)
        start = np.zeros_like(prev_start)
        frame = self.frame
        for k in range(self.max_length):
            c = local[:, k]
            # Candidate predecessors: (cost, weight, start)
            if k == 0:
                options = [
                    (c, np.ones_like(c), np.full(self.count, frame)),                       # new start
                    (prev_cost[:, 0] + c, prev_weight[:, 0] + 1, prev_start[:, 0]),          # vertical
                ]
            else:
                options = [
                    (prev_cost[:, k] + c, prev_weight[:, k] + 1, prev_start[:, k]),          # vertical
                    (prev_cost[:, k - 1] + 2 * c, prev_weight[:, k - 1] + 2, prev_start[:, k - 1]),  # diagonal
                    (cost[:, k - 1] + c, weight[:, k - 1] + 1, start[:, k - 1]),             # horizontal
                ]
            best_cost, best_weight, best_start = options[0]
            best_avg = best_cost / best_weight
            for option_cost, option_weight, option_start in options[1:]:
                average = option_cost / option_weight
                better = average < best_avg
                best_cost = np.where(better, option_cost, best_cost)
                best_weight = np.where(better, option_weight, best_weight)
                best_start = np.where(better, option_start, best_start)
                best_avg = np.where(better, average, best_avg)
            cost[:, k] = best_cost
            weight[:, k] = best_weight
            start[:, k] = best_start
        self._cost, self._weight, self._start = cost, weight, start
        rows = np.arange(self.count)
        last = self.lengths - 1
        end_cost = cost[rows, last]
        end_weight = weight[rows, last]
        used = frame - start[rows, last] + 1
        score = end_cost / end_weight
        allowed = (used * 2 >= self.lengths) & (used <= self.lengths * 2 + 1)
        self.frame += 1
        return np.where(allowed & np.isfinite(score), score, np.inf)

    def path_frames(self, template_index: int) -> int:
        last = int(self.lengths[template_index]) - 1
        return int(self.frame - 1 - self._start[template_index, last] + 1)

    def wake_scores(self, template_scores: Any) -> Any:
        """Combine template distances into one score per wake (mean of the 2 best)."""

        np = self._np
        result = np.full(len(self.wakes), np.inf)
        for index in range(len(self.wakes)):
            values = np.sort(template_scores[self.owners == index])
            values = values[np.isfinite(values)]
            if values.size:
                result[index] = float(values[: min(2, values.size)].mean())
        return result


class WakeDetector:
    """Peak-picking decision layer over :class:`TemplateMatcher`."""

    def __init__(self, wakes: Sequence[WakeSpec], normalizer: Normalizer) -> None:
        self.specs = [wake for wake in wakes if len(wake.templates) >= MIN_TEMPLATES and wake.threshold]
        self.matcher = TemplateMatcher(self.specs, normalizer)
        self._candidate: dict[str, Any] | None = None
        self._refractory_until = -1
        self._frame = 0
        self.last_scores: dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return bool(self.specs)

    def reset(self) -> None:
        self.matcher.reset()
        self._candidate = None
        self._refractory_until = -1

    def process(self, embedding: Any) -> WakeHit | None:
        import numpy as np  # type: ignore[import-not-found]

        if not self.specs:
            return None
        frame = self._frame
        self._frame += 1
        template_scores = self.matcher.step(embedding)
        scores = self.matcher.wake_scores(template_scores)
        self.last_scores = {spec.id: float(score) for spec, score in zip(self.specs, scores)}
        if frame < self._refractory_until:
            return None
        thresholds = np.asarray([spec.threshold for spec in self.specs], dtype=np.float64)
        ratios = scores / thresholds
        best = int(np.argmin(ratios))
        below = bool(ratios[best] <= 1.0)
        candidate = self._candidate
        if below and (candidate is None or ratios[best] < candidate["ratio"]):
            owner_templates = np.where(self.matcher.owners == best)[0]
            best_template = owner_templates[int(np.argmin(template_scores[owner_templates]))]
            self._candidate = {
                "index": best, "ratio": float(ratios[best]), "score": float(scores[best]),
                "frame": frame, "hold": 0, "scores": scores.copy(),
                "frames": self.matcher.path_frames(int(best_template)),
            }
            return None
        if candidate is None:
            return None
        candidate["hold"] += 1
        if below and candidate["hold"] < HOLD_FRAMES:
            return None
        self._candidate = None
        return self._fire(candidate)

    def _fire(self, candidate: dict[str, Any]) -> WakeHit:
        index = candidate["index"]
        spec = self.specs[index]
        runner_up = None
        if len(self.specs) > 1:
            others = [(float(candidate["scores"][i]) / float(self.specs[i].threshold), i)
                      for i in range(len(self.specs)) if i != index]
            ratio, other = min(others)
            if math.isfinite(ratio):
                runner_up = {"id": self.specs[other].id, "name": self.specs[other].name,
                             "score": round(float(candidate["scores"][other]), 4),
                             "ratio": round(ratio, 3)}
        current = self._frame - 1
        self._refractory_until = current + int(REFRACTORY_SECONDS / FRAME_SECONDS)
        self.matcher.reset()
        lag_frames = max(0, current - candidate["frame"])
        return WakeHit(
            wake_id=spec.id,
            name=spec.name,
            score=candidate["score"],
            threshold=float(spec.threshold or 0),
            confidence=confidence(candidate["score"], spec),
            frame=candidate["frame"],
            runner_up=runner_up,
            end_offset_seconds=lag_frames * FRAME_SECONDS,
            duration_seconds=max(1, candidate["frames"]) * FRAME_SECONDS,
        )


def confidence(score: float, spec: WakeSpec) -> float:
    high = spec.confidence_high if spec.confidence_high is not None else (spec.threshold or 0.3) * 0.6
    low = spec.confidence_low if spec.confidence_low is not None else (spec.threshold or 0.3) * 1.25
    if low <= high:
        return 1.0 if score <= high else 0.0
    return float(max(0.0, min(1.0, (low - score) / (low - high))))


# ----------------------------------------------------------------- calibration
def sequence_min_score(templates: Sequence[Any], sequence: Any, normalizer: Normalizer,
                       block_frames: int | None = None) -> list[float]:
    """Run a matcher over a whole embedding sequence.

    Returns the minimum wake score over the sequence, or one minimum per block
    of ``block_frames`` frames (used to sample many negatives from long audio).
    """

    import numpy as np  # type: ignore[import-not-found]

    spec = WakeSpec(id="x", name="x", templates=list(templates))
    matcher = TemplateMatcher([spec], normalizer)
    values = np.asarray(sequence, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    per_frame = []
    for row in values:
        per_frame.append(float(matcher.wake_scores(matcher.step(row))[0]))
    if not per_frame:
        return []
    if block_frames is None:
        return [min(per_frame)]
    blocks = []
    for start in range(0, len(per_frame), block_frames):
        chunk = [value for value in per_frame[start:start + block_frames] if math.isfinite(value)]
        if chunk:
            blocks.append(min(chunk))
    return blocks


@dataclass(slots=True)
class WakeCalibration:
    threshold: float
    positives: list[float] = field(default_factory=list)
    negatives_min: float | None = None
    negative_count: int = 0
    quality: str = "unknown"      # excellent | good | fair | poor | uncalibrated
    expected_recall: float = 0.0
    sample_scores: list[float] = field(default_factory=list)
    confidence_low: float | None = None
    confidence_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "positives": [round(v, 4) for v in self.positives],
            "negatives_min": None if self.negatives_min is None else round(self.negatives_min, 4),
            "negative_count": self.negative_count,
            "quality": self.quality,
            "expected_recall": round(self.expected_recall, 3),
            "sample_scores": [round(v, 4) for v in self.sample_scores],
            "confidence_low": None if self.confidence_low is None else round(self.confidence_low, 4),
            "confidence_high": None if self.confidence_high is None else round(self.confidence_high, 4),
        }


def calibrate_wake(
    samples: Sequence[tuple[str, Any, bool]],
    templates: Sequence[tuple[str, Any]],
    negative_sequences: Iterable[Any],
    normalizer: Normalizer,
    sensitivity: str = "medium",
    *,
    has_room_negatives: bool = True,
) -> tuple[WakeCalibration, dict[str, float]]:
    """Compute a data-driven threshold for one wake.

    ``samples`` holds ``(group, embedding_sequence, is_clean)`` for every
    recording and its noise/phase-augmented copies; ``templates`` holds
    ``(group, template)``. Positives are leave-one-*group*-out matches, so a
    recording is never compared with its own augmented copies. Returns the
    calibration and the clean-recording score per group (for sample quality).
    """

    import numpy as np  # type: ignore[import-not-found]

    position = SENSITIVITY_POSITION.get(sensitivity, 0.5)
    positives: list[float] = []
    per_group: dict[str, float] = {}
    for group, sequence, clean in samples:
        others = [template for other, template in templates if other != group]
        if len(others) < 2:
            continue
        scores = sequence_min_score(others, sequence, normalizer)
        value = scores[0] if scores else math.inf
        positives.append(value)
        if clean:
            per_group[group] = value
    finite_positive = [value for value in positives if math.isfinite(value)]
    negatives: list[float] = []
    block = max(1, int(1.0 / FRAME_SECONDS))
    all_templates = [template for _group, template in templates]
    for sequence in negative_sequences:
        negatives.extend(value for value in sequence_min_score(all_templates, sequence, normalizer, block)
                         if math.isfinite(value))

    if not finite_positive:
        return WakeCalibration(threshold=0.0, positives=positives, quality="poor"), per_group

    ordered = sorted(finite_positive)
    # Ignore the worst ~10 % of takes: one bad recording should not drag the
    # threshold into the negatives.
    upper_index = max(0, int(math.ceil(len(ordered) * 0.9)) - 1)
    positive_high = ordered[upper_index]
    positive_median = float(np.median(ordered))
    negative_min = min(negatives) if negatives else None
    # A low percentile is more robust than the absolute minimum on long audio.
    negative_floor = float(np.percentile(negatives, 1)) if len(negatives) >= 30 else negative_min

    if negative_floor is None:
        threshold = positive_high * (1.10 + 0.20 * position)
        quality = "uncalibrated"
    elif positive_high < negative_floor:
        threshold = positive_high + (negative_floor - positive_high) * position
        gap = (negative_floor - positive_high) / max(negative_floor, 1e-6)
        quality = "excellent" if gap >= 0.25 else "good" if gap >= 0.12 else "fair"
        if not has_room_negatives and quality != "fair":
            quality = "uncalibrated"
    else:
        # Overlap: stay below the closest negative unless high sensitivity was
        # explicitly requested.
        safe = negative_floor * 0.97
        threshold = max(positive_median, safe) if position >= 0.7 else safe
        quality = "poor"
    threshold = float(max(threshold, 1e-3))
    recall = sum(value <= threshold for value in finite_positive) / len(finite_positive)
    low = negative_floor if negative_floor is not None else threshold * 1.25
    high = min(positive_median, threshold * 0.9)
    calibration = WakeCalibration(
        threshold=threshold,
        positives=positives,
        negatives_min=negative_min,
        negative_count=len(negatives),
        quality=quality,
        expected_recall=recall,
        sample_scores=list(per_group.values()),
        confidence_low=float(max(low, threshold * 1.05)),
        confidence_high=float(high),
    )
    return calibration, per_group
