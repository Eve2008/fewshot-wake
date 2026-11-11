"""The compiled matcher must agree with the deployed TemplateMatcher frame by frame."""

from __future__ import annotations

import numpy as np

from nexawake.fast import template_scores, wake_score
from nexawake.features import FEATURE_DIM
from nexawake.wake import Normalizer, TemplateMatcher, WakeSpec


def test_equal_to_deployed_matcher() -> None:
    rng = np.random.default_rng(7)
    for trial in range(5):
        templates = [rng.normal(size=(int(rng.integers(3, 25)), FEATURE_DIM)).astype(np.float32) for _ in range(6)]
        stream = rng.normal(size=(300, FEATURE_DIM)).astype(np.float32)
        # Plant noisy copies of a template so that finite, low scores occur.
        stream[50:50 + len(templates[0])] = templates[0] + 0.1 * rng.normal(size=templates[0].shape)
        normalizer = Normalizer.fit([stream]) if trial % 2 else Normalizer.identity()
        matcher = TemplateMatcher([WakeSpec("w", "w", templates, 0.5)], normalizer)
        reference_t, reference_w = [], []
        for row in stream:
            per_template = matcher.step(row)
            reference_t.append(per_template)
            reference_w.append(matcher.wake_scores(per_template)[0])
        fast_t = template_scores(stream, templates, normalizer)
        np.testing.assert_allclose(fast_t, np.asarray(reference_t), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(wake_score(fast_t), np.asarray(reference_w), rtol=1e-5, atol=1e-6)
