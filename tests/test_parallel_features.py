"""Chunk-parallel feature extraction must equal the single-stream deployed path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import common  # noqa: E402


def _stream() -> np.ndarray:
    import pandas as pd

    table = pd.read_csv(common.MSWC / "test.csv").sample(n=20, random_state=3)
    values, _ = common.stream([common.read(link) for link in table.LINK], np.random.default_rng(1), 0.25)
    return common.to_pcm16(common.add_noise(values, 10.0, seed=5))


def test_parallel_features_equal(tmp_path) -> None:
    pcm = _stream()
    full_emb, full_ends = common.models().clip_features(pcm.tobytes())
    emb, ends = common.features(pcm, jobs=3, scratch=tmp_path, part_seconds=4.0)
    np.testing.assert_array_equal(ends, full_ends)
    np.testing.assert_allclose(emb, full_emb, rtol=0, atol=1e-5)


def test_parallel_mfcc_equal(tmp_path) -> None:
    pcm = _stream()
    full = common.mfcc(pcm.astype(np.float32))
    parts = common.mfcc_long(pcm, jobs=3, scratch=tmp_path, part_seconds=4.0)
    assert parts.shape == full.shape
    np.testing.assert_allclose(parts, full, rtol=1e-4, atol=1e-3)
