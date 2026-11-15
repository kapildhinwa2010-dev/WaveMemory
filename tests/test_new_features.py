"""Tests for corrected noise semantics and lattice-only recall."""

import numpy as np
import pytest

from wavememory import WaveMemory, noise_pattern


def test_noise_pattern_zero_prob_all_positive():
    p = noise_pattern((50,), 0.0, np.random.default_rng(0))
    assert (p == 1).all()


def test_noise_pattern_one_prob_all_negative():
    p = noise_pattern((50,), 1.0, np.random.default_rng(0))
    assert (p == -1).all()

def test_lattice_recall_without_stored_copies():
    rng = np.random.default_rng(0)
    mem = WaveMemory(64)
    p = rng.choice([-1.0, 1.0], size=64)
    mem.write(p)
    mem.patterns.clear()
    assert np.array_equal(mem.recall_from_lattice(p), p)


def test_lattice_recall_empty_raises():
    mem = WaveMemory(8)
    with pytest.raises(ValueError):
        mem.recall_from_lattice(np.ones(8))
