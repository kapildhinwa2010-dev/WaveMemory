import numpy as np
import pytest

from wavememory import WaveMemory, hamming_distance, noise_pattern


def _pattern(dim, ones_at):
    p = -np.ones(dim)
    p[list(ones_at)] = 1
    return p


def test_write_and_len():
    mem = WaveMemory(dim=16)
    mem.write(_pattern(16, [0, 1, 2]))
    assert len(mem) == 1


def test_write_rejects_wrong_size():
    mem = WaveMemory(dim=16)
    with pytest.raises(ValueError):
        mem.write(np.ones(8))


def test_write_rejects_non_bipolar():
    mem = WaveMemory(dim=4)
    with pytest.raises(ValueError):
        mem.write(np.array([0, 1, 1, 1]))


def test_recall_exact():
    mem = WaveMemory(dim=16)
    p1 = _pattern(16, range(8))
    p2 = _pattern(16, range(8, 16))
    mem.write_many([p1, p2])
    assert np.array_equal(mem.recall(p1), p1)
    assert np.array_equal(mem.recall(p2), p2)


def test_recall_noisy():
    rng = np.random.default_rng(0)
    mem = WaveMemory(dim=64)
    p1 = rng.choice([-1, 1], size=64)
    p2 = -p1
    mem.write_many([p1, p2])
    noisy = p1.copy()
    flips = rng.choice(64, size=4, replace=False)
    noisy[flips] *= -1
    assert np.array_equal(mem.recall(noisy), p1)


def test_recall_empty_raises():
    mem = WaveMemory(dim=8)
    with pytest.raises(ValueError):
        mem.recall(np.ones(8))


def test_hamming_distance():
    assert hamming_distance(np.array([1, -1, 1]), np.array([1, 1, 1])) == 1


def test_hamming_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        hamming_distance(np.ones(3), np.ones(4))


def test_noise_pattern():
    rng = np.random.default_rng(1)
    p = noise_pattern((10,), 0.5, rng)
    assert p.shape == (10,)
    assert set(np.unique(p)) <= {-1, 1}


def test_noise_pattern_rejects_bad_prob():
    with pytest.raises(ValueError):
        noise_pattern((4,), 1.5, np.random.default_rng(0))


def test_cleanup_recovers_small_noise():
    rng = np.random.default_rng(2)
    mem = WaveMemory(dim=128)
    p1 = rng.choice([-1, 1], size=128)
    mem.write(p1)
    noisy = p1.copy()
    flips = rng.choice(128, size=3, replace=False)
    noisy[flips] *= -1
    assert np.array_equal(mem.cleanup(noisy), p1)


def test_dim_rejects_zero():
    with pytest.raises(ValueError):
        WaveMemory(dim=0)
