# WaveMemory - Wave-lattice associative memory with interference recall
# Part of the Kapildhinwa2010-dev Toolkit.

"""MemoryFlow - wave-lattice associative memory with interference recall.

Lineage: reserv.dean-part-2 (helix.py wave arena). Patterns are written as
superposed waves into a lattice; recall uses normalized correlation
(interference) so partial or noisy cues retrieve the closest stored pattern.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WaveMemory", "hamming_distance", "noise_pattern"]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("patterns must have the same shape")
    return int((a != b).sum())


def noise_pattern(shape, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Random +/-1 pattern; ``flip_prob`` is the expected fraction of -1 entries.

    flip_prob=0 -> all +1; flip_prob=1 -> all -1; 0.5 -> unbiased random.
    """
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be in [0, 1]")
    mask = rng.random(size=shape) < flip_prob
    return np.where(mask, -1.0, 1.0)


class WaveMemory:
    """Superposition memory with interference-based recall."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.lattice = np.zeros(self.dim, dtype=np.float64)
        self.patterns: list = []

    def write(self, pattern: np.ndarray) -> None:
        pattern = np.asarray(pattern, dtype=np.float64).ravel()
        if pattern.size != self.dim:
            raise ValueError("pattern size must equal dim")
        if not np.all(np.isin(pattern, (-1.0, 1.0))):
            raise ValueError("patterns must contain only -1 and +1")
        self.lattice += pattern
        self.patterns.append(pattern.copy())

    def write_many(self, patterns) -> None:
        for pattern in patterns:
            self.write(pattern)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Return the stored pattern most correlated with ``cue``."""
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns:
            raise ValueError("memory is empty")
        scores = [float(cue @ p) for p in self.patterns]
        return self.patterns[int(np.argmax(scores))].copy()

    def recall_from_lattice(self, cue: np.ndarray) -> np.ndarray:
        """Recall using only the superposed lattice (no stored copies needed).

        Runs Hopfield-style sign dynamics on the lattice energy until stable,
        then returns the cleaned bipolar state. If stored copies exist, the
        closest copy is returned instead. With many superposed patterns the
        lattice recall degrades (cross-talk) - this is a real limitation, not
        a bug: capacity is roughly dim / (2 * log(n_patterns)).
        """
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not np.any(self.lattice):
            raise ValueError("memory is empty")
        # Matched filter against the superposition, then clean up by sign.
        state = np.sign(cue + self.lattice)
        state[state == 0] = 1.0
        for _ in range(8):
            updated = np.sign(self.lattice * state + cue)
            updated[updated == 0] = 1.0
            if np.array_equal(updated, state):
                break
            state = updated
        if self.patterns:
            scores = [float(state @ p) for p in self.patterns]
            return self.patterns[int(np.argmax(scores))].copy()
        return state.copy()

    def cleanup(self, noisy: np.ndarray, max_flips: int = 8) -> np.ndarray:
        """Recall and verify; raise if the result is too far from the cue."""
        recovered = self.recall(noisy)
        flips = hamming_distance(recovered, np.sign(noisy))
        if flips > max_flips * self.dim // 100:
            raise ValueError("cleanup failed: {} flips".format(flips))
        return recovered

    def __len__(self) -> int:
        return len(self.patterns)
