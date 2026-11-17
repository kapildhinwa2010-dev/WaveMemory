# WaveMemory

WaveMemory is a small Python library for storing and recalling bipolar (`-1`/`+1`) patterns. It keeps a one-dimensional sum of written patterns and also retains individual pattern copies for correlation-based recall. The current implementation is minimal; its tests cover basic behavior, not broad capacity or reliability claims.

## What is implemented

- `WaveMemory(dim)` initializes an empty memory and rejects dimensions below one.
- `write(pattern)` validates the flattened pattern size and requires every value to be `-1` or `+1`. It adds the pattern to the lattice and stores a copy. `write_many(patterns)` applies `write` to each supplied pattern.
- `recall(cue)` returns the stored pattern with the largest dot-product score against the cue. It requires a cue with the configured number of values and a non-empty stored-pattern list.
- `recall_from_lattice(cue)` performs up to eight element-wise sign updates using the lattice and cue. If stored pattern copies are present, it returns the copy most correlated with the resulting state; if copies have been removed, it returns the computed state. This is a simple one-dimensional heuristic, not a full matrix-based Hopfield network.
- `cleanup(noisy, max_flips=8)` calls `recall` and raises `ValueError` if the recovered pattern differs from `sign(noisy)` by more than `max_flips * dim // 100` positions. This is a check against the cue, not verification against a known correct pattern.
- `hamming_distance(a, b)` counts unequal entries and rejects arrays of different shapes.
- `noise_pattern(shape, flip_prob, rng)` creates a random bipolar pattern in which `flip_prob` is the probability of a negative entry. It does not add noise to a supplied pattern.

## Requirements

The project metadata requires Python 3.9 or newer and NumPy 1.24 or newer. `requirements.txt` also lists pytest 7.4 or newer for running tests.

From the repository root, install the declared requirements:

```sh
python -m pip install -r requirements.txt
```

## Example

Run from the repository root so that `wavememory.py` is importable:

```python
import numpy as np
from wavememory import WaveMemory

memory = WaveMemory(dim=4)
pattern_a = np.array([1, -1, 1, -1])
pattern_b = np.array([-1, 1, -1, 1])
memory.write_many([pattern_a, pattern_b])

cue = np.array([1, -1, -1, -1])
recalled = memory.recall(cue)
print(recalled)
```

The `wavememory.py` module is a top-level Python module. The repository does not define a command-line interface.

## Tests and CI

On the checked-out repository commit, the local test command `python -m pytest -q` completed successfully: **16 passed**. The tests cover input validation, exact and selected noisy-cue recall, cleanup, Hamming distance, noise-pattern generation, and lattice recall for a single stored pattern after its stored copy is removed.

`.github/workflows/ci.yml` defines a GitHub Actions test job for Python 3.10, 3.11, and 3.12, including pytest, compilation, and a smoke test. The workflow configuration was inspected, but a remote Actions run for this commit was not verified here.

## Scope and limitations

This repository contains a compact implementation and a small test suite, not measured capacity or benchmark results. Recall from the summed lattice can be affected by interference as patterns accumulate; the code does not establish a numerical capacity guarantee. Lattice updates are element-wise and capped at eight iterations. `recall` and `cleanup` rely on stored copies, and `cleanup` only compares the returned pattern with the cue's sign. The tests do not establish behavior across large pattern sets, extensive noise levels, or production workloads.

## License

The repository's `LICENSE` file contains the MIT License. See [LICENSE](LICENSE) for its terms.
