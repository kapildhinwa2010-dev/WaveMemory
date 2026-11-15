# WaveMemory

[![CI](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Wave-lattice associative memory: superposition write, correlation recall, lattice-only recall.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

WaveMemory stores +/-1 patterns two ways: as copies for direct
correlation recall (recall()), and superposed into a 1-D lattice for
recall without stored copies (recall_from_lattice()). cleanup()
recovers from noisy cues with a flip-tolerance check. The lattice
dynamics are a heuristic sign-update, and the docstring says so.

## Install

    pip install -r requirements.txt

## Quick start

    from wavememory import WaveMemory

    mem = WaveMemory(dim=64)
    mem.write_many([p1, p2])
    print(mem.recall(noisy_cue))

## API

- write / write_many / recall / recall_from_lattice / cleanup
- noise_pattern(shape, flip_prob, rng) - random +/-1 pattern with the
  given expected fraction of -1 entries (NOT a noised copy of a given
  pattern - see Memory-Lib for that variant)
- hamming_distance(a, b)

## Honest results and limitations

- recall_from_lattice() degrades fast with pattern count (cross-talk);
  the docstring quotes a rough capacity of dim / (2 * log(n)) - treat
  it as an order-of-magnitude hint, not a measured law.
- When copies exist, recall_from_lattice() falls back to the closest
  copy - it is only truly copy-free after you manage the pattern list
  yourself.
- The lattice dynamics are element-wise sign updates on a 1-D
  superposition, not a full outer-product Hopfield network (see
  Memory-Lib / Memory-Research for that).
- cleanup() judges by distance from the cue sign, not by whether the
  true pattern was recovered.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that carrier-mode recall under
  heavier noise failed about half the trials; superposition-based
  designs like this one are the safer direction but still lose
  reliability quickly as patterns accumulate.

## License

MIT - see [LICENSE](LICENSE).
