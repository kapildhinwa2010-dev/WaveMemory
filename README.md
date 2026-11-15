# WaveMemory

Wave-lattice associative memory with interference recall.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/WaveMemory/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- Superposed wave lattice with interference-based recall
- Noisy-cue cleanup with tolerance checks
- Pure numpy, bipolar (+/-1) patterns

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from wavememory import WaveMemory

mem = WaveMemory(dim=64)
mem.write_many([p1, p2])
print(np.array_equal(mem.recall(noisy_cue), p1))
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).
