# livekit-plugins-dtln

In-process DTLN noise suppression plugin for LiveKit — self-hosted, open-source alternative to Krisp/AI-coustics. Published on PyPI.

## Running

```bash
source venv/bin/activate        # Always activate venv first
pip install -e .                # Install in editable mode
python tests/test_noise_suppression.py  # Run noise suppression tests
```

## Python Environment

- Always use `python3`, never `python`
- Always activate `venv/` before running commands: `source venv/bin/activate`
- Install deps: `pip install -e .` (editable) or `pip install livekit-plugins-dtln` (from PyPI)

## Architecture

Two-stage LSTM pipeline running via ONNX Runtime:

1. **Model 1 — Spectral masking**: FFT → magnitude → LSTM spectral mask → masked IFFT
2. **Model 2 — Time-domain refinement**: Second LSTM refines waveform, removes residual artifacts

Both models are stateful (LSTM hidden states persist across frames). One instance per session.

### Key Files

| File | Purpose |
|---|---|
| `src/livekit/plugins/dtln/__init__.py` | Plugin registration, `noise_suppression()` factory |
| `src/livekit/plugins/dtln/noise_suppressor.py` | Core `DTLNNoiseSuppressor` — ONNX inference, overlap-add synthesis |
| `tests/test_noise_suppression.py` | RMS reduction tests on sample audio |
| `pyproject.toml` | Package metadata, dependencies, PyPI config |

## Key Patterns

- **Factory function**: Use `dtln.noise_suppression()` — not `DTLNNoiseSuppressor()` directly
- **One instance per session**: Each instance holds stateful LSTM hidden states scoped to a single call
- **Strength tuning**: `strength` param (0.0–1.0) controls wet/dry blend; default 0.5
- **Debug logging**: `debug_logging=True` logs per-block spectral mask diagnostics every ~800 ms
- **Model weights**: ONNX files bundled in PyPI wheel via `setuptools.package-data`

## Publishing

```bash
python -m build
twine upload dist/*
```

Package: `livekit-plugins-dtln` on PyPI. Current version defined in `pyproject.toml`.

## Detailed Docs

- `README.md` — full usage, performance benchmarks, signal flow, references
- `docs/` — demo page (GitHub Pages)
