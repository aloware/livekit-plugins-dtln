# livekit-plugins-dtln

Python [LiveKit](https://livekit.io) plugin for **DTLN** (Dual-Signal Transformation LSTM Network) noise suppression — a fully self-hosted, open-source alternative to cloud-based noise cancellation services like Krisp or AI-coustics.

Runs entirely **in-process** using [ONNX Runtime](https://onnxruntime.ai). No cloud API, no per-minute fees, no proprietary binaries. Works with **self-hosted LiveKit** servers.

> Based on [Westhausen & Meyer, "Noise Reduction with DTLN", Interspeech 2020](https://www.isca-archive.org/interspeech_2020/westhausen20_interspeech.html)
> Original implementation: [github.com/breizhn/DTLN](https://github.com/breizhn/DTLN)

**[Live audio comparison demo →](https://aloware.github.io/livekit-plugins-dtln/)**

---

## Why DTLN?

| | DTLN (this plugin) | Krisp / AI-coustics |
|---|---|---|
| **Hosting** | Self-hosted, in-process | Cloud API required |
| **Cost** | Free (open weights) | Per-minute billing |
| **LiveKit** | Works with self-hosted | Requires LiveKit Cloud |
| **Latency** | ~8 ms (one block shift) | Network round-trip |
| **Privacy** | Audio never leaves your server | Audio sent to third party |
| **Real-time factor** | ~0.05× (20× faster than real-time) | Varies |

---

## Installation

**pip:**

```bash
pip install livekit-plugins-dtln
```

**requirements.txt:**

```
livekit-plugins-dtln
```

**From source:**

```bash
git clone https://github.com/aloware/livekit-plugins-dtln.git
pip install -e ./livekit-plugins-dtln
```

> The pretrained ONNX model weights (~4 MB) are bundled in the PyPI wheel — no separate download step needed.

---

## Usage

### Session pipeline (recommended)

```python
from livekit.agents import room_io
from livekit.plugins import dtln

await session.start(
    # ...,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=dtln.noise_suppression(),
        ),
    ),
)
```

### Custom AudioStream

```python
from livekit import rtc
from livekit.plugins import dtln

stream = rtc.AudioStream.from_track(
    track=track,
    noise_cancellation=dtln.noise_suppression(),
)
```

> **Note:** Create one `dtln.noise_suppression()` instance **per session**. Each instance holds stateful LSTM hidden states that must be scoped to a single call.

> **Note:** DTLN is trained on raw microphone audio. Do not chain it with another noise cancellation model — applying two models in series produces unexpected results.

### Custom model paths

```python
dtln.noise_suppression(
    model_1_path="/path/to/model_1.onnx",
    model_2_path="/path/to/model_2.onnx",
)
```

---

## Requirements

- Python >= 3.10
- livekit >= 1.1.0
- livekit-agents >= 1.4.4
- onnxruntime >= 1.17.0
- numpy >= 1.26.0

---

## How It Works

DTLN uses two sequential LSTM-based models:

1. **Model 1 — Spectral masking**: Computes the magnitude spectrum of a 32 ms window, runs it through an LSTM to produce a spectral mask, applies the mask in the frequency domain (preserving phase), and reconstructs the time-domain signal via IFFT.

2. **Model 2 — Time-domain refinement**: Refines the output of Model 1 with a second LSTM that operates directly on the waveform, capturing residual artifacts that spectral processing misses.

The two models are chained: Model 1's output feeds Model 2. Both LSTMs are stateful — their hidden states persist across audio frames, giving the network temporal context across the full duration of a call.

**Signal flow:**

```
Input frame (any sample rate, any channels)
  → downsample to 16 kHz mono
  → overlap-add loop (512-sample window, 128-sample shift)
      → FFT → magnitude → Model 1 (spectral mask) → masked IFFT
      → Model 2 (time-domain refinement)
  → upsample back to original sample rate
  → restore original channel count
→ Denoised output frame
```

The overlap-add synthesis uses 75% overlap (512-sample window, 128-sample shift), identical to the original DTLN paper. This gives ~8 ms of algorithmic latency at 16 kHz.

---

## Performance

Benchmarked on Apple M3 Pro, processing 16 kHz mono audio:

| Metric | Value |
|---|---|
| Steady-state latency per block | ~0.7 ms |
| Real-time factor | ~0.05× |
| Headroom vs real-time | ~20× |
| Cold-start (first inference) | ~500 ms (amortized by warmup in `__init__`) |

The `__init__` method runs a dummy forward pass to trigger ONNX Runtime's JIT compilation before the first real audio frame arrives, eliminating the cold-start stall.

---

## Models

Pretrained weights are the official DTLN models published by the original authors:

| File | Source |
|---|---|
| `model_1.onnx` | [breizhn/DTLN · pretrained_model/](https://github.com/breizhn/DTLN/tree/master/pretrained_model) |
| `model_2.onnx` | [breizhn/DTLN · pretrained_model/](https://github.com/breizhn/DTLN/tree/master/pretrained_model) |

The models are not bundled in this repository (to keep it lightweight). They are downloaded automatically by `python agent.py download-files` or by calling `download_models()` directly.

---

## References

- **Original DTLN paper**: [Westhausen & Meyer, "Noise Reduction with DTLN", Interspeech 2020](https://www.isca-archive.org/interspeech_2020/westhausen20_interspeech.html)
- **Original DTLN implementation & pretrained models**: [github.com/breizhn/DTLN](https://github.com/breizhn/DTLN)
- **DataDog engineering article** — the inspiration for this plugin: [Building a Real-Time Noise Suppression Library](https://www.datadoghq.com/blog/engineering/noise-suppression-library/)
- **LiveKit noise cancellation overview**: [docs.livekit.io — Noise Cancellation](https://docs.livekit.io/transport/media/noise-cancellation/)
- **LiveKit Agents SDK**: [github.com/livekit/agents](https://github.com/livekit/agents)
- **ONNX Runtime**: [onnxruntime.ai](https://onnxruntime.ai)

---

## License

The plugin code in this repository is released under the **MIT License**.

The pretrained DTLN model weights are published by the original authors under the **MIT License** — see [breizhn/DTLN](https://github.com/breizhn/DTLN/blob/master/LICENSE).
