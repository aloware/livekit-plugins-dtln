"""The suppressor's output must not depend on how the caller chunks the audio.

Before the output queue existed, each call handed back exactly the audio that
happened to be ready at that moment, zero-padded or truncated to the length of
the incoming frame.  How much damage that did depended on how the 128-sample
hop and the two resamplers lined up with the frame length, so the same call
sounded different at 20 ms frames than at 50 ms, and at the sizes that lined up
worst the speech stopped being recognisable as speech at all.

LiveKit picks the frame size (``AudioInputOptions.frame_size_ms``, 50 by
default), and a plugin cannot assume it will stay put.
"""

import numpy as np
import pytest
from livekit import rtc

from livekit.plugins.dtln import DTLNNoiseSuppressor

SR = 48000
FRAME_SIZES_MS = [8, 10, 16, 20, 24, 32, 40, 48, 50]


def _speechlike(seconds: float = 4.0, sr: int = SR) -> np.ndarray:
    """A harmonic stack with syllable-rate amplitude modulation.

    Not real speech, but it has the property that matters here: a pitched
    signal with a moving envelope, so any chopping or zero-filling shows up as
    a change in the waveform rather than being masked by noise.
    """
    t = np.arange(int(sr * seconds)) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    sig = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([140, 280, 560, 1120], start=1))
    sig = sig * env / np.abs(sig).max()
    return (sig * 0.4 * 32767).astype(np.int16)


def _render(pcm: np.ndarray, frame_ms: int, *, strength: float = 1.0) -> np.ndarray:
    ns = DTLNNoiseSuppressor(strength=strength)
    n = SR * frame_ms // 1000
    out = []
    for i in range(0, len(pcm) - n, n):
        frame = rtc.AudioFrame(
            data=pcm[i : i + n].tobytes(),
            sample_rate=SR,
            num_channels=1,
            samples_per_channel=n,
        )
        processed = ns._process(frame)
        assert processed.samples_per_channel == n, "must return one frame's worth"
        assert processed.sample_rate == SR
        assert processed.num_channels == 1
        out.append(np.frombuffer(processed.data, dtype=np.int16))
    return np.concatenate(out).astype(np.float64) / 32768.0


def _best_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation at the best alignment, since latency may differ by size."""
    m = min(len(a), len(b))
    best = -1.0
    for lag in range(0, 3001, 16):
        for x, y in ((a[: m - lag], b[lag:m]), (a[lag:m], b[: m - lag])):
            if len(x) < SR:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            best = max(best, c)
    return best


@pytest.fixture(scope="module")
def rendered() -> dict[int, np.ndarray]:
    pcm = _speechlike()
    return {ms: _render(pcm, ms) for ms in FRAME_SIZES_MS}


@pytest.mark.parametrize("frame_ms", FRAME_SIZES_MS)
def test_output_matches_the_reference_chunking(rendered, frame_ms):
    """Every frame size must yield the same audio as the smallest one."""
    assert _best_correlation(rendered[FRAME_SIZES_MS[0]], rendered[frame_ms]) > 0.97


@pytest.mark.parametrize("frame_ms", FRAME_SIZES_MS)
def test_signal_survives_at_every_frame_size(rendered, frame_ms):
    """A guard for the failure mode itself: output that is present but ruined.

    Correlation alone would not catch a size whose output collapsed to silence,
    so check the level independently.
    """
    rms = float(np.sqrt((rendered[frame_ms] ** 2).mean()))
    assert rms > 0.01, f"output nearly silent at {frame_ms}ms (rms={rms:.5f})"


def test_no_zero_filled_gaps_in_steady_state():
    """The old force-fit injected silence mid-stream; nothing should now."""
    pcm = _speechlike()
    y = _render(pcm, 50)
    # Skip the priming latency at the head, then look for runs of exact silence.
    tail = y[SR:]
    silent = np.abs(tail) < 1e-5
    runs, run = [], 0
    for s in silent:
        if s:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    longest = max(runs) if runs else 0
    assert longest < SR // 100, f"{longest} consecutive silent samples mid-stream"


def test_frames_are_answered_one_for_one():
    """Total output length must equal total input length, whatever the size."""
    pcm = _speechlike(seconds=2.0)
    for ms in (20, 32, 50):
        n = SR * ms // 1000
        expected = len(range(0, len(pcm) - n, n)) * n
        assert len(_render(pcm, ms)) == expected
