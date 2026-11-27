"""The suppressor must add as little delay as the model allows.

Every millisecond here is added to the caller's audio before speech-to-text or
a realtime model hears it, on every turn, and no latency metric downstream can
see it.  The model's 512-sample window fixes 24 ms of it.  The rest is ours:
the lead that covers the 128-sample hop, and the two rate converters.
"""

import numpy as np
import pytest
from livekit import rtc

from livekit.plugins.dtln import DTLNNoiseSuppressor
from livekit.plugins.dtln.noise_suppressor import (
    _FirResampler,
    _SoxrResampler,
    _make_resampler,
)

FIR_RATES = [8000, 16000, 24000, 32000, 48000]
# 24 ms window + 8.5 ms hop lead + 1 ms per converter (2 ms at 8 kHz).
MAX_DELAY_MS = 37.0


def _noise(sr: int, seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(sr * seconds)) * 0.1 * 32767).astype(np.int16)


def _run(ns: DTLNNoiseSuppressor, pcm: np.ndarray, sr: int, frame_samples: int) -> np.ndarray:
    out = []
    for i in range(0, len(pcm) - frame_samples + 1, frame_samples):
        frame = rtc.AudioFrame(
            data=pcm[i : i + frame_samples].tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=frame_samples,
        )
        processed = ns._process(frame)
        assert processed.samples_per_channel == frame_samples
        out.append(np.frombuffer(processed.data, dtype=np.int16))
    return np.concatenate(out).astype(np.float64) / 32768.0


def _delay_ms(reference: np.ndarray, delayed: np.ndarray, sr: int) -> float:
    """Lag of ``delayed`` behind ``reference`` at the cross-correlation peak."""
    n = 1 << (len(reference) + len(delayed)).bit_length()
    xcorr = np.fft.irfft(np.fft.rfft(delayed, n) * np.conj(np.fft.rfft(reference, n)), n)
    return 1000.0 * int(np.argmax(xcorr[: sr * 300 // 1000])) / sr


@pytest.mark.parametrize("sr", FIR_RATES)
def test_delay_stays_near_the_model_minimum(sr):
    """Strength 0 passes the input through the full path, so the lag is exact."""
    pcm = _noise(sr)
    ns = DTLNNoiseSuppressor(strength=0.0)
    out = _run(ns, pcm, sr, sr // 100)
    delay = _delay_ms(pcm.astype(np.float64) / 32768.0, out, sr)
    assert 24.0 <= delay <= MAX_DELAY_MS, f"{delay:.1f} ms at {sr} Hz"


@pytest.mark.parametrize("sr", FIR_RATES)
@pytest.mark.parametrize("frame_ms", [7, 10, 20, 33, 50])
def test_the_hop_lead_never_runs_dry(sr, frame_ms):
    ns = DTLNNoiseSuppressor(strength=1.0)
    _run(ns, _noise(sr), sr, sr * frame_ms // 1000)
    assert ns._underruns == 0


def test_first_frame_is_answered_with_audio_not_a_silent_placeholder():
    sr = 48000
    ns = DTLNNoiseSuppressor(strength=0.0)
    out = _run(ns, _noise(sr, 1.0), sr, sr // 100)
    first_sound_ms = 1000.0 * int(np.argmax(np.abs(out) > 1e-3)) / sr
    assert first_sound_ms <= MAX_DELAY_MS


def test_other_rates_fall_back_to_the_buffering_converter():
    sr = 44100
    assert isinstance(_make_resampler(sr, 16000), _SoxrResampler)
    ns = DTLNNoiseSuppressor(strength=0.0)
    pcm = _noise(sr)
    out = _run(ns, pcm, sr, sr // 100)
    assert float(np.sqrt((out[sr:] ** 2).mean())) > 0.01


@pytest.mark.parametrize("in_rate,out_rate", [(48000, 16000), (24000, 16000), (16000, 24000), (16000, 48000)])
def test_fir_converter_is_chosen_and_keeps_sample_count(in_rate, out_rate):
    rs = _make_resampler(in_rate, out_rate)
    assert isinstance(rs, _FirResampler)
    x = np.random.default_rng(1).standard_normal(in_rate).astype(np.float32)
    out = np.concatenate([rs.push(x[i : i + 237]) for i in range(0, len(x), 237)])
    assert abs(len(out) - out_rate) <= 1


def _tone_gain_db(in_rate: int, out_rate: int, freq: float) -> float:
    rs = _FirResampler(in_rate, out_rate)
    t = np.arange(in_rate) / in_rate
    y = rs.push(np.sin(2 * np.pi * freq * t).astype(np.float32))
    tail = y[out_rate // 4 :]
    return float(20 * np.log10(np.sqrt(np.mean(tail**2)) / np.sqrt(0.5) + 1e-12))


@pytest.mark.parametrize("in_rate", [24000, 48000])
@pytest.mark.parametrize("freq", [300, 1000, 3400, 6000])
def test_fir_converter_passes_the_speech_band(in_rate, freq):
    assert abs(_tone_gain_db(in_rate, 16000, freq)) < 0.5


@pytest.mark.parametrize(
    "in_rate,freq",
    [(24000, 8500), (24000, 9000), (24000, 11500), (48000, 8500), (48000, 9000), (48000, 15000), (48000, 20000)],
)
def test_fir_converter_rejects_what_would_alias(in_rate, freq):
    assert _tone_gain_db(in_rate, 16000, freq) < -40.0


def test_fir_converter_leaves_no_image_above_the_source_band():
    """Upsampling a 7.5 kHz tone must not leave its mirror at 8.5 kHz."""
    rs = _FirResampler(16000, 48000)
    t = np.arange(16000) / 16000
    y = rs.push(np.sin(2 * np.pi * 7500 * t).astype(np.float32))[12000:]
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 48000)
    tone = spectrum[(freqs > 7300) & (freqs < 7700)].max()
    image = spectrum[(freqs > 8300) & (freqs < 8700)].max()
    assert 20 * np.log10(image / tone) < -30.0


RATE_CHANGES = [(48000, 24000), (16000, 48000), (48000, 8000), (24000, 16000), (8000, 48000)]


@pytest.mark.parametrize("before,after", RATE_CHANGES)
@pytest.mark.parametrize("frames_before", [1, 7, 50, 53])
def test_a_rate_change_keeps_the_fresh_start_delay(before, after, frames_before):
    """Queued output at the old rate is dropped, and what is still inside the
    16 kHz path counts towards the lead instead of being added on top of it."""
    ns = DTLNNoiseSuppressor(strength=0.0)
    n = before // 100
    _run(ns, _noise(before, 1.0)[: n * frames_before], before, n)
    pcm = _noise(after, 2.0)
    out = _run(ns, pcm, after, after // 100)
    assert ns._underruns == 0
    assert float(np.sqrt((out[after:] ** 2).mean())) > 0.01
    assert _delay_ms(pcm.astype(np.float64) / 32768.0, out, after) <= MAX_DELAY_MS


@pytest.mark.parametrize("frame_ms", [7, 20, 33])
def test_a_rate_change_never_runs_dry_at_other_frame_sizes(frame_ms):
    ns = DTLNNoiseSuppressor(strength=0.0)
    for sr in (48000, 24000, 16000, 48000, 8000, 32000):
        _run(ns, _noise(sr, 1.0), sr, sr * frame_ms // 1000)
    assert ns._underruns == 0


@pytest.mark.parametrize("in_rate,out_rate", [(48000, 16000), (24000, 16000), (16000, 24000), (16000, 48000)])
def test_fir_converter_is_chosen_and_keeps_sample_count(in_rate, out_rate):
    rs = _make_resampler(in_rate, out_rate)
    assert isinstance(rs, _FirResampler)
    x = np.random.default_rng(1).standard_normal(in_rate).astype(np.float32)
    out = np.concatenate([rs.push(x[i : i + 237]) for i in range(0, len(x), 237)])
    assert abs(len(out) - out_rate) <= 1


def _tone_gain_db(in_rate: int, out_rate: int, freq: float) -> float:
    rs = _FirResampler(in_rate, out_rate)
    t = np.arange(in_rate) / in_rate
    y = rs.push(np.sin(2 * np.pi * freq * t).astype(np.float32))
    tail = y[out_rate // 4 :]
    return float(20 * np.log10(np.sqrt(np.mean(tail**2)) / np.sqrt(0.5) + 1e-12))


@pytest.mark.parametrize("in_rate", [24000, 48000])
@pytest.mark.parametrize("freq", [300, 1000, 3400, 6000])
def test_fir_converter_passes_the_speech_band(in_rate, freq):
    assert abs(_tone_gain_db(in_rate, 16000, freq)) < 0.5


@pytest.mark.parametrize(
    "in_rate,freq",
    [(24000, 8500), (24000, 9000), (24000, 11500), (48000, 8500), (48000, 9000), (48000, 15000), (48000, 20000)],
)
def test_fir_converter_rejects_what_would_alias(in_rate, freq):
    assert _tone_gain_db(in_rate, 16000, freq) < -40.0


def test_fir_converter_leaves_no_image_above_the_source_band():
    """Upsampling a 7.5 kHz tone must not leave its mirror at 8.5 kHz."""
    rs = _FirResampler(16000, 48000)
    t = np.arange(16000) / 16000
    y = rs.push(np.sin(2 * np.pi * 7500 * t).astype(np.float32))[12000:]
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 48000)
    tone = spectrum[(freqs > 7300) & (freqs < 7700)].max()
    image = spectrum[(freqs > 8300) & (freqs < 8700)].max()
    assert 20 * np.log10(image / tone) < -30.0


def test_a_rate_change_drops_queued_output_and_starts_again():
    ns = DTLNNoiseSuppressor(strength=0.0)
    _run(ns, _noise(48000, 1.0), 48000, 480)
    assert len(ns._out_native) > 0
    out = _run(ns, _noise(24000, 2.0), 24000, 240)
    assert ns._underruns == 0
    assert float(np.sqrt((out[24000:] ** 2).mean())) > 0.01
    delay = _delay_ms(_noise(24000, 2.0).astype(np.float64) / 32768.0, out, 24000)
    assert delay <= MAX_DELAY_MS + 10.0


def test_fir_converter_output_does_not_depend_on_chunking():
    x = np.random.default_rng(2).standard_normal(24000).astype(np.float32)

    def run(chunk):
        rs = _FirResampler(24000, 16000)
        return np.concatenate([rs.push(x[i : i + chunk]) for i in range(0, len(x), chunk)])

    np.testing.assert_allclose(run(37), run(240), atol=1e-5)
    np.testing.assert_allclose(run(240), _FirResampler(24000, 16000).push(x), atol=1e-5)
