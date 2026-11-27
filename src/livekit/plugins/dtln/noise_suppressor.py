"""DTLN noise suppression as a LiveKit FrameProcessor.

Implements the Dual-Signal Transformation LSTM Network (DTLN) from:
  Noise Reduction with DTLN (Westhausen & Meyer, Interspeech 2020)
  https://github.com/breizhn/DTLN

Uses ONNX Runtime for in-process inference — no cloud API required.
Drop-in replacement for livekit-plugins-noise-cancellation / ai-coustics.
"""

import logging
import math
import os
import urllib.request

import numpy as np
import onnxruntime as ort
from livekit import rtc

logger = logging.getLogger(__name__)

# DTLN requires 16 kHz mono audio (fixed by the pretrained model)
_SAMPLE_RATE = 16_000
# 32 ms window, 8 ms shift (75% overlap)
_BLOCK_LEN = 512
_BLOCK_SHIFT = 128

# Silence put in front of the output, in 16 kHz samples.  The core only emits
# whole 128-sample hops, so up to 127 samples of a frame can still be inside it
# when the frame has to be answered.  One hop of lead covers that; the extra
# samples absorb the rounding of the two rate converters.
_HOP_RESERVE_16K = _BLOCK_SHIFT + 8
# Denoised audio held back before the first frame is released when the rate
# converters buffer audio of their own (see _SoxrResampler), in 16 kHz samples.
_PRIME_RESERVE_16K = _BLOCK_LEN
# Largest up or down factor the short FIR converter handles.  Rates outside it
# (44.1 kHz needs 160/441) would need thousands of taps per frame.
_MAX_FIR_FACTOR = 8
# rfft of 512-sample block gives 257 unique frequency bins
_N_BINS = _BLOCK_LEN // 2 + 1
# Overlap-add latency: a sample only leaves the synthesis buffer once the
# window has slid past it, so the denoised (wet) signal trails the input by
# one window minus one shift = 384 samples (24 ms at 16 kHz).
_OVERLAP_ADD_LATENCY = _BLOCK_LEN - _BLOCK_SHIFT

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def _silent_like(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    """A frame of digital silence matching ``frame``'s shape."""
    return rtc.AudioFrame(
        data=b"\x00\x00" * (frame.samples_per_channel * frame.num_channels),
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
        samples_per_channel=frame.samples_per_channel,
    )


class _HFRolloffFilter:
    """Stateful high-shelf cut above ~5kHz for 16kHz audio.

    DTLN's spectral mask can leave high frequencies untouched while it
    suppresses low/mid, which sounds thin and metallic. This lowers the band
    above ~5kHz to 30% of its level (-10.5 dB, the floor the earlier 70/30
    wet/dry blend aimed for) and leaves the speech band alone: under 0.5 dB of
    loss up to 3kHz, 0.9 dB at 3.4kHz.

    One biquad (RBJ high shelf, S=1) in transposed direct form II, with state
    carried across calls, so any chunk size gives the same output.
    """

    def __init__(self, sr: float = 16000.0, shelf_hz: float = 5000.0, floor: float = 0.3):
        a_lin = np.sqrt(floor)  # shelf amplitude: sqrt of the high-band gain
        w0 = 2.0 * np.pi * shelf_hz / sr
        cos_w0 = np.cos(w0)
        alpha = np.sin(w0) / np.sqrt(2.0)  # S = 1
        two_sqrt_a_alpha = 2.0 * np.sqrt(a_lin) * alpha
        a0 = (a_lin + 1) - (a_lin - 1) * cos_w0 + two_sqrt_a_alpha
        self._b = (
            a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 + two_sqrt_a_alpha) / a0,
            -2.0 * a_lin * ((a_lin - 1) + (a_lin + 1) * cos_w0) / a0,
            a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 - two_sqrt_a_alpha) / a0,
        )
        self._a = (
            2.0 * ((a_lin - 1) - (a_lin + 1) * cos_w0) / a0,
            ((a_lin + 1) - (a_lin - 1) * cos_w0 - two_sqrt_a_alpha) / a0,
        )
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return the filtered samples; ``samples`` is not modified."""
        b0, b1, b2 = self._b
        a1, a2 = self._a
        z1, z2 = self._z1, self._z2
        out = np.empty(len(samples), dtype=np.float32)
        for i, x in enumerate(samples.astype(np.float64)):
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            out[i] = y
        self._z1, self._z2 = z1, z2
        return out


class _FirResampler:
    """Streaming rational rate converter with under 1 ms of delay.

    Zero-stuffs by ``up``, low-passes with a Kaiser-windowed sinc and keeps
    every ``down``-th sample.  The filter is centred on the lower Nyquist
    frequency with a transition band 3/8 of it wide (6.5-9.5 kHz when the lower
    rate is 16 kHz) and about 50 dB of stopband rejection.  Filter history and
    decimation phase are carried across calls, so any chunk size gives the same
    output, and nothing is held back beyond the filter's own group delay.
    """

    holds_audio = False

    def __init__(self, in_rate: int, out_rate: int) -> None:
        g = math.gcd(in_rate, out_rate)
        self._up = out_rate // g
        self._down = in_rate // g
        hi = in_rate * self._up
        nyquist = min(in_rate, out_rate) / 2.0
        atten_db = 50.0
        width = 0.375 * nyquist
        n = int(math.ceil((atten_db - 8.0) / (2.285 * 2.0 * math.pi * width / hi))) | 1
        k = np.arange(n) - (n - 1) / 2.0
        taps = np.sinc(2.0 * nyquist / hi * k) * np.kaiser(n, 0.1102 * (atten_db - 8.7))
        self._taps = (taps / taps.sum() * self._up).astype(np.float32)
        self._hist = np.zeros(n - 1, dtype=np.float32)
        self._phase = 0

    @staticmethod
    def supports(in_rate: int, out_rate: int) -> bool:
        g = math.gcd(in_rate, out_rate)
        return max(in_rate // g, out_rate // g) <= _MAX_FIR_FACTOR

    def push(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        stuffed = np.zeros(len(samples) * self._up, dtype=np.float32)
        stuffed[:: self._up] = samples
        buf = np.concatenate([self._hist, stuffed])
        filtered = np.convolve(buf, self._taps, mode="valid")
        out = filtered[self._phase :: self._down]
        self._phase += len(out) * self._down - len(filtered)
        self._hist = buf[len(buf) - len(self._hist) :]
        return out.astype(np.float32)


class _SoxrResampler:
    """LiveKit's resampler behind the same float interface, for any rate.

    It buffers around 20 ms per direction and releases it unevenly, so the
    suppressor has to queue output before it starts answering frames.
    """

    holds_audio = True

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self._in_rate = in_rate
        self._resampler = rtc.AudioResampler(
            input_rate=in_rate,
            output_rate=out_rate,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.MEDIUM,
        )

    def push(self, samples: np.ndarray) -> np.ndarray:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        frames = self._resampler.push(
            rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=self._in_rate,
                num_channels=1,
                samples_per_channel=len(pcm),
            )
        )
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in frames
        ])


def _make_resampler(in_rate: int, out_rate: int) -> "_FirResampler | _SoxrResampler":
    if _FirResampler.supports(in_rate, out_rate):
        return _FirResampler(in_rate, out_rate)
    return _SoxrResampler(in_rate, out_rate)


_DTLN_BASE_URL = "https://github.com/breizhn/DTLN/raw/master/pretrained_model"
_MODEL_FILES = ["model_1.onnx", "model_2.onnx"]


def download_models(models_dir: str = _DEFAULT_MODEL_DIR) -> None:
    """Download pretrained DTLN ONNX models (skips files already present)."""
    os.makedirs(models_dir, exist_ok=True)
    for filename in _MODEL_FILES:
        dest = os.path.join(models_dir, filename)
        if os.path.exists(dest):
            logger.info("Model already present: %s", dest)
            continue
        url = f"{_DTLN_BASE_URL}/{filename}"
        logger.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, dest)
        logger.info("Saved to %s", dest)


class DTLNNoiseSuppressor(rtc.FrameProcessor[rtc.AudioFrame]):
    """In-process DTLN noise suppressor that works with self-hosted LiveKit.

    Pass to AudioInputOptions(noise_cancellation=DTLNNoiseSuppressor(...)).
    Each instance is stateful (LSTM state persists across frames) — create
    one instance per call session.

    Args:
        model_1_path: Path to model_1.onnx (spectral masking stage).
        model_2_path: Path to model_2.onnx (time-domain refinement stage).
    """

    def __init__(
        self,
        model_1_path: str = os.path.join(_DEFAULT_MODEL_DIR, "model_1.onnx"),
        model_2_path: str = os.path.join(_DEFAULT_MODEL_DIR, "model_2.onnx"),
        strength: float = 0.5,
        debug_logging: bool = False,
    ) -> None:
        self._sess1 = ort.InferenceSession(model_1_path)
        self._sess2 = ort.InferenceSession(model_2_path)

        # Input/output tensor names (read from model to avoid hardcoding)
        self._m1_in_mag = self._sess1.get_inputs()[0].name
        self._m1_in_state = self._sess1.get_inputs()[1].name
        self._m2_in_time = self._sess2.get_inputs()[0].name
        self._m2_in_state = self._sess2.get_inputs()[1].name

        # LSTM states — shape read from model (e.g. [1, 2, 128, 2])
        state1_shape = self._sess1.get_inputs()[1].shape
        state2_shape = self._sess2.get_inputs()[1].shape
        self._state1 = np.zeros(state1_shape, dtype=np.float32)
        self._state2 = np.zeros(state2_shape, dtype=np.float32)

        # Rolling 512-sample buffers at 16 kHz (float32)
        self._in_buf = np.zeros(_BLOCK_LEN, dtype=np.float32)
        self._out_buf = np.zeros(_BLOCK_LEN, dtype=np.float32)

        # Queues for handling arbitrary incoming frame sizes
        self._input_queue = np.zeros(0, dtype=np.float32)
        self._output_queue = np.zeros(0, dtype=np.float32)

        # Dry queue — original 16 kHz samples kept for wet/dry blending.
        # Pre-padded with _OVERLAP_ADD_LATENCY samples so that draining the dry
        # and output queues by the same count also lines them up in TIME:
        # the wet path leaves the overlap-add loop 384 samples late, so the
        # dry path has to start 384 samples late too.  Without the padding
        # each output sample is mixed with a 24 ms newer copy of itself,
        # which combs the signal (~3 dB of cancellation on speech).
        self._dry_queue = np.zeros(_OVERLAP_ADD_LATENCY, dtype=np.float32)

        # Denoised audio at the caller's own sample rate, waiting to be handed
        # back.  Draining through this is what keeps the output independent of
        # how the caller chunks the input: each call gets exactly as many
        # samples as it gave us, and the remainder waits here for the next one
        # instead of being zero-filled or discarded.
        self._out_native = np.zeros(0, dtype=np.float32)
        self._primed = False
        self._underruns = 0

        # Wet/dry blend: 0.0 = full bypass, 1.0 = full suppression
        self._strength = max(0.0, min(1.0, strength))

        # Debug logging: logs mask stats + RMS every 100 blocks
        self._debug_logging = debug_logging
        self._debug_frame_count = 0

        # Resamplers — created lazily on the first frame
        self._downsampler: _FirResampler | _SoxrResampler | None = None
        self._upsampler: _FirResampler | _SoxrResampler | None = None
        self._native_rate: int = 0

        self._enabled = True

        # High-frequency rolloff filter (stateful IIR, works on any chunk size)
        self._hf_rolloff = _HFRolloffFilter()

        # Pre-warm ONNX Runtime's JIT compiler so the first real frame
        # doesn't stall the audio pipeline (~500ms cold-start otherwise).
        self._warmup()

    def _warmup(self) -> None:
        dummy_mag = np.zeros((1, 1, _N_BINS), dtype=np.float32)
        dummy_time = np.zeros((1, 1, _BLOCK_LEN), dtype=np.float32)
        self._sess1.run(None, {self._m1_in_mag: dummy_mag, self._m1_in_state: self._state1})
        self._sess2.run(None, {self._m2_in_time: dummy_time, self._m2_in_state: self._state2})
        # Reset states — warmup outputs shouldn't carry over into real audio
        state1_shape = self._sess1.get_inputs()[1].shape
        state2_shape = self._sess2.get_inputs()[1].shape
        self._state1 = np.zeros(state1_shape, dtype=np.float32)
        self._state2 = np.zeros(state2_shape, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # FrameProcessor interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        # Lazily create resamplers when we learn the incoming sample rate.
        if frame.sample_rate != self._native_rate:
            self._native_rate = frame.sample_rate
            if frame.sample_rate != _SAMPLE_RATE:
                self._downsampler = _make_resampler(frame.sample_rate, _SAMPLE_RATE)
                self._upsampler = _make_resampler(_SAMPLE_RATE, frame.sample_rate)
            else:
                self._downsampler = None
                self._upsampler = None

        # Convert int16 → float32 mono
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

        # Downsample to 16 kHz.  An empty result is a buffering converter
        # still filling at startup.  The frame is not passed through: it has
        # not been denoised, and splicing raw audio into a denoised stream is
        # audible.  Fall through and let the output queue decide what to hand
        # back.
        if self._downsampler is not None:
            samples_16k = self._downsampler.push(samples)
        else:
            samples_16k = samples

        self._input_queue = np.concatenate([self._input_queue, samples_16k])
        self._dry_queue = np.concatenate([self._dry_queue, samples_16k])

        # Process in BLOCK_SHIFT (128-sample) steps.
        while len(self._input_queue) >= _BLOCK_SHIFT:
            new = self._input_queue[:_BLOCK_SHIFT]
            self._input_queue = self._input_queue[_BLOCK_SHIFT:]

            self._in_buf[:-_BLOCK_SHIFT] = self._in_buf[_BLOCK_SHIFT:]
            self._in_buf[-_BLOCK_SHIFT:] = new

            denoised = self._infer_block(self._in_buf)

            self._out_buf[:-_BLOCK_SHIFT] = self._out_buf[_BLOCK_SHIFT:]
            self._out_buf[-_BLOCK_SHIFT:] = 0.0
            self._out_buf += denoised

            self._output_queue = np.concatenate([
                self._output_queue, self._out_buf[:_BLOCK_SHIFT]
            ])

        # Move everything the block loop has finished into the native-rate
        # queue.  Draining by the number of samples that went in this call
        # keeps the 16 kHz side aligned, but it cannot by itself make the
        # native-rate result the length of the frame, because both resamplers
        # hold their own partial samples.  Whatever comes out is queued rather
        # than fitted to this frame.
        n_16k = min(len(self._output_queue), len(samples_16k))
        if n_16k:
            out_16k = self._output_queue[:n_16k]
            self._output_queue = self._output_queue[n_16k:]

            # High-shelf cut to prevent metallic artifacts
            out_16k = self._hf_rolloff.process(out_16k)

            # Wet/dry blend: mix denoised with original to prevent over-suppression.
            # Both queues carry the same 384-sample offset, so dry_16k is the
            # stretch of input that produced out_16k.
            if self._strength < 1.0:
                dry_16k = self._dry_queue[:n_16k]
                out_16k = self._strength * out_16k + (1.0 - self._strength) * dry_16k
            self._dry_queue = self._dry_queue[n_16k:]

            # Upsample back to the native rate
            out_16k = np.clip(out_16k, -1.0, 1.0).astype(np.float32)
            if self._upsampler is not None:
                out_16k = self._upsampler.push(out_16k)
            self._out_native = np.concatenate([self._out_native, out_16k])

        # Hand back exactly one frame's worth, taken from the queue rather than
        # from whatever this particular call happened to finish.
        target = frame.samples_per_channel
        if not self._primed and not self._converters_hold_audio():
            # Nothing buffers except the hop loop, so a fixed lead of silence
            # is enough and the first frame is answered straight away.
            lead = math.ceil(_HOP_RESERVE_16K * self._resample_ratio())
            self._out_native = np.concatenate([
                np.zeros(lead, dtype=np.float32), self._out_native
            ])
            self._primed = True
        if not self._primed:
            reserve = target + int(_PRIME_RESERVE_16K * self._resample_ratio())
            if len(self._out_native) < reserve:
                # Still filling.  Emit silence rather than the raw frame: the
                # caller's audio has not been denoised yet, and splicing it in
                # is both audible and a privacy surprise for a suppressor.
                return _silent_like(frame)
            self._primed = True

        if len(self._out_native) < target:
            # Should not happen once primed.  Pad rather than drop the frame,
            # and say so, because a recurring underrun means the reserve is
            # sized wrong for this rate.
            self._underruns += 1
            if self._underruns in (1, 10, 100, 1000):
                logger.warning(
                    "DTLN output underrun (%d so far): have %d samples, need %d",
                    self._underruns,
                    len(self._out_native),
                    target,
                )
            out_samples = np.pad(self._out_native, (0, target - len(self._out_native)))
            self._out_native = np.zeros(0, dtype=np.float32)
        else:
            out_samples = self._out_native[:target]
            self._out_native = self._out_native[target:]

        # Restore original channel count (duplicate mono → stereo if needed)
        if frame.num_channels > 1:
            out_samples = np.repeat(out_samples, frame.num_channels)

        out_int16 = (np.clip(out_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        return rtc.AudioFrame(
            data=out_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _resample_ratio(self) -> float:
        """Native samples per 16 kHz sample, for sizing the reserve."""
        return (self._native_rate / _SAMPLE_RATE) if self._native_rate else 1.0

    def _converters_hold_audio(self) -> bool:
        return bool(
            (self._downsampler is not None and self._downsampler.holds_audio)
            or (self._upsampler is not None and self._upsampler.holds_audio)
        )

    def _close(self) -> None:
        self._enabled = False
        self._sess1 = None  # type: ignore[assignment]
        self._sess2 = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Internal inference                                                   #
    # ------------------------------------------------------------------ #

    def _infer_block(self, block: np.ndarray) -> np.ndarray:
        """Run one 512-sample block through both DTLN models.

        Returns a 512-sample denoised block (float32, range ~[-1, 1]).
        """
        # --- Model 1: spectral masking ---
        spec = np.fft.rfft(block)  # complex128, shape (257,)
        mag = np.abs(spec).reshape(1, 1, _N_BINS).astype(np.float32)

        out1 = self._sess1.run(None, {
            self._m1_in_mag: mag,
            self._m1_in_state: self._state1,
        })
        mask = out1[0].reshape(_N_BINS)   # (257,)
        self._state1 = out1[1]

        # Apply mask in frequency domain, reconstruct time domain
        enhanced_spec = mask * spec       # element-wise, preserves phase
        enhanced_time = np.fft.irfft(enhanced_spec, n=_BLOCK_LEN).astype(np.float32)

        # --- Model 2: time-domain refinement ---
        time_in = enhanced_time.reshape(1, 1, _BLOCK_LEN).astype(np.float32)
        out2 = self._sess2.run(None, {
            self._m2_in_time: time_in,
            self._m2_in_state: self._state2,
        })
        denoised = out2[0].reshape(_BLOCK_LEN)  # (512,)
        self._state2 = out2[1]

        if self._debug_logging and self._debug_frame_count % 100 == 0:
            logger.debug(
                "DTLN block: mask_mean=%.3f mask_min=%.3f mask_max=%.3f "
                "input_rms=%.5f output_rms=%.5f strength=%.2f",
                float(mask.mean()), float(mask.min()), float(mask.max()),
                float(np.sqrt(np.mean(block**2))),
                float(np.sqrt(np.mean(denoised**2))),
                self._strength,
            )
        self._debug_frame_count += 1

        return denoised
