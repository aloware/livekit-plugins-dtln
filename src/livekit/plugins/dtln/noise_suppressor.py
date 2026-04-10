"""DTLN noise suppression as a LiveKit FrameProcessor.

Implements the Dual-Signal Transformation LSTM Network (DTLN) from:
  Noise Reduction with DTLN (Westhausen & Meyer, Interspeech 2020)
  https://github.com/breizhn/DTLN

Uses ONNX Runtime for in-process inference — no cloud API required.
Drop-in replacement for livekit-plugins-noise-cancellation / ai-coustics.
"""

import logging
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
# rfft of 512-sample block gives 257 unique frequency bins
_N_BINS = _BLOCK_LEN // 2 + 1

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

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
        remove_background_speech: bool = False,
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

        # Dry queue — tracks original 16 kHz samples aligned with DTLN pipeline
        # latency so wet/dry blending stays in sync
        self._dry_queue = np.zeros(0, dtype=np.float32)

        # Wet/dry blend: 0.0 = full bypass, 1.0 = full suppression
        self._strength = max(0.0, min(1.0, strength))

        # Debug logging: logs mask stats + RMS every 100 blocks
        self._debug_logging = debug_logging
        self._debug_frame_count = 0

        # Resamplers — created lazily on the first frame
        self._downsampler: rtc.AudioResampler | None = None
        self._upsampler: rtc.AudioResampler | None = None
        self._native_rate: int = 0

        self._enabled = True

        # Speaker extraction for background speech removal
        self._speaker_extractor = None
        if remove_background_speech:
            from .speaker_extractor import SpeakerExtractor
            self._speaker_extractor = SpeakerExtractor(debug_logging=debug_logging)

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
                self._downsampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=_SAMPLE_RATE,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.MEDIUM,
                )
                self._upsampler = rtc.AudioResampler(
                    input_rate=_SAMPLE_RATE,
                    output_rate=frame.sample_rate,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.MEDIUM,
                )
            else:
                self._downsampler = None
                self._upsampler = None

        # Convert int16 → float32 mono
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

        # Build a mono AudioFrame at the native rate for the resampler
        mono_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        mono_frame = rtc.AudioFrame(
            data=mono_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_int16),
        )

        # Downsample to 16 kHz
        if self._downsampler is not None:
            frames_16k = self._downsampler.push(mono_frame)
        else:
            frames_16k = [mono_frame]

        if not frames_16k:
            return frame  # resampler buffering startup, pass through

        samples_16k = np.concatenate([
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in frames_16k
        ])

        self._input_queue = np.concatenate([self._input_queue, samples_16k])
        self._dry_queue = np.concatenate([self._dry_queue, samples_16k])

        # Process in BLOCK_SHIFT (128-sample) steps; count steps taken.
        n_steps = 0
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
            n_steps += 1

        # Drain exactly what we produced this step — no more, no less.
        # This eliminates the periodic silence that occurs when draining by
        # n_16k (downsampler output) instead of n_steps * _BLOCK_SHIFT.
        # During startup the output_queue fills with pipeline latency (~24ms);
        # in steady state it stays constant.
        n_produced = n_steps * _BLOCK_SHIFT
        if n_produced == 0:
            return frame

        out_16k = self._output_queue[:n_produced]
        self._output_queue = self._output_queue[n_produced:]

        # Speaker isolation: compute embedding on raw audio (better speaker
        # characteristics), apply gain to the pure DTLN output before wet/dry
        raw_16k = self._dry_queue[:n_produced]
        if self._speaker_extractor is not None:
            gain = self._speaker_extractor.process_block(raw_16k)
            out_16k = out_16k * gain
            raw_16k = raw_16k * gain  # also attenuate the dry signal

        # Wet/dry blend: mix denoised with original to prevent over-suppression
        if self._strength < 1.0:
            out_16k = self._strength * out_16k + (1.0 - self._strength) * raw_16k
        self._dry_queue = self._dry_queue[n_produced:]

        # Build 16 kHz AudioFrame and upsample back to native rate
        out_int16_16k = (np.clip(out_16k, -1.0, 1.0) * 32767.0).astype(np.int16)
        out_frame_16k = rtc.AudioFrame(
            data=out_int16_16k.tobytes(),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(out_int16_16k),
        )

        if self._upsampler is not None:
            out_frames = self._upsampler.push(out_frame_16k)
        else:
            out_frames = [out_frame_16k]

        if not out_frames:
            return frame

        out_samples = np.concatenate([
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in out_frames
        ])

        # Trim or pad to exactly match the input frame length
        target = frame.samples_per_channel
        if len(out_samples) > target:
            out_samples = out_samples[:target]
        elif len(out_samples) < target:
            out_samples = np.pad(out_samples, (0, target - len(out_samples)))

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
