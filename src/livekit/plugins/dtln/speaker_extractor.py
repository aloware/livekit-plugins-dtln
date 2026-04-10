"""Speaker-aware background speech removal using WeSpeaker embeddings.

Enrolls the primary speaker from the first few seconds of audio, then
attenuates frames where a different speaker is detected. Uses cosine
similarity between the enrollment embedding and a sliding-window embedding
to produce a soft gain mask.

Requires the WeSpeaker ResNet34 ONNX model (~6 MB quantized).
"""

import logging
import os

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "wespeaker_resnet34.onnx"
)

# Mel filterbank parameters (must match WeSpeaker training config)
_SAMPLE_RATE = 16_000
_N_MELS = 80
_FRAME_LEN_MS = 25
_FRAME_SHIFT_MS = 10
_FRAME_LEN = int(_SAMPLE_RATE * _FRAME_LEN_MS / 1000)   # 400
_FRAME_SHIFT = int(_SAMPLE_RATE * _FRAME_SHIFT_MS / 1000)  # 160
_N_FFT = 512

# Enrollment / inference parameters
_ENROLLMENT_SECONDS = 3.0
_ENROLLMENT_SAMPLES = int(_SAMPLE_RATE * _ENROLLMENT_SECONDS)
_WINDOW_SECONDS = 2.0  # sliding window for per-frame embeddings
_WINDOW_SAMPLES = int(_SAMPLE_RATE * _WINDOW_SECONDS)
_HOP_SECONDS = 0.3  # how often to recompute embedding
_HOP_SAMPLES = int(_SAMPLE_RATE * _HOP_SECONDS)

# Similarity thresholds for soft gating
# WeSpeaker cosine similarity ranges: same speaker ~0.35-0.45, different ~0.15-0.25
# Mixed audio (primary + background) falls ~0.20-0.30 depending on mix ratio
_SIM_HIGH = 0.32   # above this = full pass (same speaker)
_SIM_LOW = 0.18    # below this = full attenuation (different speaker)
_MIN_GAIN = 0.05   # floor gain to avoid complete silence
_ENERGY_THRESHOLD = 0.005  # RMS below this = silence/noise, skip similarity check
_GAIN_SMOOTHING = 0.6  # exponential smoothing factor (higher = faster convergence)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a Mel filterbank matrix [n_mels, n_fft//2+1]."""
    n_bins = n_fft // 2 + 1
    low_freq = 0.0
    high_freq = sr / 2.0

    # Mel scale conversion
    low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m, k] = (f_right - k) / (f_right - f_center)
    return fbank


def _compute_fbank_features(audio: np.ndarray, mel_fb: np.ndarray) -> np.ndarray:
    """Compute log-mel filterbank features from 16kHz float32 audio.

    Returns shape [T, 80] where T is the number of frames.
    """
    # Hamming window
    window = np.hamming(_FRAME_LEN).astype(np.float32)

    # Frame the signal
    n_frames = max(0, (len(audio) - _FRAME_LEN) // _FRAME_SHIFT + 1)
    if n_frames == 0:
        return np.zeros((0, _N_MELS), dtype=np.float32)

    frames = np.stack([
        audio[i * _FRAME_SHIFT : i * _FRAME_SHIFT + _FRAME_LEN]
        for i in range(n_frames)
    ])

    # Apply window and FFT
    frames *= window
    spec = np.fft.rfft(frames, n=_N_FFT)
    power = np.abs(spec) ** 2

    # Apply mel filterbank and log
    mel_energy = power @ mel_fb.T
    mel_energy = np.maximum(mel_energy, 1e-10)
    log_mel = np.log(mel_energy)

    return log_mel.astype(np.float32)


class SpeakerExtractor:
    """Enrolls a target speaker and attenuates background speakers.

    Designed to be chained after DTLN noise suppression. Operates on
    16kHz float32 mono audio.

    Usage:
        extractor = SpeakerExtractor()
        # Feed audio blocks sequentially:
        gain = extractor.process_block(audio_16k_float32)
        output = audio * gain
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        enrollment_seconds: float = _ENROLLMENT_SECONDS,
        similarity_high: float = _SIM_HIGH,
        similarity_low: float = _SIM_LOW,
        debug_logging: bool = False,
    ) -> None:
        self._sess = ort.InferenceSession(model_path)
        self._input_name = self._sess.get_inputs()[0].name

        self._mel_fb = _mel_filterbank(_SAMPLE_RATE, _N_FFT, _N_MELS)

        self._enrollment_samples = int(_SAMPLE_RATE * enrollment_seconds)
        self._sim_high = similarity_high
        self._sim_low = similarity_low
        self._debug_logging = debug_logging

        # State
        self._enrollment_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._enrollment_embedding: np.ndarray | None = None
        self._enrolled = False

        # Sliding window buffer for computing per-frame embeddings
        self._window_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._hop_accumulator: int = 0
        self._current_gain: float = 1.0  # start fully open

        self._debug_count = 0

        # Warmup ONNX
        dummy = np.zeros((1, 10, _N_MELS), dtype=np.float32)
        self._sess.run(None, {self._input_name: dummy})

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Compute a 256-dim speaker embedding from audio samples."""
        features = _compute_fbank_features(audio, self._mel_fb)
        if features.shape[0] < 5:
            return np.zeros(256, dtype=np.float32)
        features = features[np.newaxis, :, :]  # [1, T, 80]
        output = self._sess.run(None, {self._input_name: features})
        emb = output[0][0]  # [256]
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both are L2-normalized

    def _similarity_to_gain(self, sim: float) -> float:
        """Map cosine similarity to a gain value with soft transition."""
        if sim >= self._sim_high:
            return 1.0
        if sim <= self._sim_low:
            return _MIN_GAIN
        # Linear interpolation between low and high
        t = (sim - self._sim_low) / (self._sim_high - self._sim_low)
        return _MIN_GAIN + t * (1.0 - _MIN_GAIN)

    def process_block(self, audio: np.ndarray) -> float:
        """Process a block of 16kHz float32 mono audio.

        Returns a gain value [_MIN_GAIN, 1.0] to apply to the audio.
        During enrollment (first ~3s), always returns 1.0.
        """
        # Phase 1: accumulate enrollment audio
        if not self._enrolled:
            self._enrollment_buffer = np.concatenate([self._enrollment_buffer, audio])
            if len(self._enrollment_buffer) >= self._enrollment_samples:
                self._enrollment_embedding = self._get_embedding(
                    self._enrollment_buffer[:self._enrollment_samples]
                )
                self._enrolled = True
                # Start window buffer empty — don't seed with enrollment audio,
                # so the first post-enrollment window reflects actual incoming audio
                self._window_buffer = np.zeros(0, dtype=np.float32)
                self._enrollment_buffer = np.zeros(0, dtype=np.float32)  # free memory
                if self._debug_logging:
                    logger.debug(
                        "Speaker enrolled (%.1fs, embedding norm=%.3f)",
                        self._enrollment_samples / _SAMPLE_RATE,
                        float(np.linalg.norm(self._enrollment_embedding)),
                    )
            return 1.0

        # Phase 2: compute similarity on sliding window
        self._window_buffer = np.concatenate([self._window_buffer, audio])
        self._hop_accumulator += len(audio)

        # Only recompute embedding every _HOP_SAMPLES
        if self._hop_accumulator >= _HOP_SAMPLES:
            self._hop_accumulator = 0
            window = self._window_buffer[-_WINDOW_SAMPLES:]

            # Energy gate: if audio is mostly silence/noise, skip similarity
            # check and keep current gain — let DTLN handle ambient noise.
            # Only apply speaker isolation when speech is actually present.
            window_rms = float(np.sqrt(np.mean(window ** 2)))
            if window_rms < _ENERGY_THRESHOLD:
                target_gain = 1.0  # no speech detected, pass through
            else:
                emb = self._get_embedding(window)
                sim = self._cosine_similarity(self._enrollment_embedding, emb)
                target_gain = self._similarity_to_gain(sim)

            # Smooth gain transitions to avoid audible pumping
            self._current_gain += _GAIN_SMOOTHING * (target_gain - self._current_gain)

            if self._debug_logging and self._debug_count % 50 == 0:
                logger.debug(
                    "Speaker: rms=%.4f gain=%.3f target=%.3f",
                    window_rms, self._current_gain, target_gain,
                )
            self._debug_count += 1

        # Keep window buffer bounded
        if len(self._window_buffer) > _WINDOW_SAMPLES * 2:
            self._window_buffer = self._window_buffer[-_WINDOW_SAMPLES:]

        return self._current_gain

    @property
    def enrolled(self) -> bool:
        return self._enrolled

    @property
    def enrollment_embedding(self) -> np.ndarray | None:
        return self._enrollment_embedding
