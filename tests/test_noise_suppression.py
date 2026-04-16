"""Integration test: run original audio files through DTLNNoiseSuppressor and verify noise reduction."""

import os
import subprocess
import tempfile
import wave

import numpy as np
from livekit import rtc
from livekit.plugins.dtln import DTLNNoiseSuppressor

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "audio", "originals")

# Minimum expected RMS reduction (%) for each file.
# These are conservative thresholds — real reduction is typically higher.
TEST_FILES = {
    "krisp-original.mp3": 10,
    "taxi-sample.mp3": 10,
    "noproblem_raw.wav": 3,  # clean speech — expect minimal suppression
}


def read_wav_int16(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (int16 samples mono, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)[:, 0]  # take first channel
    return samples, sr


def convert_to_wav(src: str) -> str:
    """Convert mp3 to a temporary 16-bit PCM WAV via ffmpeg. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name],
        check=True,
        capture_output=True,
    )
    return tmp.name


def process_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run int16 mono samples through DTLNNoiseSuppressor, return int16 output."""
    ns = DTLNNoiseSuppressor()

    chunk_ms = 20
    chunk_size = sample_rate * chunk_ms // 1000
    output_chunks = []

    for i in range(0, len(samples) - chunk_size + 1, chunk_size):
        chunk = samples[i : i + chunk_size]
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=chunk_size,
        )
        out_frame = ns._process(frame)
        out_samples = np.frombuffer(out_frame.data, dtype=np.int16)
        output_chunks.append(out_samples)

    return np.concatenate(output_chunks)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def test_file(name: str, min_reduction_pct: int) -> bool:
    src = os.path.join(AUDIO_DIR, name)
    assert os.path.exists(src), f"Missing: {src}"

    tmp_wav = None
    if name.endswith(".mp3"):
        tmp_wav = convert_to_wav(src)
        wav_path = tmp_wav
    else:
        wav_path = src

    try:
        samples, sr = read_wav_int16(wav_path)
        output = process_audio(samples, sr)

        # Trim to same length for fair comparison
        n = min(len(samples), len(output))
        in_rms = rms(samples[:n])
        out_rms = rms(output[:n])
        reduction_pct = (1 - out_rms / in_rms) * 100

        print(f"  {name}")
        print(f"    Input RMS:  {in_rms:.1f}")
        print(f"    Output RMS: {out_rms:.1f}")
        print(f"    Reduction:  {reduction_pct:.1f}%")

        if reduction_pct < min_reduction_pct:
            print(f"    FAIL: expected >= {min_reduction_pct}% reduction")
            return False
        else:
            print(f"    PASS")
            return True
    finally:
        if tmp_wav:
            os.unlink(tmp_wav)


def longest_zero_run(samples: np.ndarray) -> int:
    """Return the length of the longest contiguous run of exact-zero samples."""
    is_zero = (samples == 0)
    if not np.any(is_zero):
        return 0
    # Count consecutive zeros using a run-length approach
    longest = current = 0
    for v in is_zero:
        if v:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def test_non_aligned_frame_sizes() -> bool:
    """Regression test: non-aligned frame sizes must not produce silence gaps.

    Frame sizes that aren't multiples of BLOCK_SHIFT (128) at 16kHz — e.g.
    10ms (160), 20ms (320), 30ms (480) — previously triggered pad/truncate
    logic that appended zeros or dropped samples every other frame, causing
    audible crackling. The symptom was contiguous runs of zero samples
    (~32-96 samples long) embedded in the output stream.

    This test feeds continuous white noise through _process() and verifies
    the concatenated output contains no long zero-runs. Real denoised audio
    over a non-silent input cannot produce extended runs of exact zeros,
    so any such run is a pad/truncate regression.
    """
    sample_rate = 16000
    n_frames = 100
    startup_frames = 4  # skip pipeline-latency transient (~24ms)
    # Max plausible zero-run in genuine denoised output of a continuous
    # signal. The bug produces runs of 32-96 samples; set threshold safely
    # below that.
    max_allowed_zero_run = 16
    rng = np.random.default_rng(seed=42)

    all_ok = True
    for chunk_ms in (10, 20, 30):
        chunk_size = sample_rate * chunk_ms // 1000
        signal = (rng.standard_normal(chunk_size * n_frames) * 0.3 * 32767).astype(np.int16)

        ns = DTLNNoiseSuppressor()
        out_frames = []
        for i in range(n_frames):
            chunk = signal[i * chunk_size : (i + 1) * chunk_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=chunk_size,
            )
            out_frame = ns._process(frame)
            out_frames.append(np.frombuffer(out_frame.data, dtype=np.int16))

        # Concatenate the steady-state portion (skip startup transient)
        steady = np.concatenate(out_frames[startup_frames:])
        zero_run = longest_zero_run(steady)

        print(f"  {chunk_ms}ms frames ({chunk_size} samples):")
        print(f"    frames processed: {n_frames} (skipped startup: {startup_frames})")
        print(f"    longest zero-run: {zero_run} samples (threshold: {max_allowed_zero_run})")

        if zero_run > max_allowed_zero_run:
            print(f"    FAIL: zero-run exceeds threshold — pad/truncate regression")
            all_ok = False
        else:
            print(f"    PASS")

    return all_ok


def main():
    print("DTLN Noise Suppression — Integration Test")
    print("=" * 50)

    results = []
    for name, threshold in TEST_FILES.items():
        results.append(test_file(name, threshold))

    print()
    print("Frame-alignment regression test:")
    results.append(test_non_aligned_frame_sizes())

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
