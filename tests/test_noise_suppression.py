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


def main():
    print("DTLN Noise Suppression — Integration Test")
    print("=" * 50)

    results = []
    for name, threshold in TEST_FILES.items():
        results.append(test_file(name, threshold))

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
