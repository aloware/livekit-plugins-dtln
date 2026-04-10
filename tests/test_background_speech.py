"""Integration test: verify background speech removal via speaker extraction.

Creates a synthetic mix of two speakers, processes through the speaker extractor,
and verifies that the background speaker is attenuated while the primary speaker
is preserved.
"""

import os
import subprocess
import tempfile
import wave

import numpy as np
from livekit import rtc
from livekit.plugins.dtln import DTLNNoiseSuppressor
from livekit.plugins.dtln.speaker_extractor import SpeakerExtractor

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "audio", "originals")


def read_wav_int16(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (int16 samples mono, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)[:, 0]
    return samples, sr


def convert_to_wav(src: str) -> str:
    """Convert mp3 to a temporary 16-bit PCM WAV via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name],
        check=True,
        capture_output=True,
    )
    return tmp.name


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def rms_in_range(samples: np.ndarray, start_s: float, end_s: float, sr: int) -> float:
    """Compute RMS of a time range within samples."""
    s = int(start_s * sr)
    e = int(end_s * sr)
    return rms(samples[s:e])


def process_audio(samples: np.ndarray, sample_rate: int, remove_bg_speech: bool = False) -> np.ndarray:
    """Run int16 mono samples through DTLNNoiseSuppressor."""
    ns = DTLNNoiseSuppressor(remove_background_speech=remove_bg_speech)
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


def create_mixed_audio(
    primary_samples: np.ndarray,
    background_samples: np.ndarray,
    bg_gain: float = 0.5,
) -> np.ndarray:
    """Mix two int16 audio streams. Primary speaker starts first (enrollment),
    background speaker is mixed in after the enrollment period.

    Layout:
      [0 .. 3s]: primary speaker only (enrollment region)
      [3s .. end]: primary + background at bg_gain level
    """
    enrollment_samples = int(16_000 * 3.0)  # 3 seconds

    # Ensure both are long enough
    min_len = enrollment_samples + 16_000 * 4  # at least 4s after enrollment
    primary = primary_samples[:min_len] if len(primary_samples) >= min_len else np.pad(
        primary_samples, (0, min_len - len(primary_samples))
    )

    # Background: pad with silence for enrollment period, then mix
    bg_padded = np.zeros(len(primary), dtype=np.int16)
    bg_len = min(len(background_samples), len(primary) - enrollment_samples)
    bg_section = background_samples[:bg_len]
    bg_padded[enrollment_samples : enrollment_samples + bg_len] = (bg_section * bg_gain).astype(np.int16)

    # Mix with clipping protection
    mixed = np.clip(
        primary.astype(np.float64) + bg_padded.astype(np.float64),
        -32768, 32767,
    ).astype(np.int16)

    return mixed


def test_speaker_extractor_standalone():
    """Test SpeakerExtractor directly on float32 audio."""
    print("\n--- Test: SpeakerExtractor standalone ---")

    # Load primary speaker
    primary_path = os.path.join(AUDIO_DIR, "noproblem_raw.wav")
    primary_i16, sr = read_wav_int16(primary_path)
    primary_f32 = primary_i16.astype(np.float32) / 32768.0

    # Load background speaker
    krisp_wav = convert_to_wav(os.path.join(AUDIO_DIR, "krisp-original.mp3"))
    try:
        bg_i16, _ = read_wav_int16(krisp_wav)
    finally:
        os.unlink(krisp_wav)
    bg_f32 = bg_i16.astype(np.float32) / 32768.0

    extractor = SpeakerExtractor(debug_logging=True)

    # Phase 1: enrollment with primary speaker (first 3s)
    enrollment_len = int(sr * 3.0)
    chunk_size = int(sr * 0.02)  # 20ms chunks

    gains_enrollment = []
    for i in range(0, enrollment_len, chunk_size):
        chunk = primary_f32[i : i + chunk_size]
        gain = extractor.process_block(chunk)
        gains_enrollment.append(gain)

    assert extractor.enrolled, "Speaker should be enrolled after 3s"
    avg_enrollment_gain = np.mean(gains_enrollment)
    print(f"  Enrollment gains (should be 1.0): avg={avg_enrollment_gain:.3f}")
    assert avg_enrollment_gain == 1.0, "Enrollment phase should always return gain=1.0"

    # Phase 2: feed primary speaker only — gains should stay high
    gains_primary = []
    primary_test = primary_f32[enrollment_len : enrollment_len + int(sr * 3)]
    for i in range(0, len(primary_test) - chunk_size + 1, chunk_size):
        chunk = primary_test[i : i + chunk_size]
        gain = extractor.process_block(chunk)
        gains_primary.append(gain)

    avg_primary_gain = np.mean(gains_primary)
    print(f"  Primary speaker gains: avg={avg_primary_gain:.3f}")

    # Phase 3: feed background speaker only — gains should drop
    extractor2 = SpeakerExtractor()
    # Enroll with primary
    for i in range(0, enrollment_len, chunk_size):
        chunk = primary_f32[i : i + chunk_size]
        extractor2.process_block(chunk)

    gains_background = []
    # Feed 5s of background speaker — first ~2s is warm-up as the sliding
    # window transitions from enrollment audio to background audio
    bg_test = bg_f32[:int(sr * 5)]
    for i in range(0, len(bg_test) - chunk_size + 1, chunk_size):
        chunk = bg_test[i : i + chunk_size]
        gain = extractor2.process_block(chunk)
        gains_background.append(gain)

    # Measure only after warm-up (last 2.5s) for a fair assessment
    warmup_chunks = int(2.5 * sr / chunk_size)
    gains_after_warmup = gains_background[warmup_chunks:]
    avg_bg_gain = np.mean(gains_after_warmup) if gains_after_warmup else np.mean(gains_background)
    avg_bg_gain_full = np.mean(gains_background)
    print(f"  Background speaker gains (full): avg={avg_bg_gain_full:.3f}")
    print(f"  Background speaker gains (after warmup): avg={avg_bg_gain:.3f}")

    # Verify: primary gains should be significantly higher than background gains
    passed = True
    if avg_primary_gain < 0.6:
        print(f"  FAIL: primary speaker gain too low ({avg_primary_gain:.3f} < 0.6)")
        passed = False
    else:
        print(f"  PASS: primary speaker preserved (gain={avg_primary_gain:.3f})")

    if avg_bg_gain > 0.7:
        print(f"  FAIL: background speaker not attenuated ({avg_bg_gain:.3f} > 0.7)")
        passed = False
    else:
        print(f"  PASS: background speaker attenuated (gain={avg_bg_gain:.3f})")

    if avg_primary_gain - avg_bg_gain < 0.15:
        print(f"  FAIL: insufficient gain separation ({avg_primary_gain:.3f} - {avg_bg_gain:.3f} = {avg_primary_gain - avg_bg_gain:.3f} < 0.15)")
        passed = False
    else:
        print(f"  PASS: gain separation = {avg_primary_gain - avg_bg_gain:.3f}")

    return passed


def test_full_pipeline_mixed():
    """Test the full DTLN + speaker extraction pipeline on mixed audio."""
    print("\n--- Test: Full pipeline (DTLN + speaker extraction) on mixed audio ---")

    # Load speakers
    primary_i16, sr = read_wav_int16(os.path.join(AUDIO_DIR, "noproblem_raw.wav"))

    krisp_wav = convert_to_wav(os.path.join(AUDIO_DIR, "krisp-original.mp3"))
    try:
        bg_i16, _ = read_wav_int16(krisp_wav)
    finally:
        os.unlink(krisp_wav)

    # Create mix: primary only for 3s, then primary + background at 50%
    mixed = create_mixed_audio(primary_i16, bg_i16, bg_gain=0.5)

    # Process with DTLN only (no speaker extraction)
    output_dtln_only = process_audio(mixed, sr, remove_bg_speech=False)

    # Process with DTLN + speaker extraction
    output_with_extraction = process_audio(mixed, sr, remove_bg_speech=True)

    # Measure RMS in the mixed region (after 3s enrollment)
    enrollment_end = 3.0
    measure_end = min(len(mixed) / sr, len(output_dtln_only) / sr, len(output_with_extraction) / sr)

    rms_input_mixed = rms_in_range(mixed, enrollment_end, measure_end, sr)
    rms_dtln_only = rms_in_range(output_dtln_only, enrollment_end, measure_end, sr)
    rms_with_extraction = rms_in_range(output_with_extraction, enrollment_end, measure_end, sr)

    print(f"  Mixed region ({enrollment_end:.0f}s - {measure_end:.1f}s):")
    print(f"    Input RMS:              {rms_input_mixed:.1f}")
    print(f"    DTLN only RMS:          {rms_dtln_only:.1f}")
    print(f"    DTLN + extraction RMS:  {rms_with_extraction:.1f}")

    reduction_dtln = (1 - rms_dtln_only / rms_input_mixed) * 100
    reduction_full = (1 - rms_with_extraction / rms_input_mixed) * 100
    extra_reduction = reduction_full - reduction_dtln

    print(f"    DTLN only reduction:    {reduction_dtln:.1f}%")
    print(f"    Full pipeline reduction: {reduction_full:.1f}%")
    print(f"    Extra from extraction:  {extra_reduction:.1f}%")

    passed = True
    if rms_with_extraction >= rms_dtln_only:
        print(f"  FAIL: speaker extraction did not reduce RMS beyond DTLN alone")
        passed = False
    else:
        print(f"  PASS: speaker extraction provides additional reduction")

    return passed


def main():
    print("Background Speech Removal — Integration Test")
    print("=" * 55)

    results = []
    results.append(test_speaker_extractor_standalone())
    results.append(test_full_pipeline_mixed())

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
