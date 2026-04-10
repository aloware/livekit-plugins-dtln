"""Run Deepgram Nova 3 STT on audio files.

Usage:
    python tests/run_stt.py docs/audio/originals/gym.wav
    python tests/run_stt.py docs/audio/dtln-gym.wav docs/audio/dtln-spk-gym.wav
    python tests/run_stt.py --all-demo    # run on all demo audio files
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
if not API_KEY:
    print("Error: DEEPGRAM_API_KEY not set. Add it to .env")
    sys.exit(1)

DEMO_FILES = [
    ("Original (gym)", "docs/audio/originals/gym.wav"),
    ("DTLN (gym)", "docs/audio/dtln-gym.wav"),
    ("DTLN + Voice Isolation (gym)", "docs/audio/dtln-spk-gym.wav"),
    ("Krisp BVC (gym)", "docs/audio/competitors/gym-krisp-bvc.wav"),
    ("AI-coustics QUAIL_VF_L (gym)", "docs/audio/competitors/gym-quail-vfl.wav"),
    ("Krisp NC (gym)", "docs/audio/competitors/gym-krisp-nc.wav"),
    ("AI-coustics QUAIL_L (gym)", "docs/audio/competitors/gym-quail-l.wav"),
]


def convert_to_wav_16k(path: str) -> str:
    """Convert any audio to 16kHz mono WAV for Deepgram."""
    if path.endswith(".wav"):
        return path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name


def transcribe(path: str) -> str:
    """Send audio to Deepgram Nova 3 and return the transcript."""
    wav_path = convert_to_wav_16k(path)
    try:
        with open(wav_path, "rb") as f:
            audio_data = f.read()
    finally:
        if wav_path != path:
            os.unlink(wav_path)

    url = "https://api.deepgram.com/v1/listen?model=nova-3&language=en"
    req = urllib.request.Request(
        url,
        data=audio_data,
        headers={
            "Authorization": f"Token {API_KEY}",
            "Content-Type": "audio/wav",
        },
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    if "--all-demo" in args:
        files = [(label, path) for label, path in DEMO_FILES if os.path.exists(path)]
    else:
        files = [(os.path.basename(p), p) for p in args]

    print(f"Running Deepgram Nova 3 STT on {len(files)} file(s)...\n")

    for label, path in files:
        if not os.path.exists(path):
            print(f"  {label}: FILE NOT FOUND ({path})")
            continue
        try:
            transcript = transcribe(path)
            print(f"  {label}:")
            print(f"    {transcript}")
            print()
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            print()


if __name__ == "__main__":
    main()
