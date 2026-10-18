"""The high-frequency cut must leave the speech band alone.

It used to be four one-pole smoothers in a row blended 70/30 with the input,
documented as a Butterworth at 4 kHz. That took about 7 dB off at 2 kHz and
10 dB from 3 kHz up, which is where consonants are on a phone line, on every
frame. The cut is meant for the band above that.
"""

import numpy as np
import pytest

from livekit.plugins.dtln.noise_suppressor import _HFRolloffFilter

SR = 16000


def _gain_db(freq: float) -> float:
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    y = _HFRolloffFilter().process(x)
    tail = slice(SR // 4, None)  # past the filter's settling
    return float(20 * np.log10(np.sqrt(np.mean(y[tail] ** 2)) / np.sqrt(np.mean(x[tail] ** 2))))


@pytest.mark.parametrize("freq", [300, 1000, 2000, 3000, 3400])
def test_speech_band_passes(freq):
    assert _gain_db(freq) > -1.0


@pytest.mark.parametrize("freq", [7000, 7500])
def test_high_band_is_cut_to_the_floor(freq):
    assert _gain_db(freq) == pytest.approx(20 * np.log10(0.3), abs=0.5)


def test_output_does_not_depend_on_chunking():
    x = np.random.default_rng(0).standard_normal(SR).astype(np.float32)

    def run(chunk):
        f = _HFRolloffFilter()
        return np.concatenate([f.process(x[i : i + chunk]) for i in range(0, len(x), chunk)])

    np.testing.assert_allclose(run(37), run(160), atol=1e-6)
    np.testing.assert_allclose(run(160), _HFRolloffFilter().process(x), atol=1e-6)
