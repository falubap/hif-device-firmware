"""features.py against the libraries and the thesis code it re-implements.

    .venv/bin/python -m pytest python -q

The building blocks are checked against the library each one replaces (pywt, scipy) and against
the thesis's own transforms. The end-to-end check against the thesis's feature matrix over every
test row is python/export_windows.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import pywt
from scipy import signal as sps
from scipy import stats

import features as F
import paths

paths.import_thesis()
from hif.transforms.stockwell import Stockwell  # noqa: E402
from hif.transforms.wavelet import Wavelet  # noqa: E402

RNG = np.random.default_rng(7)


def harmonic_signal(seed: int = 0) -> np.ndarray:
    """A 60 Hz fundamental with harmonics up to the 20th and some noise, like a measured window."""
    rng = np.random.default_rng(seed)
    t = np.arange(F.N) / (F.F0 * F.SPC)
    x = sum(rng.uniform(0.01, 1) / h * np.cos(2 * np.pi * F.F0 * h * t + rng.uniform(-np.pi, np.pi))
            for h in range(1, F.MAX_ORDER + 1))
    return x + 0.01 * rng.standard_normal(F.N)


def test_db4_taps_match_pywt():
    w = pywt.Wavelet("db4")
    assert np.array_equal(F.DB4_DEC_LO, np.array(w.dec_lo))
    assert np.allclose(F.DB4_DEC_HI, np.array(w.dec_hi), rtol=0, atol=1e-16)


def test_swt_matches_pywt():
    x = RNG.standard_normal(F.N)
    ref = pywt.swt(x, "db4", level=F.WAVELET_LEVELS, trim_approx=True, norm=True)
    ref = list(ref[1:][::-1]) + [ref[0]]  # [d1..d8, a8]
    for got, want in zip(F.swt_db4(x), ref):
        assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_wavelet_matches_thesis():
    x = harmonic_signal(1)
    want = Wavelet("db4", F.WAVELET_LEVELS).analyze(x.reshape(F.N_CYCLES, F.SPC)).coefficients.real
    assert np.allclose(F.wavelet(x), want, rtol=1e-12, atol=0)


def test_stockwell_matches_thesis():
    x = harmonic_signal(2)
    want = Stockwell(F.MAX_ORDER).analyze(x.reshape(F.N_CYCLES, F.SPC)).coefficients
    assert np.allclose(F.stockwell(x), want, rtol=1e-12, atol=1e-15)


def test_stockwell_and_dft_agree_on_a_steady_harmonic():
    """For a stationary signal both transforms must report each harmonic's peak amplitude."""
    t = np.arange(F.N) / (F.F0 * F.SPC)
    x = 3.0 * np.cos(2 * np.pi * 60 * t) + 0.5 * np.cos(2 * np.pi * 180 * t + 0.3)
    for c in (F.dft(x), F.stockwell(x)):
        assert np.allclose(np.abs(c[0]), 3.0, rtol=1e-6)
        assert np.allclose(np.abs(c[2]), 0.5, rtol=1e-6)


def test_hilbert_rotation_matches_scipy():
    x = harmonic_signal(3)
    for rad in (2 * np.pi / 3, 4 * np.pi / 3):
        want = (np.exp(1j * rad) * sps.hilbert(x)).real
        assert np.allclose(F.hilbert_rotate(x, rad), want, rtol=0, atol=1e-12)


def test_circular_statistics_match_scipy():
    theta = RNG.uniform(-np.pi, np.pi, size=(50, F.N_CYCLES))
    theta[0] = np.pi - 1e-3 + RNG.normal(0, 1e-3, F.N_CYCLES)  # straddles the branch cut
    assert np.allclose(F.circmean(theta), stats.circmean(theta, high=np.pi, low=-np.pi, axis=1))
    assert np.allclose(F.circstd(theta), stats.circstd(theta, high=np.pi, low=-np.pi, axis=1))


@pytest.mark.parametrize("v", [
    RNG.standard_normal(F.N_CYCLES),
    RNG.exponential(size=F.N_CYCLES),
    np.full(F.N_CYCLES, 2.5),                      # constant: undefined -> 0
    1e4 + 1e-9 * RNG.standard_normal(F.N_CYCLES),  # variance at the mean's resolution
])
def test_kurtosis_matches_thesis(v):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        want = np.nan_to_num(stats.kurtosis(v, bias=False), nan=0.0)
    assert np.allclose(F.kurtosis(v), want, rtol=1e-10, atol=1e-12)


def test_negative_sequence_is_zero_for_a_balanced_positive_sequence():
    t = np.arange(F.N) / (F.F0 * F.SPC)
    phases = [np.cos(2 * np.pi * 60 * t - k * 2 * np.pi / 3) for k in range(3)]
    x = np.zeros((2, 7, F.N))
    x[:, F.VA:F.VC + 1] = phases
    x[:, F.IA:F.IC + 1] = phases
    for transform in ("dft", "stockwell", "wavelet"):
        order = "d6" if transform == "wavelet" else "h1"
        fid = f"{transform}.negative_sequence_voltage.{order}.module.mean"
        assert abs(F.extract(x, 0, [fid])[0]) < 1e-6, fid


# --------------------------------------------------------------------------- python/fast.py
# The algorithms the firmware uses must equal the definitions above (to float64 rounding).

import fast  # noqa: E402


def test_dft2560_from_decimated_ffts():
    x = harmonic_signal(4)
    assert np.allclose(fast.dft2560(x, F.N // 2), np.fft.fft(x)[: F.N // 2 + 1], rtol=0, atol=1e-9)


def test_irfft2560_inverts_the_dft():
    x = harmonic_signal(5)
    assert np.allclose(fast.irfft2560(np.fft.fft(x)[: F.N // 2 + 1]), x, rtol=0, atol=1e-12)


def test_fast_hilbert_rotation_matches_definition():
    x = harmonic_signal(6)
    assert np.allclose(fast.hilbert_rotate(x, 2.0), F.hilbert_rotate(x, 2.0), rtol=0, atol=1e-12)


def test_fast_stockwell_matches_definition():
    """Truncating the voice sum at |m| <= 1.5 k0 drops a Gaussian tail below e^-44."""
    for seed in (7, 8):
        x = harmonic_signal(seed)
        want = F.stockwell(x)
        assert np.allclose(fast.stockwell(x), want, rtol=1e-11, atol=1e-14 * np.abs(want).max())
