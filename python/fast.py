"""The algorithms the C firmware uses, in numpy: the reference for each C function.

features.py states the definitions in the most direct form (full 2560-point FFTs). The firmware
cannot do that: CMSIS-DSP's FFTs are powers of two only, and an inverse 2560-point FFT per
Stockwell voice would be wasteful. The rewrites here are exact (up to rounding and a Gaussian
tail below e^-44) and are checked against features.py in python/test_features.py. Each function
maps onto one C function in firmware/src/dsp2560.c or firmware/src/features.c.
"""

from __future__ import annotations

import numpy as np

from features import MAX_ORDER, N, N_CYCLES, SPC

D = N // SPC                 # 10: decimation factor, N = D * SPC
#: Stockwell voices are summed over |m| <= STOCKWELL_SPAN * k0; the Gaussian there is
#: exp(-2 pi^2 * 1.5^2) ~ 7e-20, far below float32 resolution for any dynamic range.
STOCKWELL_SPAN = 1.5


def dft2560(x: np.ndarray, k_max: int) -> np.ndarray:
    """H[k] = sum_n x[n] e^{-j2pi kn/N} for k = 0..k_max, from D real 256-point FFTs.

    With n = D r + s:  H[k] = sum_s e^{-j2pi sk/N} Y_s[k mod 256],  Y_s = FFT256(x[s::D]).
    (C: dsp2560_dft)
    """
    Y = np.fft.fft(x.reshape(SPC, D).T, axis=1)          # (D, 256), row s = FFT of x[s::D]
    k = np.arange(k_max + 1)
    tw = np.exp(-2j * np.pi * np.outer(np.arange(D), k) / N)
    return (tw * Y[:, k % SPC]).sum(axis=0)


def irfft2560(H: np.ndarray) -> np.ndarray:
    """Inverse of a real signal's 2560-point DFT, given bins 0..N/2, from D inverse 256-point FFTs.

    x[D r + s] = IFFT256_r(Z_s)[r] / D,  Z_s[k1] = sum_{k2} H[k1 + 256 k2] e^{j2pi (k1+256 k2) s/N}.
    (C: dsp2560_irfft)
    """
    full = np.concatenate([H, np.conj(H[-2:0:-1])])      # Hermitian extension to N bins
    k1 = np.arange(SPC)
    x = np.empty(N)
    for s in range(D):
        k = k1[np.newaxis, :] + SPC * np.arange(D)[:, np.newaxis]     # (k2, k1)
        Z = (full[k] * np.exp(2j * np.pi * k * s / N)).sum(axis=0)
        x[s::D] = np.fft.ifft(Z).real / D
    return x


def hilbert_rotate(x: np.ndarray, radians: float) -> np.ndarray:
    """cos(t) x - sin(t) H{x}: the Hilbert transform is irfft of -j X on 0 < k < N/2.

    (C: hilbert_rotate in features.c)
    """
    X = dft2560(x, N // 2)
    Y = -1j * X
    Y[0] = Y[N // 2] = 0.0
    return np.cos(radians) * x - np.sin(radians) * irfft2560(Y)


def cycle_mean_kernel(m: np.ndarray) -> np.ndarray:
    """B[m] = (1/256) sum_{u<256} e^{j2pi mu/N}: the cycle average of one inverse-DFT term."""
    out = np.ones(m.shape, dtype=complex)
    nz = (m % N) != 0
    z = np.exp(2j * np.pi * m[nz] / N)
    out[nz] = (1 - z**SPC) / (SPC * (1 - z))
    return out


def stockwell(x: np.ndarray, orders=range(1, MAX_ORDER + 1)) -> np.ndarray:
    """Per-cycle means of the S-transform voices, without any inverse FFT.

    X_{h,c} = (2/N) sum_{|m|<=M} H[k0+m] G[m] B[m] e^{j2pi mc/D},  k0 = 10h, M = 1.5 k0.
    Grouping m by (m mod D) turns the sum over cycles into one D-point DFT. Bins k0+m < 0 use
    H[-k] = conj(H[k]) (x is real). (C: stockwell_voices in features.c)
    """
    orders = list(orders)
    k_max = int(np.ceil((1 + STOCKWELL_SPAN) * N_CYCLES * max(orders)))
    H = dft2560(x, k_max)
    out = np.zeros((MAX_ORDER, N_CYCLES), dtype=complex)
    for h in orders:
        k0 = N_CYCLES * h
        M = int(np.ceil(STOCKWELL_SPAN * k0))
        m = np.arange(-M, M + 1)
        k = k0 + m
        Hk = np.where(k >= 0, H[np.abs(k)], np.conj(H[np.abs(k)]))
        v = Hk * np.exp(-2 * np.pi**2 * m**2 / k0**2) * cycle_mean_kernel(m)
        S = np.zeros(D, dtype=complex)
        np.add.at(S, m % D, v)                                         # S_r = sum_{m=r mod D} v_m
        c = np.arange(N_CYCLES)
        out[h - 1] = (2 / N) * (S[np.newaxis, :] * np.exp(2j * np.pi * np.outer(c, np.arange(D)) / D)).sum(axis=1)
    return out
