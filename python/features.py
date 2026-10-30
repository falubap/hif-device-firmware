"""Reference feature extraction: the definition the C firmware is checked against.

Reproduces the thesis's feature contract (tcc/configs/extraction.yaml) for any feature id, from
one raw window, using numpy only. Every transform is written out explicitly rather than calling a
library, so each step here maps onto one C function.

Input window
    ``x[w, c, n]``: w = 0 pre-fault / 1 post-fault, c = raw channel (RAW_CHANNELS order),
    n = 0..2559 (10 cycles x 256 samples, fs = 15360 Hz), already resampled and with noise added.
    ``phase``: faulted phase, 0/1/2 = A/B/C (from scenario metadata, as in the thesis).

Feature id
    ``transform.channel.order.parameter.statistic``, e.g.
    ``stockwell.neutral_current.h3.module.ratio`` -- the thesis's naming.

    extract(x, phase, feature_ids) -> np.ndarray (float64)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

F0 = 60.0
SPC = 256                 # samples per cycle
N_CYCLES = 10
N = SPC * N_CYCLES        # 2560 samples per window
MAX_ORDER = 20            # harmonics 1..20
WAVELET_LEVELS = 8
EPS = 1e-12               # parameter and ratio guard, as in the thesis

RAW_CHANNELS = ("V/FaseA", "V/FaseB", "V/FaseC", "I/FaseA", "I/FaseB", "I/FaseC", "I/Neutro")
VA, VB, VC, IA, IB, IC, IN = range(7)
PRE, POST = 0, 1

CHANNELS = (
    "fault_voltage", "fault_current",
    "negative_sequence_voltage", "negative_sequence_current",
    "neutral_current", "voltage_sum",
)
PAIRS = (
    ("fault_voltage", "fault_current"),
    ("negative_sequence_voltage", "negative_sequence_current"),
    ("voltage_sum", "neutral_current"),
)
CIRCULAR = ("angle", "vi_angle")

#: Fortescue operator; negative sequence = (A + a^2 B + a C) / 3.
A = np.exp(2j * np.pi / 3)

#: Daubechies-4 decomposition low-pass (pywt "db4" dec_lo), 8 taps.
DB4_DEC_LO = np.array([
    -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
    -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
    0.7148465705529157, 0.2303778133088965,
])
#: Quadrature mirror: dec_hi[k] = (-1)^(k+1) dec_lo[L-1-k].
DB4_DEC_HI = np.array([(-1) ** (k + 1) * DB4_DEC_LO[7 - k] for k in range(8)])


# --------------------------------------------------------------------------- transforms
# Each maps one real window (N,) to a coefficient matrix (n_orders, N_CYCLES).


def dft(signal: np.ndarray) -> np.ndarray:
    """Per-cycle DFT, harmonics 1..20, scaled to peak amplitude (2/SPC)."""
    spectrum = np.fft.rfft(signal.reshape(N_CYCLES, SPC), axis=1) * (2.0 / SPC)
    return spectrum[:, 1 : MAX_ORDER + 1].T


@lru_cache(maxsize=None)
def stockwell_gaussian(freq_index: int) -> np.ndarray:
    """Frequency-domain Gaussian of the S-transform voice at ``freq_index``, over m in [-N/2, N/2)."""
    m = np.arange(N)
    m = np.where(m > N // 2, m - N, m).astype(np.float64)
    return np.exp(-2.0 * np.pi**2 * m**2 / float(freq_index) ** 2)


def stockwell(signal: np.ndarray) -> np.ndarray:
    """S-transform voices at the harmonic frequencies, averaged (complex) over each cycle.

    Harmonic h of a 10-cycle window sits at frequency index n = 10h. The voice is
    ifft(H[m + n] * G_n[m]) over the whole window, scaled by 2 to peak amplitude.
    """
    spectrum = np.fft.fft(signal)
    out = np.empty((MAX_ORDER, N_CYCLES), dtype=np.complex128)
    for h in range(1, MAX_ORDER + 1):
        n = h * N_CYCLES
        shifted = np.roll(spectrum, -n)            # shifted[m] = H[(m + n) mod N]
        voice = np.fft.ifft(shifted * stockwell_gaussian(n)) * 2.0
        out[h - 1] = voice.reshape(N_CYCLES, SPC).mean(axis=1)
    return out


def swt_db4(signal: np.ndarray) -> list[np.ndarray]:
    """Stationary (undecimated) wavelet transform, db4, 8 levels, periodic, energy-normalised.

    Equivalent to ``pywt.swt(signal, "db4", level=8, trim_approx=True, norm=True)``, returned as
    [d1, ..., d8, a8] (finest first). Level j filters the previous approximation with the filters
    upsampled by 2^(j-1) ("a trous"); norm=True divides the filters by sqrt(2).
    """
    lo, hi = DB4_DEC_LO / np.sqrt(2.0), DB4_DEC_HI / np.sqrt(2.0)
    approx = signal.astype(np.float64)
    bands = []
    for level in range(WAVELET_LEVELS):
        step = 1 << level
        detail = _swt_filter(approx, hi, step)
        approx = _swt_filter(approx, lo, step)
        bands.append(detail)
    bands.append(approx)
    return bands


def _swt_filter(x: np.ndarray, f: np.ndarray, step: int) -> np.ndarray:
    """Circular convolution with the filter upsampled by ``step``, aligned as pywt.swt does:

        out[i] = sum_k f[k] * x[(i - step * (k - SWT_OFFSET)) mod N]
    """
    out = np.zeros_like(x)
    for k in range(f.size):
        out += f[k] * np.roll(x, step * (k - SWT_OFFSET))
    return out


#: pywt's periodic SWT is a circular convolution advanced by L/2 = 4 taps (found against
#: pywt.swt; pinned by python/test_features.py).
SWT_OFFSET = 4


def wavelet(signal: np.ndarray) -> np.ndarray:
    """Per-cycle RMS of each SWT band: rows d1..d8, a8. Real-valued (no phase)."""
    bands = swt_db4(signal)
    return np.stack([np.sqrt((b.reshape(N_CYCLES, SPC) ** 2).mean(axis=1)) for b in bands])


def hilbert_rotate(signal: np.ndarray, radians: float) -> np.ndarray:
    """Re{exp(j*radians) * analytic(signal)}: a phase rotation at every frequency.

    The analytic signal is ifft(X * u) with u = 1 at DC and Nyquist, 2 for positive frequencies,
    0 for negative ones (scipy.signal.hilbert for even N).
    """
    u = np.zeros(N)
    u[0] = u[N // 2] = 1.0
    u[1 : N // 2] = 2.0
    analytic = np.fft.ifft(np.fft.fft(signal) * u)
    return (np.exp(1j * radians) * analytic).real


# --------------------------------------------------------------------------- channels


def channel_waveform(x: np.ndarray, phase: int, w: int, channel: str) -> np.ndarray:
    """One analysis channel as a time-domain waveform (used by the real-valued wavelet)."""
    v, i = x[w, VA : VC + 1], x[w, IA : IC + 1]
    if channel == "fault_voltage":
        return v[phase]
    if channel == "fault_current":
        return i[phase]
    if channel == "neutral_current":
        return x[w, IN]
    if channel == "voltage_sum":
        return v[0] + v[1] + v[2]
    if channel in ("negative_sequence_voltage", "negative_sequence_current"):
        a, b, c = v if channel.endswith("voltage") else i
        return (a + hilbert_rotate(b, 4 * np.pi / 3) + hilbert_rotate(c, 2 * np.pi / 3)) / 3.0
    raise KeyError(channel)


def channel_phasors(raw: list[np.ndarray], phase: int, channel: str) -> np.ndarray:
    """One analysis channel from per-raw-channel complex coefficients (DFT, Stockwell).

    The transforms are linear, so derived channels combine coefficients; the negative sequence
    needs this route, since Fortescue's operator is complex.
    """
    if channel == "fault_voltage":
        return raw[VA + phase]
    if channel == "fault_current":
        return raw[IA + phase]
    if channel == "neutral_current":
        return raw[IN]
    if channel == "voltage_sum":
        return raw[VA] + raw[VB] + raw[VC]
    if channel == "negative_sequence_voltage":
        return raw[VA] / 3 + A**2 * raw[VB] / 3 + A * raw[VC] / 3
    if channel == "negative_sequence_current":
        return raw[IA] / 3 + A**2 * raw[IB] / 3 + A * raw[IC] / 3
    raise KeyError(channel)


# --------------------------------------------------------------------------- parameters


def parameter(name: str, c: np.ndarray, c2: np.ndarray | None = None) -> np.ndarray:
    """(n_orders, N_CYCLES) -> (n_orders, N_CYCLES), or (1, N_CYCLES) for THD."""
    if name == "module":
        return np.abs(c)
    if name == "energy":
        return np.abs(c) ** 2
    if name == "angle":
        return np.angle(c)
    if name == "thd":
        mag = np.abs(c)
        return (np.sqrt((mag[1:] ** 2).sum(axis=0)) / np.maximum(mag[0], EPS))[np.newaxis]
    if name == "impedance_magnitude":
        return np.abs(c) / np.maximum(np.abs(c2), EPS)
    if name == "vi_angle":
        return np.angle(c * np.conj(c2))
    raise KeyError(name)


# --------------------------------------------------------------------------- statistics


def circmean(theta: np.ndarray) -> np.ndarray:
    """Circular mean along the cycle axis, in [-pi, pi) (scipy.stats.circmean)."""
    mean = np.angle(np.exp(1j * theta).mean(axis=-1))
    return np.where(mean >= np.pi, mean - 2 * np.pi, mean)


def circstd(theta: np.ndarray) -> np.ndarray:
    """Circular standard deviation sqrt(-2 ln R) (scipy.stats.circstd)."""
    r = np.abs(np.exp(1j * theta).mean(axis=-1))
    return np.sqrt(-2.0 * np.log(np.minimum(r, 1.0)))


def kurtosis(v: np.ndarray) -> np.ndarray:
    """Unbiased Fisher kurtosis (scipy.stats.kurtosis(bias=False)); undefined -> 0."""
    n = v.shape[-1]
    mean = v.mean(axis=-1, keepdims=True)
    d = v - mean
    m2 = (d**2).mean(axis=-1)
    m4 = (d**4).mean(axis=-1)
    # scipy treats a variance at the floating-point resolution of the mean as zero.
    zero = m2 <= (np.finfo(np.float64).resolution * mean[..., 0]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = ((n**2 - 1.0) * m4 / m2**2 - 3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return np.where(zero, 0.0, g2)


def entropy(v: np.ndarray) -> np.ndarray:
    """Shannon entropy of the cycle series normalised to sum to one."""
    mag = np.abs(v)
    p = mag / np.maximum(mag.sum(axis=-1, keepdims=True), EPS)
    return -(p * np.log(np.maximum(p, EPS))).sum(axis=-1)


def statistic(name: str, pre: np.ndarray, post: np.ndarray, circular: bool) -> np.ndarray:
    """Collapse the cycle axis. Within-window statistics use the post-fault window."""
    if circular:
        if name == "mean":
            return circmean(post)
        if name == "std":
            return circstd(post)
        if name == "delta":
            return np.angle(np.exp(1j * (circmean(post) - circmean(pre))))
        raise KeyError(f"{name} is not defined for a circular parameter")

    if name == "mean":
        return post.mean(axis=-1)
    if name == "max":
        return post.max(axis=-1)
    if name == "amplitude":
        return post.max(axis=-1) - post.min(axis=-1)
    if name == "std":
        return post.std(axis=-1)
    if name == "kurtosis":
        return kurtosis(post)
    if name == "entropy":
        return entropy(post)
    if name == "delta":
        return post.mean(axis=-1) - pre.mean(axis=-1)
    if name == "ratio":
        p = pre.mean(axis=-1)
        guard = np.where(np.abs(p) < EPS, np.copysign(EPS, p + EPS), p)
        return post.mean(axis=-1) / guard
    raise KeyError(name)


# --------------------------------------------------------------------------- extraction

ORDER_INDEX = {f"h{h}": h - 1 for h in range(1, MAX_ORDER + 1)}
ORDER_INDEX.update({f"d{j}": j - 1 for j in range(1, WAVELET_LEVELS + 1)})
ORDER_INDEX[f"a{WAVELET_LEVELS}"] = WAVELET_LEVELS
ORDER_INDEX["agg"] = 0


def parse(feature_id: str) -> tuple[str, tuple[str, ...], str, str, str]:
    transform, channel, order, param, stat = feature_id.split(".")
    return transform, tuple(channel.split("__")), order, param, stat


class _Coefficients:
    """Lazily computed coefficient matrices for one window, keyed by (transform, channel, w)."""

    def __init__(self, x: np.ndarray, phase: int) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.phase = phase
        self._raw: dict[tuple[str, int], list[np.ndarray]] = {}
        self._cache: dict[tuple[str, str, int], np.ndarray] = {}

    def get(self, transform: str, channel: str, w: int) -> np.ndarray:
        key = (transform, channel, w)
        if key not in self._cache:
            if transform == "wavelet":
                self._cache[key] = wavelet(channel_waveform(self.x, self.phase, w, channel))
            else:
                fn = dft if transform == "dft" else stockwell
                if (transform, w) not in self._raw:
                    self._raw[(transform, w)] = [fn(s) for s in self.x[w]]
                self._cache[key] = channel_phasors(self._raw[(transform, w)], self.phase, channel)
        return self._cache[key]


def extract(x: np.ndarray, phase: int, feature_ids: list[str]) -> np.ndarray:
    """Compute the given features for one window."""
    coeffs = _Coefficients(x, phase)
    out = np.empty(len(feature_ids))
    for k, fid in enumerate(feature_ids):
        transform, channels, order, param, stat = parse(fid)
        mats = [
            parameter(param, *(coeffs.get(transform, ch, w) for ch in channels))
            for w in (PRE, POST)
        ]
        row = ORDER_INDEX[order]
        out[k] = statistic(stat, mats[PRE][row], mats[POST][row], param in CIRCULAR)
    return out
