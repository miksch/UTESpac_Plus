"""Ramp / microfront detection.

Four detectors: the wavelet zero-crossing method (Collineau & Brunet
1993), the structure-function surface-renewal method (Van Atta 1977 cubic,
Spano et al. 1997 lags and flux, Paw U et al. 2005 two-lag d/s split), the
TKE trigger (Mangan et al. 2022), and the Thomas & Foken (2005) detection
with the Zhang et al. (2011) conditional averages and ejection/sweep flux
split (:func:`detect_tf2005`, called by :func:`run` when ``ramps.tf2005``
is on). Science and
sources: library/writeups/ec_ramps.md.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy import signal as sig

from . import preprocess as pp
from .io import HFFile, iter_windows, token

GROUP = "ramps"
D_G = {"mhat": np.pi / np.sqrt(2.0)}       # Collineau & Brunet 1993 I, Table I p. 359


def mhat(x: np.ndarray) -> np.ndarray:
    """MHAT basic wavelet (1 - x^2) exp(-x^2/2)  [CITED] Collineau & Brunet 1993 I, Table I."""
    x = np.asarray(x, dtype=float)
    return (1.0 - x * x) * np.exp(-0.5 * x * x)


def ramp_wavelet(x: np.ndarray) -> np.ndarray:
    """RAMP basic wavelet: 2x + 1 on (-0.5, 0], 2x - 1 on (0, 0.5]  [CITED] CB 1993 I, Table I."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m1 = (x > -0.5) & (x <= 0.0)
    m2 = (x > 0.0) & (x <= 0.5)
    out[m1] = 2.0 * x[m1] + 1.0
    out[m2] = 2.0 * x[m2] - 1.0
    return out


def haar(x: np.ndarray) -> np.ndarray:
    """HAAR basic wavelet: 1 on (-0.5, 0], -1 on (0, 0.5]  [CITED] CB 1993 I, Table I."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    out[(x > -0.5) & (x <= 0.0)] = 1.0
    out[(x > 0.0) & (x <= 0.5)] = -1.0
    return out


_WAVELETS = {"mhat": (mhat, 4.0), "ramp": (ramp_wavelet, 0.5), "haar": (haar, 0.5)}   # (g, half-support in a)


def cwt(h: np.ndarray, scales_s: np.ndarray, fs: float, wavelet: str = "mhat") -> np.ndarray:
    """T_1(a, b) = (1/a) int h(t) g((t - b)/a) dt  [CITED] CB 1993 I eq. 4 (p = 1), a in seconds.

    Returns an array ``(len(scales_s), len(h))``; the convolution is zero-padded at the ends.
    """
    if wavelet not in _WAVELETS:
        raise ValueError(f"wavelet {wavelet!r}; expected one of {sorted(_WAVELETS)}")
    gfun, support = _WAVELETS[wavelet]
    h = np.asarray(h, dtype=float)
    dt = 1.0 / fs
    out = np.empty((len(scales_s), h.size))
    for i, a in enumerate(scales_s):
        half = int(np.ceil(support * a * fs))
        t = np.arange(-half, half + 1) * dt
        g = gfun(t / a)[::-1]                            # correlation with g((t - b)/a)
        out[i] = sig.fftconvolve(h, g, mode="same") * dt / a
    return out


def wavelet_variance(T: np.ndarray, fs: float) -> np.ndarray:
    """W_1(a) = int |T_1(a, b)|^2 db  [CITED] CB 1993 I eq. 8, p. 364 (discrete sum times dt)."""
    return np.sum(T * T, axis=1) / fs


def log_scales(a_min_s: float, a_max_s: float, n_per_decade: int) -> np.ndarray:
    n = max(int(np.ceil(np.log10(a_max_s / a_min_s) * n_per_decade)), 1)
    return np.logspace(np.log10(a_min_s), np.log10(a_max_s), n + 1)


def duration_scale(W: np.ndarray, scales_s: np.ndarray, wavelet: str = "mhat",
                   peak: str = "smallest_scale", D_min_s: float = 0.0) -> Tuple[float, float, int]:
    """(a0, D, index): a0 the variance-peak scale, D = a0 D_g  [CITED] CB 1993 I eq. 22, p. 369.

    The search runs over scales with D >= *D_min_s* (Thomas & Foken 2007 §3.1: fluctuations
    with event durations below 6.2 s removed first). ``peak="smallest_scale"`` takes the
    smallest-scale interior local maximum (Thomas & Foken), ``"global"`` the interior global
    maximum (Collineau & Brunet). A maximum on the edge of the searched grid is not a peak:
    NaNs and index -1 are returned.
    """
    W = np.asarray(W, dtype=float)
    lo = int(np.searchsorted(scales_s * D_G[wavelet], D_min_s))
    Ws = W[lo:]
    if Ws.size < 3 or not np.isfinite(Ws).any():
        return np.nan, np.nan, -1
    interior = np.flatnonzero((Ws[1:-1] > Ws[:-2]) & (Ws[1:-1] >= Ws[2:])) + 1
    if interior.size == 0:
        return np.nan, np.nan, -1
    if peak == "global":
        j = int(interior[np.nanargmax(Ws[interior])])
    elif peak == "smallest_scale":
        j = int(interior[0])
    else:
        raise ValueError(f"peak {peak!r}; expected smallest_scale or global")
    i = lo + j
    a0 = float(scales_s[i])
    return a0, a0 * D_G[wavelet], i


def zero_crossings(T: np.ndarray, fs: float, slope: str = "negative") -> np.ndarray:
    """Times (s) of the zero-crossings of the detection function with the given slope sign
    [CITED] CB 1993 I §4.3.3 p. 375; linear interpolation between samples."""
    T = np.asarray(T, dtype=float)
    s = np.sign(T)
    if slope == "negative":
        idx = np.flatnonzero((s[:-1] > 0) & (s[1:] < 0))
    elif slope == "positive":
        idx = np.flatnonzero((s[:-1] < 0) & (s[1:] > 0))
    elif slope == "both":
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    else:
        raise ValueError(f"slope {slope!r}; expected negative, positive or both")
    frac = T[idx] / (T[idx] - T[idx + 1])
    return (idx + frac) / fs


def refine_times(x: np.ndarray, fs: float, times: np.ndarray, a0: float, wavelet: str = "ramp") -> np.ndarray:
    """Move each detection to the extremum of |T_1(a0, b)| of a first derivative-like wavelet
    (RAMP or HAAR, CB 1993 I section 4.3.2) within +-a0 of the zero-crossing. Deviation, see
    ec_ramps.md: the paper uses those wavelets with a threshold, not as a re-timing step."""
    if times.size == 0:
        return times
    T = np.abs(cwt(x, np.array([a0]), fs, wavelet)[0])
    w = int(round(a0 * fs))
    out = np.empty_like(times)
    for k, ti in enumerate(times):
        i0 = int(round(ti * fs))
        lo, hi = max(i0 - w, 0), min(i0 + w + 1, x.size)
        out[k] = (lo + int(np.argmax(T[lo:hi]))) / fs
    return out


@dataclass
class RampEvents:
    """Wavelet-detected microfronts of one series."""
    a0: float
    D: float
    times: np.ndarray          # s from window start
    slope: str
    W: np.ndarray
    scales_s: np.ndarray
    refine: str = "none"

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def mean_spacing(self) -> float:
        return float(np.mean(np.diff(self.times))) if self.times.size > 1 else np.nan


def detect(x: np.ndarray, fs: float, scales_s: np.ndarray, slope: str = "negative",
           wavelet: str = "mhat", peak: str = "smallest_scale", edge_scales: float = 3.0,
           refine: str = "none", D_min_s: float = 0.0) -> RampEvents:
    """Zero-crossing detection of *x* at the wavelet-variance peak scale (CB 1993 I section 4.3);
    ``refine`` in {"none", "ramp", "haar"} re-times each event (see :func:`refine_times`)."""
    if wavelet != "mhat":
        raise ValueError("the zero-crossing method needs the MHAT wavelet (CB 1993 I section 4.3.3)")
    T = cwt(x, scales_s, fs, wavelet)
    W = wavelet_variance(T, fs)
    a0, D, i = duration_scale(W, scales_s, wavelet, peak, D_min_s)
    if i < 0:
        return RampEvents(np.nan, np.nan, np.array([]), slope, W, scales_s, refine)
    times = zero_crossings(T[i], fs, slope)
    lo, hi = edge_scales * a0, x.size / fs - edge_scales * a0
    times = times[(times >= lo) & (times <= hi)]
    if refine != "none":
        times = np.unique(refine_times(x, fs, times, a0, refine))
    return RampEvents(a0, D, times, slope, W, scales_s, refine)


# ---------------------------------------------------------------- structure functions

def structure_function(x: np.ndarray, j: int, orders: Sequence[int] = (2, 3, 5)) -> Dict[int, float]:
    """S^n(r) at sample lag *j*: mean of (x_i - x_{i-j})^n  [CITED] Spano 1997 eq. 3."""
    if j < 1 or j >= x.size:
        raise ValueError(f"sample lag {j} outside 1..{x.size - 1}")
    d = x[j:] - x[:-j]
    return {n: float(np.mean(d ** n)) for n in orders}


def _amplitude_root(p: float, q: float, S3: float) -> float:
    """Real root of a^3 + p a + q = 0 with sign(a) = -sign(S3) (unstable ramps: S3 < 0, a > 0).

    Van Atta found "only one positive real root" (p. 168); if the sign rule leaves several
    real roots, the largest-magnitude one is taken [ASSUMED], none gives NaN.
    """
    r = np.roots([1.0, 0.0, p, q])
    real = r[np.abs(r.imag) <= 1e-9 * np.abs(r).max()].real
    cand = real[real * S3 < 0]
    return float(cand[np.argmax(np.abs(cand))]) if cand.size else np.nan


@dataclass
class SRRamp:
    """Structure-function ramp parameters of one series at one lag."""
    a: float          # linearized Van Atta amplitude (sign carries the flux direction)
    period: float     # l + s, linearized (s)
    d: float          # two-lag ramp duration (s)
    s: float          # two-lag quiet gap (s)
    a2: float         # two-lag P-corrected amplitude
    S3_rate: float    # S^3(r)/r, the Chen et al. (1997) t_m diagnostic


def vanatta(x: np.ndarray, fs: float, lag_s: float, min_period_lags: float = 10.0) -> SRRamp:
    """Ramp amplitude and dimensions from structure functions at time lag *lag_s*.

    Linearized path [CITED] Van Atta 1977 eqs. 2.13, 2.15 / Spano 1997 eqs. 4-7:
    a^3 + p a + q = 0 with p = 10 S^2 - S^5/S^3, q = 10 S^3; l+s = -a^3 r / S^3.
    Two-lag path [CITED] Paw U et al. 2005 eqs. 3-5 (lag pair r, 2r): the cubic in
    v = r/d, the P-corrected amplitude cubic, and s = -a^3 r P_3 / S^3 - d.
    Lags with l+s < *min_period_lags* * r are rejected (NaN) [CITED] Spano 1997 p. 261;
    so is l+s longer than the window itself (no full ramp period sampled) [ASSUMED].
    """
    r = lag_s
    j = int(round(r * fs))
    S = structure_function(x, j)
    out = SRRamp(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    if not np.isfinite(list(S.values())).all() or S[3] == 0.0:
        return out
    out.S3_rate = S[3] / r
    max_period = x.size / fs
    p = 10.0 * S[2] - S[5] / S[3]
    q = 10.0 * S[3]
    a = _amplitude_root(p, q, S[3])
    if np.isfinite(a):
        period = -(a ** 3) * r / S[3]
        if min_period_lags * r <= period <= max_period:
            out.a, out.period = a, period
    # two-lag split of d and s (Paw U et al. 2005 eq. 3a with b = 2)
    if 2 * j >= x.size:
        return out
    S3b = structure_function(x, 2 * j, orders=(3,))[3]
    if not np.isfinite(S3b) or S3b == 0.0:
        return out
    R = S3b / S[3]
    if R == 16.0:
        return out
    roots = np.roots([1.0, 0.0, -(12.0 - 3.0 * R) / (16.0 - R), (4.0 - 2.0 * R) / (16.0 - R)])
    v = roots[(np.abs(roots.imag) <= 1e-9) & (roots.real > 0.0) & (roots.real < 1.0)].real
    if v.size == 0:
        return out
    v = float(np.min(v))                      # smallest v = largest d if several [ASSUMED]
    P2 = 1.0 - v ** 2 / 3.0
    P3 = 1.0 - 1.5 * v + 0.5 * v ** 3
    P5 = 1.0 - 2.5 * v + (10.0 / 3.0) * v ** 2 - 2.5 * v ** 3 + (2.0 / 3.0) * v ** 5
    if P5 == 0.0 or P3 <= 0.0:
        return out
    a2 = _amplitude_root(p * P3 / P5, q * P2 / P5, S[3])
    if not np.isfinite(a2):
        return out
    d = r / v
    s = -(a2 ** 3) * r * P3 / S[3] - d
    if min_period_lags * r <= d + s <= max_period and s >= 0.0:
        out.d, out.s, out.a2 = d, s, a2
    return out


def sr_flux(a: float, period: float, z: float, alpha: float = 1.0) -> float:
    """Kinematic surface-renewal flux alpha * a * z / (l+s)  [K m s^-1]
    [CITED] Spano 1997 eq. 2 (H = rho c_p a z / (l+s) well above the canopy)."""
    return alpha * a * z / period


def phi_h(zeta):
    """Stability function for heat, Hogstrom (1988): 0.95 + 7.8 zeta (0 <= zeta <= 1),
    0.95 (1 - 11.6 zeta)^-1/2 (-2 <= zeta <= 0); NaN outside the stated ranges.

    [CITED] Castellvi & Snyder 2009 eq. 5 (their "116" is a misprint for 11.6,
    confirmed against Foken 2008; ec_ramps.md)."""
    zeta = np.asarray(zeta, dtype=float)
    out = np.full(zeta.shape, np.nan)
    st = (zeta >= 0.0) & (zeta <= 1.0)
    un = (zeta >= -2.0) & (zeta < 0.0)
    out[st] = 0.95 + 7.8 * zeta[st]
    out[un] = 0.95 * (1.0 - 11.6 * zeta[un]) ** -0.5
    return out


def alpha_castellvi(period, ustar, zeta, z: float, d: float = 0.0, k: float = 0.4):
    """Similarity-based SR weighting factor, inertial-sublayer branch (z > z*):
    alpha = [ (k/pi) (z - d)/z^2 * tau * u_* / phi_h(zeta) ]^1/2

    [CITED] Castellvi & Snyder 2009 eq. 3 (from Castellvi 2004); tau is the SR ramp
    period l+s, zeta = (z - d)/L. The z <= z* branch is not implemented."""
    period, ustar = np.asarray(period, dtype=float), np.asarray(ustar, dtype=float)
    return np.sqrt(k / np.pi * (z - d) / z ** 2 * period * ustar / phi_h(zeta))


# ---------------------------------------------------------------- TKE trigger (Mangan et al. 2022)

def utke(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """u_TKE = sqrt(u'^2 + v'^2 + w'^2) = sqrt(2e)  [CITED] Mangan 2022 §4 p. 52."""
    return np.sqrt(u * u + v * v + w * w)


def utke_lp(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Centred moving mean of u_TKE over *window_s*  [CITED] Mangan 2022 eq. 5 (10 s;
    the printed integrand inconsistency is recorded in ec_ramps.md)."""
    n = max(int(round(window_s * fs)) | 1, 1)          # odd length keeps the filter centred
    return sig.fftconvolve(x, np.full(n, 1.0 / n), mode="same")


@dataclass
class TKEEvents:
    """TKE-triggered coherent-structure events of one window."""
    times: np.ndarray          # trigger times, s from window start (wave minima)
    A_mean: float              # record-mean min-to-max wave amplitude
    sweep_frac: np.ndarray     # per-event bulk-sweep time fraction (Z < 0), IQA

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def mean_spacing(self) -> float:
        return float(np.mean(np.diff(self.times))) if self.times.size > 1 else np.nan


def detect_tke(u: np.ndarray, v: np.ndarray, w: np.ndarray, fs: float, a_s: float = 10.0,
               lp_s: float = 10.0, thresh: float = 1.25, edge_scales: float = 3.0) -> TKEEvents:
    """Trigger coherent structures on the MHAT transform of low-passed u_TKE.

    [CITED] Mangan 2022 §4: MHAT at one fixed scale *a_s*; a structure is identified
    "starting at the minimum of the wavelet coefficient's wave" when the wave's
    min-to-max amplitude exceeds *thresh* times the mean amplitude. Their multi-height
    amplitude calibration collapses to the per-record mean here (one sonic; deviation
    register). Triggers within edge_scales * a_s of the window ends are dropped
    (zero-padded convolution [ASSUMED], as for the wavelet detector).
    """
    x = utke_lp(utke(u, v, w), fs, lp_s)
    C = cwt(x - x.mean(), np.array([a_s]), fs, "mhat")[0]
    mins = np.flatnonzero((C[1:-1] < C[:-2]) & (C[1:-1] <= C[2:])) + 1
    maxs = np.flatnonzero((C[1:-1] > C[:-2]) & (C[1:-1] >= C[2:])) + 1
    if mins.size == 0 or maxs.size == 0:
        return TKEEvents(np.array([]), np.nan, np.array([]))
    nxt = np.searchsorted(maxs, mins)                  # following maximum closes each wave
    keep = nxt < maxs.size
    mins, nxt = mins[keep], nxt[keep]
    amp = C[maxs[nxt]] - C[mins]
    A_mean = float(np.mean(amp))
    trig = mins[amp > thresh * A_mean]
    lo, hi = edge_scales * a_s * fs, C.size - edge_scales * a_s * fs
    trig = trig[(trig >= lo) & (trig <= hi)]
    times = trig / fs
    frac = np.array([np.mean(iqa(u[i0:i1], v[i0:i1], w[i0:i1], fs)[2] < 0.0)
                     for i0, i1 in zip(trig, np.append(trig[1:], C.size))])
    return TKEEvents(times, A_mean, frac)


def iqa(u: np.ndarray, v: np.ndarray, w: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrated quadrant analysis positions X, Y, Z = cumsum(u', v', w') * dt
    [CITED] Mangan 2022 eqs. 2-4; bulk sweep Z < 0, bulk ejection Z > 0 (p. 50)."""
    dt = 1.0 / fs
    return np.cumsum(u) * dt, np.cumsum(v) * dt, np.cumsum(w) * dt


# ---------------------------------------------------------------- Thomas & Foken 2005 (Zhang et al. 2011)
#
# Loci are PDF pages of the journal articles: TF05 = Thomas & Foken 2005, Theor. Appl.
# Climatol. 80, 91-104; Z11 = Zhang et al. 2011, Theor. Appl. Climatol. 103, 119-131.
# The Z11 duration grid and bin count were set for EBEX, a flood-irrigated
# cotton field, with CSAT3 sonics sampled at 10 Hz at 8.7 and 2.7 m
# (Z11 pp. 120-121); they are carried here unchanged for another site.

TF_FS_TARGET = 2.0                  # [CITED] TF05 p. 96; Z11 p. 122 step (1), block-averaged rate
TF_D_MIN_S = 8.0                    # [CITED] Z11 p. 122 step (4), shortest analysed duration
TF_D_MAX_S = 240.0                  # [CITED] Z11 p. 122 step (4), longest analysed duration
TF_D_STEP_S = 1.0                   # [CITED] Z11 p. 122 step (4); TF05 used 8-140 s in 0.5 s
TF_MORLET_OMEGA0 = 5.0              # [CITED] TF05 p. 97
TF_MORLET_SUPPORT = 4.0             # [CITED] TF05 p. 97, supp_min = -4, symmetric
TF_MEXH_OMEGA0 = np.sqrt(2.0)       # [CITED] TF05 p. 97
TF_BIOR_WAVELET = "bior5.5"         # [CITED] TF05 p. 94
TF_BIOR_LEVEL_MIN = 4               # [CITED] TF05 p. 97, levels i >= 4 retained (removes D < 6.2 s)
TF_BIOR_OMEGA0 = 4.0                # [CITED] TF05 p. 97, BIOR5.5 peak frequency ~ 4
TF_BIOR_MODE = "zero"               # [ASSUMED] DWT boundary mode; the record ends are already zero
TF_PEAK_PROMINENCE = 0.10           # [ASSUMED] relative prominence of a clearly expressed maximum;
                                    # weaker maxima of W are not event durations
TF_N_BINS = 24                      # [CITED] Z11 Fig. 3 caption
TF_SUBSAMPLES_PER_BIN = 10          # [ASSUMED] lag points per composite bin


def tf_scale_from_duration(duration_s, omega0: float):
    """Scale in seconds for an event duration, the inverse of TF05 eq. 6, D = a dt pi / omega_0."""
    return np.asarray(duration_s, dtype=float) * omega0 / np.pi


def tf_critical_duration(fs: float = TF_FS_TARGET, level: int = TF_BIOR_LEVEL_MIN,
                         omega0: float = TF_BIOR_OMEGA0) -> float:
    """Critical duration of the BIOR5.5 low-pass, TF05 eq. 6 at a = 2^level samples.

    At 2 Hz and level 4 this is 16 x 0.5 x pi / 4 = 6.28 s [DERIVED]; TF05
    p. 97 states D_c ~ 6.2 s.
    """
    return (2 ** level) / fs * np.pi / omega0


def tf_duration_grid(d_min: float = TF_D_MIN_S, d_max: float = TF_D_MAX_S,
                     step: float = TF_D_STEP_S) -> np.ndarray:
    """Event durations D in seconds on the Z11 grid (8 to 240 s in 1 s steps)."""
    return np.arange(d_min, d_max + 0.5 * step, step)


def block_average(x: np.ndarray, fs: float, fs_target: float = TF_FS_TARGET) -> np.ndarray:
    """Non-overlapping block means from *fs* to *fs_target* along axis 0.

    TF05 p. 96: "the sampling frequency is reduced from the original 20 to
    2 Hz resolution by applying standard block averaging" [CITED].

    Parameters
    ----------
    x : ndarray
        Record, shape ``(n, ...)``.
    fs, fs_target : float
        Input and output rates in hertz; their ratio must be a positive integer.

    Returns
    -------
    ndarray
        Shape ``(n // factor, ...)``; a trailing partial block is dropped.
    """
    factor = int(round(fs / fs_target))
    if factor < 1 or not np.isclose(fs / fs_target, factor):
        raise ValueError(f"fs / fs_target must be a positive integer, got {fs / fs_target}")
    x = np.asarray(x, dtype=float)
    m = x.shape[0] // factor
    return x[:m * factor].reshape((m, factor) + x.shape[1:]).mean(axis=1)


def tf_normalise(x: np.ndarray, remove_mean: bool = True) -> Optional[np.ndarray]:
    """Record divided by its standard deviation, mean removed unless *remove_mean* is False.

    TF05 p. 97 normalises w by its standard deviation only [CITED]; Z11
    p. 122 step (2) omits that exception. None for a record with a
    non-finite sample or zero variance.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        return None
    sd = x.std()
    if not np.isfinite(sd) or sd == 0.0:
        return None
    return ((x - x.mean()) if remove_mean else x) / sd


def tf_pad_width(fs: float = TF_FS_TARGET, d_max: float = TF_D_MAX_S,
                 omega0: float = TF_MORLET_OMEGA0, support: float = TF_MORLET_SUPPORT) -> int:
    """Samples of zero padding per end, |a_n supp_min| at the largest Morlet scale (TF05 p. 97)."""
    return int(np.ceil(support * tf_scale_from_duration(d_max, omega0) * fs))


def bior_lowpass(x: np.ndarray, level_min: int = TF_BIOR_LEVEL_MIN,
                 wavelet: str = TF_BIOR_WAVELET, mode: str = TF_BIOR_MODE) -> np.ndarray:
    """BIOR5.5 wavelet low-pass keeping the decomposition levels ``>= level_min`` (TF05 p. 97).

    The record is decomposed to its maximum level, the detail coefficients of
    levels 1 to ``level_min - 1`` are set to zero, and the record is
    reconstructed and trimmed to its input length. TF05 run the filter on the
    zero-padded record (p. 98) [CITED]; the pywt ``zero`` boundary mode is
    [ASSUMED].

    Parameters
    ----------
    x : ndarray
        One-dimensional record.
    level_min : int, optional
        Smallest retained decomposition level.
    wavelet, mode : str, optional
        pywt wavelet name and boundary mode.

    Returns
    -------
    ndarray
        Filtered record, same length as *x*.
    """
    import pywt

    x = np.asarray(x, dtype=float)
    w = pywt.Wavelet(wavelet)
    level = max(pywt.dwt_max_level(x.size, w.dec_len), level_min - 1)
    coeffs = pywt.wavedec(x, w, mode=mode, level=level)
    for i in range(1, level_min):          # coeffs = [cA_L, cD_L, ..., cD_1]; cD_i at L + 1 - i
        coeffs[level + 1 - i] = np.zeros_like(coeffs[level + 1 - i])
    return pywt.waverec(coeffs, w, mode=mode)[:x.size]


def morlet_hat(omega: np.ndarray, omega0: float = TF_MORLET_OMEGA0) -> np.ndarray:
    """Fourier transform of the complex Morlet, pi^(-1/4) sqrt(2 pi) exp(-(omega - omega0)^2 / 2).

    The transform of ``psi(t) = pi^(-1/4) exp(i omega0 t) exp(-t^2 / 2)`` with
    ``Psi(omega) = integral psi(t) exp(-i omega t) dt``. The admissibility
    correction is omitted [ASSUMED]: it is exp(-omega0^2 / 2) ~ 4e-6 of the
    peak at omega0 = 5.
    """
    omega = np.asarray(omega, dtype=float)
    return np.pi ** -0.25 * np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (omega - omega0) ** 2)


def mhat_hat(omega: np.ndarray) -> np.ndarray:
    """Fourier transform of :func:`mhat`, sqrt(2 pi) omega^2 exp(-omega^2 / 2)."""
    omega = np.asarray(omega, dtype=float)
    return np.sqrt(2.0 * np.pi) * omega * omega * np.exp(-0.5 * omega * omega)


def _fft_cwt_rows(x: np.ndarray, fs: float, scales_s: Sequence[float], hat, complex_out: bool):
    """Yield T_1(a, b) = a^-1 int x(t) psi*((t - b)/a) dt row by row, a linear correlation by FFT."""
    x = np.asarray(x, dtype=float)
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.fft(x, nfft)
    omega = 2.0 * np.pi * np.fft.fftfreq(nfft, d=1.0 / fs)
    for a in np.asarray(scales_s, dtype=float):
        row = np.fft.ifft(spectrum * np.conj(hat(a * omega)))[:n]
        yield row if complex_out else row.real


def tf_wavelet_variance(x: np.ndarray, fs: float, durations: np.ndarray, pad: int = 0,
                        omega0: float = TF_MORLET_OMEGA0,
                        support: float = TF_MORLET_SUPPORT) -> np.ndarray:
    """Morlet wavelet variance W(a) = integral |T_1(a, b)|^2 db (TF05 eq. 5, p = 1).

    Parameters
    ----------
    x : ndarray
        Record, already zero-padded by *pad* samples at each end.
    fs : float
        Sampling rate, hertz.
    durations : ndarray
        Event durations D in seconds; the Morlet scales are ``D omega0 / pi``
        (TF05 eq. 6).
    pad : int, optional
        Zero padding per end included in *x*.
    omega0, support : float, optional
        Morlet peak frequency and effective half-support.

    Returns
    -------
    ndarray
        W at each duration, the sum of |T|^2 dt over the translations
        ``b in [t_min - a supp, t_max + a supp]`` (TF05 p. 97) clipped to the
        padded record.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    scales = tf_scale_from_duration(durations, omega0)
    out = np.empty(scales.size)
    for i, row in enumerate(_fft_cwt_rows(x, fs, scales, morlet_hat, True)):
        k = int(np.ceil(support * scales[i] * fs))
        lo, hi = max(pad - k, 0), min(n - pad + k, n)
        out[i] = np.sum(np.abs(row[lo:hi]) ** 2) / fs
    return out


def tf_event_duration(W: np.ndarray, durations: np.ndarray,
                      prominence: float = TF_PEAK_PROMINENCE) -> Tuple[float, np.ndarray]:
    """Characteristic duration D_e, the first clearly expressed maximum of W (TF05 p. 97).

    TF05 p. 97: "For a robust determination of a_e, a cubic spline
    interpolation is applied to the approximate derivatives between adjacent
    wavelet variances" [CITED]. The derivative is the central difference of W
    on the duration grid; its cubic spline is solved for the roots going from
    positive to negative. A maximum is clearly expressed when
    :func:`scipy.signal.find_peaks` gives it a prominence above
    ``prominence (max W - min W)`` [ASSUMED]; its duration is the spline root
    nearest the grid peak. Maxima on the grid boundary are not maxima.

    Parameters
    ----------
    W : ndarray
        Wavelet variance on *durations*.
    durations : ndarray
        Uniform duration grid, seconds.
    prominence : float, optional
        Relative prominence threshold.

    Returns
    -------
    D_e : float
        The smallest-duration clearly expressed maximum, NaN when none.
    maxima : ndarray
        Every clearly expressed maximum, ascending.
    """
    from scipy import interpolate

    W = np.asarray(W, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if not np.isfinite(W).all() or W.size < 4:
        return np.nan, np.array([])
    span = W.max() - W.min()
    if span <= 0:
        return np.nan, np.array([])
    peaks, _ = sig.find_peaks(W, prominence=prominence * span)
    if peaks.size == 0:
        return np.nan, np.array([])
    spline = interpolate.CubicSpline(durations, np.gradient(W, durations))
    roots = spline.roots(extrapolate=False)
    roots = roots[spline(roots, 1) < 0]
    step = durations[1] - durations[0]
    maxima = []
    for p in peaks:
        near = roots[np.abs(roots - durations[p]) <= step]
        maxima.append(near[np.argmin(np.abs(near - durations[p]))] if near.size else durations[p])
    maxima = np.array(maxima, dtype=float)
    return float(maxima[0]), maxima


def tf_crossing_sign(r: float) -> int:
    """Crossing direction of TF05 Table 2 (p. 98) for (theta_s, w): -1 is + to -, +1 is - to +."""
    return -1 if r > 0 else +1


def tf_crossings(x: np.ndarray, fs: float, sign: int, D_e: float, pad: int = 0,
                 omega0: float = TF_MEXH_OMEGA0) -> np.ndarray:
    """Event instants from the MEXH zero crossings at D_e (TF05 p. 97; Z11 p. 122 step (5)).

    The MEXH scale is ``D_e omega0 / pi`` with omega0 = sqrt(2) (TF05 p. 97),
    the convention D = a0 pi / sqrt(2) of :data:`D_G` [DERIVED].

    Parameters
    ----------
    x : ndarray
        Filtered, normalised record, zero-padded by *pad* samples per end.
    fs : float
        Sampling rate, hertz.
    sign : {-1, +1}
        -1 keeps crossings from + to -, +1 from - to + (:func:`tf_crossing_sign`).
    D_e : float
        Characteristic duration, seconds.
    pad : int, optional
        Padding per end included in *x*.
    omega0 : float, optional
        MEXH peak frequency of TF05 eq. 6.

    Returns
    -------
    ndarray
        Crossing instants in seconds from the first data sample, linearly
        interpolated, restricted to ``[t_min, t_max]`` of the unpadded record
        (TF05 p. 98): a crossing counts only when both bracketing samples lie
        inside it.
    """
    if not np.isfinite(D_e):
        return np.array([])
    x = np.asarray(x, dtype=float)
    c = next(_fft_cwt_rows(x, fs, [float(tf_scale_from_duration(D_e, omega0))], mhat_hat, False))
    keep = np.zeros(x.size, dtype=bool)
    keep[pad:x.size - pad] = True
    t = (np.arange(x.size) - pad) / fs
    c0, c1 = c[:-1], c[1:]
    hit = ((c0 > 0) & (c1 <= 0)) if sign < 0 else ((c0 < 0) & (c1 >= 0))
    hit &= keep[:-1] & keep[1:]
    i = np.flatnonzero(hit)
    return t[i] + c0[i] / (c0[i] - c1[i]) * (t[i + 1] - t[i])


def tf_spacing_stats(times: np.ndarray) -> Tuple[float, float]:
    """Mean T_x and standard deviation sigma_Tx of the event separation (TF05 Table 3).

    The mean runs over the N - 1 intervals and the standard deviation has an
    N - 2 denominator (Thomas & Foken 2007, Boundary-Layer Meteorol. 122,
    p. 129, eqs. 4a-4b); NaN where undefined.
    """
    gaps = np.abs(np.diff(np.asarray(times, dtype=float)))
    mean = float(gaps.mean()) if gaps.size >= 1 else np.nan
    std = float(np.sqrt(np.sum((gaps - mean) ** 2) / (gaps.size - 1))) if gaps.size >= 2 else np.nan
    return mean, std


def tf_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """r_xy = mean(x'y') / (sigma_x sigma_y) (TF05 eq. 9)."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    den = x.std() * y.std()
    return float(np.mean(x * y) / den) if den > 0 else np.nan


def tf_lag_grid(n_bins: int = TF_N_BINS, sub: int = TF_SUBSAMPLES_PER_BIN) -> np.ndarray:
    """Lag points in units of D_e: sub-bin centres over [-1, 1]."""
    m = n_bins * sub
    return -1.0 + (np.arange(m) + 0.5) * 2.0 / m


def tf_event_windows(x: np.ndarray, fs: float, times: np.ndarray, D_e: float,
                     n_bins: int = TF_N_BINS, sub: int = TF_SUBSAMPLES_PER_BIN) -> np.ndarray:
    """Each event's window ``[t_i - D_e, t_i + D_e]`` of *x* on :func:`tf_lag_grid`.

    Linear interpolation between samples. Only events whose whole window lies
    inside the record are kept [ASSUMED].

    Returns
    -------
    ndarray
        Shape ``(n_used, n_bins * sub)``.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size) / fs
    lags = tf_lag_grid(n_bins, sub) * D_e
    used = [ti for ti in np.asarray(times, float) if ti - D_e >= t[0] and ti + D_e <= t[-1]]
    if not used:
        return np.empty((0, lags.size))
    return np.array([np.interp(ti + lags, t, x) for ti in used])


def tf_conditional_average(x: np.ndarray, fs: float, times: np.ndarray, D_e: float,
                           n_bins: int = TF_N_BINS, sub: int = TF_SUBSAMPLES_PER_BIN
                           ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Conditional average <x>(t) = (1/N) sum_n x_n(t), t in [-D_e, D_e] (Z11 eq. 1, p. 122).

    The ejection phase is [-D_e, 0] and the sweep phase [0, D_e] (Z11 p. 123,
    whose "[1, De]" is read as [0, De] [ASSUMED]). The composite is
    block-averaged to *n_bins* bins (Z11 Fig. 3 caption: "Data are normalized
    with their respective standard deviations and then block-averaged to 24
    bins") [CITED].

    Parameters
    ----------
    x : ndarray
        Series on the record time base.
    fs : float
        Sampling rate, hertz.
    times : ndarray
        Event instants, seconds from the record start.
    D_e : float
        Characteristic duration, seconds.
    n_bins, sub : int, optional
        Composite bins and lag points per bin.

    Returns
    -------
    binned : ndarray
        The composite in *n_bins* bins over t / D_e in [-1, 1].
    fine : ndarray
        The composite on :func:`tf_lag_grid`.
    n_used : int
        Events in the average.
    """
    win = tf_event_windows(x, fs, times, D_e, n_bins, sub)
    if win.shape[0] == 0:
        return np.full(n_bins, np.nan), np.full(n_bins * sub, np.nan), 0
    fine = win.mean(axis=0)
    return fine.reshape(n_bins, sub).mean(axis=1), fine, win.shape[0]


def tf_flux_split(x: np.ndarray, y: np.ndarray, fs: float, times: np.ndarray, D_e: float,
                  n_bins: int = TF_N_BINS, sub: int = TF_SUBSAMPLES_PER_BIN) -> Dict[str, float]:
    """Coherent, ejection, sweep and total flux of Z11 eqs. 1-4 (pp. 122-123).

    ``F_cs`` is the window average of ``<x'><y'>``; ``F_ej`` and ``F_sw`` are
    the same operator over [-D_e, 0] and [0, D_e] with the 1 / (2 D_e)
    normaliser of eq. 3, so F_cs = F_ej + F_sw; ``F_tot`` is the window
    average of ``<x'y'>``.

    Returns
    -------
    dict
        ``F_cs``, ``F_ej``, ``F_sw``, ``F_tot``; NaN when no event fits.
    """
    wx = tf_event_windows(x, fs, times, D_e, n_bins, sub)
    wy = tf_event_windows(y, fs, times, D_e, n_bins, sub)
    if wx.shape[0] == 0:
        return dict(F_cs=np.nan, F_ej=np.nan, F_sw=np.nan, F_tot=np.nan)
    prod = wx.mean(axis=0) * wy.mean(axis=0)
    half = prod.size // 2
    return dict(F_cs=float(prod.mean()), F_ej=float(prod[:half].sum() / prod.size),
                F_sw=float(prod[half:].sum() / prod.size),
                F_tot=float((wx * wy).mean(axis=0).mean()))


@dataclass
class TF2005Events:
    """Thomas & Foken (2005) events of one record, with the Zhang et al. (2011) composites."""
    durations: np.ndarray      # duration grid D, s
    W: np.ndarray              # Morlet wavelet variance on the grid
    D_e: float                 # characteristic duration, s (NaN: no clearly expressed maximum)
    maxima: np.ndarray         # every clearly expressed maximum of W, s
    r_Tw: float                # r between the filtered, normalised ts and w (TF05 eq. 9)
    sign: int                  # crossing direction, -1 + to -, +1 - to +, 0 undetermined
    times: np.ndarray          # crossing instants, s from the record start
    used: np.ndarray           # mask of the events whose window fits the record
    T_x: float                 # mean event separation, s (TF05 Table 3)
    sigma_Tx: float            # standard deviation of the separation, s
    f_tc: float                # time cover N D_e / L
    length_s: float            # record length L at the block-averaged rate, s
    comp: Dict[str, np.ndarray]          # series -> binned conditional average
    eject: Dict[str, float]              # series -> mean composite over [-D_e, 0]
    sweep: Dict[str, float]              # series -> mean composite over [0, D_e]
    flux: Dict[str, Dict[str, float]]    # pair -> F_cs, F_ej, F_sw, F_tot

    @property
    def n(self) -> int:
        return int(self.times.size)

    @property
    def mean_spacing(self) -> float:
        return self.T_x


def detect_tf2005(ts: np.ndarray, w: Optional[np.ndarray], fs: float,
                  others: Optional[Dict[str, np.ndarray]] = None,
                  pairs: Optional[Dict[str, Tuple[str, str]]] = None,
                  durations: Optional[np.ndarray] = None, sign: Optional[int] = None,
                  prominence: float = TF_PEAK_PROMINENCE) -> TF2005Events:
    """Thomas & Foken (2005) detection, Zhang et al. (2011) conditional sampling and flux split.

    Detection runs on the sonic temperature only and every series is sampled
    on its event instants (Z11 p. 123: "We sampled all time series data based
    on the moments of occurrence (ti) and De derived from temperature time
    series data") [CITED]. Each series is block-averaged to 2 Hz, normalised
    (mean removed and divided by the standard deviation; w divided only,
    TF05 p. 97), zero-padded by |a_n supp_min| per end and BIOR5.5
    low-passed (levels i >= 4). D_e is the first clearly expressed maximum of
    the complex Morlet (omega0 = 5) wavelet variance over *durations*; the
    events are the MEXH zero crossings at D_e in the direction of TF05
    Table 2 from the sign of r(ts, w). r is computed on the normalised,
    filtered 2 Hz series [ASSUMED which series]. The composited and flux
    series are the normalised, filtered 2 Hz series [ASSUMED; TF05 Fig. 4
    shows filtered series with the detected events], so the flux terms are in
    units of sigma_x sigma_y of the unfiltered 2 Hz series.

    The time cover f_tc = N D_e / L follows Barthlott et al. 2007 eq. 11
    (summed structure durations over the record length) [CITED], each
    structure's duration taken as D_e [DERIVED].

    Parameters
    ----------
    ts : ndarray
        Gap-free sonic temperature at *fs*.
    w : ndarray or None
        Vertical velocity at *fs*; required unless *sign* is given.
    fs : float
        Raw rate, hertz; an integer multiple of :data:`TF_FS_TARGET`.
    others : dict of str to ndarray, optional
        Further series to composite (e.g. ``u_pf``, ``rho_h2o``), keyed by
        series name.
    pairs : dict of str to (str, str), optional
        Flux pairs over the series names ``ts``, ``w_pf`` and the keys of
        *others*; defaults to ``{"wTs": ("w_pf", "ts")}``.
    durations : ndarray, optional
        Duration grid, seconds; defaults to :func:`tf_duration_grid`.
    sign : {-1, +1}, optional
        Crossing direction; by default :func:`tf_crossing_sign` of r(ts, w).
    prominence : float, optional
        Relative prominence threshold of :func:`tf_event_duration`.

    Returns
    -------
    TF2005Events
        Series that cannot be normalised (None, non-finite, zero variance)
        return NaN composites and fluxes.
    """
    durations = tf_duration_grid() if durations is None else np.asarray(durations, float)
    pad = tf_pad_width(TF_FS_TARGET, float(durations.max()))
    raw = {"ts": ts, "w_pf": w, **(others or {})}
    pairs = {"wTs": ("w_pf", "ts")} if pairs is None else dict(pairs)
    filt: Dict[str, Optional[np.ndarray]] = {}
    for name, x in raw.items():
        z = None
        if x is not None:
            x2 = block_average(x, fs, TF_FS_TARGET) if fs != TF_FS_TARGET else np.asarray(x, float)
            z = tf_normalise(x2, remove_mean=(name != "w_pf"))
        filt[name] = None if z is None else bior_lowpass(np.pad(z, pad))
    nan_flux = dict(F_cs=np.nan, F_ej=np.nan, F_sw=np.nan, F_tot=np.nan)
    ev = TF2005Events(durations=durations, W=np.full(durations.size, np.nan), D_e=np.nan,
                      maxima=np.array([]), r_Tw=np.nan, sign=0, times=np.array([]),
                      used=np.array([], dtype=bool), T_x=np.nan, sigma_Tx=np.nan, f_tc=np.nan,
                      length_s=np.nan, comp={k: np.full(TF_N_BINS, np.nan) for k in raw},
                      eject={k: np.nan for k in raw}, sweep={k: np.nan for k in raw},
                      flux={p: dict(nan_flux) for p in pairs})
    if filt["ts"] is None:
        return ev
    trimmed = {k: None if v is None else v[pad:v.size - pad] for k, v in filt.items()}
    n = trimmed["ts"].size
    ev.length_s = n / TF_FS_TARGET
    if trimmed["w_pf"] is not None:
        ev.r_Tw = tf_correlation(trimmed["ts"], trimmed["w_pf"])
    if sign is None:
        if not np.isfinite(ev.r_Tw):
            return ev
        sign = tf_crossing_sign(ev.r_Tw)
    ev.sign = int(sign)
    ev.W = tf_wavelet_variance(filt["ts"], TF_FS_TARGET, durations, pad)
    ev.D_e, ev.maxima = tf_event_duration(ev.W, durations, prominence)
    if not np.isfinite(ev.D_e):
        return ev
    ev.times = tf_crossings(filt["ts"], TF_FS_TARGET, sign, ev.D_e, pad)
    ev.used = (ev.times - ev.D_e >= 0.0) & (ev.times + ev.D_e <= (n - 1) / TF_FS_TARGET)
    ev.T_x, ev.sigma_Tx = tf_spacing_stats(ev.times)
    ev.f_tc = ev.times.size * ev.D_e / ev.length_s
    if not ev.used.any():
        return ev
    t_used = ev.times[ev.used]
    for k, x in trimmed.items():
        if x is None:
            continue
        binned, fine, _ = tf_conditional_average(x, TF_FS_TARGET, t_used, ev.D_e)
        half = fine.size // 2
        ev.comp[k] = binned
        ev.eject[k] = float(fine[:half].mean())
        ev.sweep[k] = float(fine[half:].mean())
    for p, (a, b) in pairs.items():
        if trimmed.get(a) is not None and trimmed.get(b) is not None:
            ev.flux[p] = tf_flux_split(trimmed[a], trimmed[b], TF_FS_TARGET, t_used, ev.D_e)
    return ev


def _ancillary(hf: HFFile, name: str, nr: int, nh: int) -> np.ndarray:
    """Per-record ancillary as (record, height), broadcast when stored per record only."""
    if name not in hf.ds:
        return np.full((nr, nh), np.nan)
    v = hf.ds[name].values.astype(float)
    return np.broadcast_to(v[:, None], (nr, nh)).copy() if v.ndim == 1 else v


def _slope_for(name: str, cfg_slope: str, wT: float) -> str:
    """Slope sign per signal: config value, or 'auto' from the sign of w'T' (ec_ramps.md)."""
    if cfg_slope != "auto":
        return cfg_slope
    if name == "u_pf":
        return "positive"
    if not np.isfinite(wT) or wT == 0:
        return "negative"
    return "negative" if wT > 0 else "positive"


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """``/ramps`` group: ramp events per record and height.

    Wavelet, structure-function and TKE-triggered detectors run on every
    record and height. With ``cfg.ramps.tf2005`` set,
    :func:`detect_tf2005` also runs on each record and height and its
    record scalars are written as the ``tf_`` variables listed in
    ``_TF_VARS``: the event duration ``tf_D_e``, the event count
    ``tf_n_events``, the separation ``tf_mean_spacing`` and
    ``tf_sigma_Tx``, the time cover ``tf_f_tc``, the correlation
    ``tf_r_Tw``, the crossing direction ``tf_sign`` and the Zhang et al.
    (2011) flux split ``tf_F_cs_wTs``, ``tf_F_ej_wTs``, ``tf_F_sw_wTs`` and
    ``tf_F_tot_wTs``. Its input is the gap-filled, non-detrended ``ts`` and
    ``w_pf`` of the record, so the mean of w survives into the TF05 p. 97
    normalisation; a record where either series is rejected by the gap
    screen stays NaN. A rate that is not an integer multiple of
    :data:`TF_FS_TARGET` raises one warning and leaves every ``tf_``
    variable NaN. With the switch off the group carries no ``tf_``
    variable.

    Parameters
    ----------
    hf : HFFile
        High-frequency file.
    cfg : ECConfig
        Configuration; ``cfg.ramps`` and ``cfg.preprocess`` are read.
    records : sequence of int, optional
        Record indices to process; default all.

    Returns
    -------
    xarray.Dataset
        The ``/ramps`` group on (record, height) and its event, scale and
        lag dimensions.
    """
    rc, pc = cfg.ramps, cfg.preprocess
    signals = [s for s in rc.signals if s != "e" and s in hf.ds]
    do_tke = "e" in rc.signals
    sr_signals = [s for s in rc.sr_signals if s in hf.ds]
    lags = np.asarray(rc.sr_lags_s, dtype=float)
    scales = log_scales(rc.a_min_s, rc.a_max_s, rc.n_scales_per_decade)
    nr, nh, ns, ne, nl = len(hf.records), len(hf.heights), len(scales), rc.max_events, len(lags)
    a0 = {s: np.full((nr, nh), np.nan) for s in signals}
    D = {s: np.full((nr, nh), np.nan) for s in signals}
    n_ev = {s: np.zeros((nr, nh), dtype=int) for s in signals}
    spacing = {s: np.full((nr, nh), np.nan) for s in signals}
    times = {s: np.full((nr, nh, ne), np.nan) for s in signals}
    W = {s: np.full((nr, nh, ns), np.nan) for s in signals}
    slope_code = {s: np.full((nr, nh), np.nan) for s in signals}      # -1 negative, +1 positive, 0 both
    sr = {s: {f: np.full((nr, nh, nl), np.nan) for f in ("a", "period", "d", "s", "a2", "S3_rate")}
          for s in sr_signals}
    tke_n = np.zeros((nr, nh), dtype=int)
    tke_spacing = np.full((nr, nh), np.nan)
    tke_times = np.full((nr, nh, ne), np.nan)
    tke_sweep = np.full((nr, nh, ne), np.nan)
    tke_A = np.full((nr, nh), np.nan)
    wT_all = np.full((nr, nh), np.nan)
    do_tf = rc.tf2005
    tf = {k: np.full((nr, nh), np.nan) for k in _TF_VARS} if do_tf else {}
    if do_tf:
        ratio = hf.fs / TF_FS_TARGET
        if not np.isclose(ratio, round(ratio)):
            warnings.warn(
                f"tf2005: sampling rate {hf.fs:g} Hz is not an integer "
                f"multiple of {TF_FS_TARGET:g} Hz; tf_ variables are NaN",
                stacklevel=2)
            do_tf = False
    need = sorted(set(signals) | set(sr_signals) | {"u_pf", "v_pf", "w_pf", "ts"})

    for win in iter_windows(hf, variables=need, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w_pf", ih):
                continue
            prep = pp.prepare(win, ih, need, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            wT = np.nan
            if prep.accepted.get("w_pf") and prep.accepted.get("ts"):
                wT = float(np.nanmean(prep.prime["w_pf"] * prep.prime["ts"]))
            wT_all[i, ih] = wT
            for s in signals:
                if not prep.accepted.get(s):
                    continue
                x = prep.prime[s]
                if not np.isfinite(x).all():
                    continue
                slope = _slope_for(s, getattr(rc, f"slope_{token(s)}", "auto"), wT)
                ev = detect(x, hf.fs, scales, slope=slope, wavelet=rc.wavelet, peak=rc.peak,
                            edge_scales=rc.edge_scales, refine=rc.refine, D_min_s=rc.D_min_s)
                a0[s][i, ih], D[s][i, ih] = ev.a0, ev.D
                n_ev[s][i, ih] = ev.n
                spacing[s][i, ih] = ev.mean_spacing
                k = min(ev.n, ne)
                times[s][i, ih, :k] = ev.times[:k]
                W[s][i, ih] = ev.W
                slope_code[s][i, ih] = {"negative": -1, "positive": 1, "both": 0}[slope]
            for s in sr_signals:
                if not prep.accepted.get(s):
                    continue
                x = prep.prime[s]
                if not np.isfinite(x).all():
                    continue
                for il, r in enumerate(lags):
                    sm = vanatta(x, hf.fs, r, min_period_lags=rc.sr_min_period_lags)
                    for f in ("a", "period", "d", "s", "a2", "S3_rate"):
                        sr[s][f][i, ih, il] = getattr(sm, f)
            if do_tke and all(prep.accepted.get(c) for c in ("u_pf", "v_pf", "w_pf")):
                up, vp, wp = (prep.prime[c] for c in ("u_pf", "v_pf", "w_pf"))
                if np.isfinite(up).all() and np.isfinite(vp).all() and np.isfinite(wp).all():
                    ev = detect_tke(up, vp, wp, hf.fs, a_s=rc.tke_a_s, lp_s=rc.tke_lp_s,
                                    thresh=rc.tke_thresh, edge_scales=rc.edge_scales)
                    tke_n[i, ih] = ev.n
                    tke_spacing[i, ih] = ev.mean_spacing
                    tke_A[i, ih] = ev.A_mean
                    k = min(ev.n, ne)
                    tke_times[i, ih, :k] = ev.times[:k]
                    tke_sweep[i, ih, :k] = ev.sweep_frac[:k]
            if (do_tf and prep.accepted.get("ts")
                    and prep.accepted.get("w_pf")):
                # TF05 p. 97 keeps the mean of w: gap-filled, not detrended
                tr, wr = (pp.fill_gaps(win.series(c, ih), pc.nan_max_frac).x
                          for c in ("ts", "w_pf"))
                _fill_tf2005(tf, i, ih, detect_tf2005(tr, wr, hf.fs))

    ds = xr.Dataset(coords={"record": ("record", hf.records), "height": ("height", hf.heights),
                            "scale": ("scale", scales), "event": ("event", np.arange(ne)),
                            "sr_lag": ("sr_lag", lags)})
    ds["scale"].attrs.update(units="s", long_name="wavelet dilation a (seconds)")
    ds["sr_lag"].attrs.update(units="s", long_name="structure-function time lag r (Spano 1997)")
    for s in signals:
        t = token(s)
        ds[f"a0_{t}"] = (("record", "height"), a0[s], {"units": "s", "long_name": f"wavelet-variance peak scale of {t}'"})
        ds[f"D_{t}"] = (("record", "height"), D[s], {"units": "s", "long_name": f"duration scale D = a0 D_g of {t}' (CB 1993 I eq. 22)"})
        ds[f"n_events_{t}"] = (("record", "height"), n_ev[s], {"long_name": f"MHAT zero-crossings of {t}' at a0 (edges excluded)"})
        ds[f"mean_spacing_{t}"] = (("record", "height"), spacing[s], {"units": "s", "long_name": "mean interval between consecutive detections"})
        ds[f"event_time_{t}"] = (("record", "height", "event"), times[s], {"units": "s", "long_name": "detection time from window start (NaN padded)"})
        ds[f"slope_{t}"] = (("record", "height"), slope_code[s], {"long_name": "slope sign used: -1 negative, +1 positive, 0 both"})
        ds[f"W_{t}"] = (("record", "height", "scale"), W[s], {"long_name": f"wavelet variance W_1(a) of {t}' (CB 1993 I eq. 8)"})
    ds["wT"] = (("record", "height"), wT_all, {"long_name": "window covariance w'Ts' (sign drives slope='auto')"})
    dims3 = ("record", "height", "sr_lag")
    long = {"a": "linearized Van Atta ramp amplitude (sign carries the flux direction)",
            "period": "linearized ramp period l+s (Van Atta eq. 2.15)",
            "d": "two-lag ramp duration (Paw U et al. 2005 eq. 3a)",
            "s": "two-lag quiet gap (Paw U et al. 2005 eq. 5)",
            "a2": "two-lag P-corrected ramp amplitude (Paw U et al. 2005 eq. 4)",
            "S3_rate": "S^3(r)/r, the Chen et al. (1997) t_m diagnostic"}
    for s in sr_signals:
        t = token(s)
        base = "K" if s == "ts" else "m s-1"
        units = {"a": base, "period": "s", "d": "s", "s": "s", "a2": base, "S3_rate": f"({base})^3 s-1"}
        for f in ("a", "period", "d", "s", "a2", "S3_rate"):
            ds[f"sr_{f}_{t}"] = (dims3, sr[s][f], {"units": units[f], "long_name": f"{long[f]} of {t}'"})
    if "ts" in sr_signals:
        # kinematic SR flux F = alpha * a * z / (l+s); alpha per sr_alpha_mode (ruling 2026-08-23)
        z_h = np.asarray(hf.heights, dtype=float)[None, :, None]
        F0 = sr["ts"]["a"] * z_h / sr["ts"]["period"]
        alpha_arr = np.full((nr, nh, nl), np.nan)
        if rc.sr_alpha_mode == "fixed":
            alpha_arr[:] = rc.sr_alpha
        elif rc.sr_alpha_mode == "castellvi":
            ust = _ancillary(hf, "ustar_pf", nr, nh)
            L_ob = _ancillary(hf, "L", nr, nh)
            for ih in range(nh):
                zz = float(hf.heights[ih])
                with np.errstate(divide="ignore", invalid="ignore"):
                    zeta = (zz - rc.sr_d) / L_ob[:, ih]
                for il in range(nl):
                    alpha_arr[:, ih, il] = alpha_castellvi(sr["ts"]["period"][:, ih, il],
                                                           ust[:, ih], zeta, zz, rc.sr_d)
        elif rc.sr_alpha_mode == "fit":
            for il in range(nl):          # [SITE-TUNED] least squares through the origin vs w'Ts'
                x, y = F0[:, :, il].ravel(), wT_all.ravel()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 2 and float(x[m] @ x[m]) > 0.0:
                    alpha_arr[:, :, il] = float(x[m] @ y[m]) / float(x[m] @ x[m])
        else:
            raise ValueError(f"sr_alpha_mode {rc.sr_alpha_mode!r}; expected fixed, castellvi or fit")
        ds["sr_flux_ts"] = (dims3, alpha_arr * F0,
                            {"units": "K m s^-1",
                             "long_name": "kinematic surface-renewal heat flux alpha a z/(l+s) (Spano 1997 eq. 2)"})
        ds["sr_alpha_ts"] = (dims3, alpha_arr,
                             {"long_name": f"weighting factor alpha applied (mode {rc.sr_alpha_mode})"})
    if do_tke:
        ds["n_events_e"] = (("record", "height"), tke_n, {"long_name": "TKE-triggered coherent structures (Mangan 2022)"})
        ds["mean_spacing_e"] = (("record", "height"), tke_spacing, {"units": "s", "long_name": "mean interval between TKE triggers"})
        ds["event_time_e"] = (("record", "height", "event"), tke_times, {"units": "s", "long_name": "TKE trigger time from window start (wave minimum, NaN padded)"})
        ds["sweep_frac_e"] = (("record", "height", "event"), tke_sweep, {"long_name": "bulk-sweep (Z<0) time fraction of each event, IQA"})
        ds["A_mean_e"] = (("record", "height"), tke_A, {"units": "m s^-1", "long_name": "record-mean MHAT wave amplitude of low-passed u_TKE"})
    ds.attrs.update(wavelet=rc.wavelet, D_g=float(D_G[rc.wavelet]), peak=rc.peak, refine=rc.refine,
                    D_min_s=rc.D_min_s,
                    edge_scales=rc.edge_scales, a_min_s=rc.a_min_s, a_max_s=rc.a_max_s,
                    detrend_method=pc.detrend, signals=",".join(token(s) for s in signals),
                    sr_signals=",".join(token(s) for s in sr_signals), sr_alpha=rc.sr_alpha,
                    sr_alpha_mode=rc.sr_alpha_mode, sr_d=rc.sr_d,
                    sr_min_period_lags=rc.sr_min_period_lags,
                    tke_lp_s=rc.tke_lp_s, tke_a_s=rc.tke_a_s, tke_thresh=rc.tke_thresh,
                    library_note="library/writeups/ec_ramps.md")
    if rc.tf2005:
        for k, (units, long_name) in _TF_VARS.items():
            attrs = {"long_name": long_name}
            if units is not None:
                attrs = {"units": units, **attrs}
            ds[k] = (("record", "height"), tf[k], attrs)
        ds.attrs.update(
            tf2005_method=("Thomas and Foken (2005) detection on ts, Zhang "
                           "et al. (2011) conditional sampling and flux "
                           "split of w_pf and ts"),
            tf2005_input=("gap-filled, non-detrended ts and w_pf of the "
                          "record (preprocess.fill_gaps) [ASSUMED]"),
            tf2005_r_Tw=("r on the normalised, filtered 2 Hz series "
                         "[ASSUMED which series]"),
            tf2005_flux=("flux terms of w Ts' (w not mean-removed) on the "
                         "normalised, filtered 2 Hz series [ASSUMED], "
                         "units of sigma_w sigma_Ts"),
            tf2005_f_tc=("N D_e / L, Barthlott et al. 2007 eq. 11 [CITED], "
                         "structure duration D_e [DERIVED]"),
            tf2005_durations_s=(f"{TF_D_MIN_S:g} to {TF_D_MAX_S:g} "
                                f"by {TF_D_STEP_S:g}"),
            tf2005_prominence=TF_PEAK_PROMINENCE)
    return ds


_TF_FLUX = "w Ts' (w not mean-removed), in sigma_w sigma_Ts"
_TF_VARS = {
    "tf_D_e": ("s", "characteristic event duration D_e, first clearly "
                    "expressed Morlet variance maximum of ts (Thomas and "
                    "Foken 2005)"),
    "tf_n_events": (None, "number of coherent structures N, MEXH zero "
                          "crossings at D_e in the Table 2 direction "
                          "(Thomas and Foken 2005)"),
    "tf_mean_spacing": ("s", "mean event separation T_x (Thomas and Foken "
                             "2005 Table 3)"),
    "tf_sigma_Tx": ("s", "standard deviation of the event separation "
                         "(Thomas and Foken 2007, Boundary-Layer Meteorol. "
                         "122, p. 129, eqs. 4a-4b)"),
    "tf_f_tc": (None, "time cover N D_e / L (Barthlott et al. 2007 "
                      "eq. 11)"),
    "tf_r_Tw": (None, "correlation of the filtered, normalised ts and w "
                      "(Thomas and Foken 2005 eq. 9)"),
    "tf_sign": (None, "crossing direction from r: -1 + to -, +1 - to + "
                      "(Thomas and Foken 2005 Table 2)"),
    "tf_F_cs_wTs": ("1", f"coherent flux F_cs = F_ej + F_sw of {_TF_FLUX} "
                         "(Zhang et al. 2011)"),
    "tf_F_ej_wTs": ("1", f"ejection flux F_ej over [-D_e, 0] of {_TF_FLUX} "
                         "(Zhang et al. 2011)"),
    "tf_F_sw_wTs": ("1", f"sweep flux F_sw over [0, D_e] of {_TF_FLUX} "
                         "(Zhang et al. 2011)"),
    "tf_F_tot_wTs": ("1", f"total flux in the event windows of {_TF_FLUX} "
                          "(Zhang et al. 2011)"),
}


def _fill_tf2005(tf: Dict[str, np.ndarray], i: int, ih: int,
                 ev: TF2005Events) -> None:
    """Write the record scalars of *ev* into the ``tf_`` arrays.

    Scalars the detector did not find stay NaN.
    """
    tf["tf_r_Tw"][i, ih] = ev.r_Tw
    if ev.sign != 0:
        tf["tf_sign"][i, ih] = ev.sign
    if not np.isfinite(ev.D_e):
        return
    tf["tf_D_e"][i, ih] = ev.D_e
    tf["tf_n_events"][i, ih] = ev.n
    tf["tf_mean_spacing"][i, ih] = ev.T_x
    tf["tf_sigma_Tx"][i, ih] = ev.sigma_Tx
    tf["tf_f_tc"][i, ih] = ev.f_tc
    flux = ev.flux["wTs"]
    for part in ("F_cs", "F_ej", "F_sw", "F_tot"):
        tf[f"tf_{part}_wTs"][i, ih] = flux[part]
