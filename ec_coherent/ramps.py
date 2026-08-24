"""Ramp / microfront detection.

Three detectors: the wavelet zero-crossing method (Collineau & Brunet
1993), the structure-function surface-renewal method (Van Atta 1977 cubic,
Spano et al. 1997 lags and flux, Paw U et al. 2005 two-lag d/s split), and
the TKE trigger (Mangan et al. 2022). Science and sources:
library/writeups/ec_ramps.md.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy import signal as sig

from . import preprocess as pp
from .io import HFFile, iter_windows

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
    if name == "u":
        return "positive"
    if not np.isfinite(wT) or wT == 0:
        return "negative"
    return "negative" if wT > 0 else "positive"


def run(hf: HFFile, cfg, records: Optional[Sequence[int]] = None) -> xr.Dataset:
    """``/ramps`` group: wavelet, structure-function and TKE-triggered events per record and height."""
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
    need = sorted(set(signals) | set(sr_signals) | {"u", "v", "w", "Ts"})

    for win in iter_windows(hf, variables=need, records=records):
        i = win.index
        for ih in range(nh):
            if not win.has("w", ih):
                continue
            prep = pp.prepare(win, ih, need, method=pc.detrend, tau_s=pc.filter_tau_s,
                              nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
            wT = np.nan
            if prep.accepted.get("w") and prep.accepted.get("Ts"):
                wT = float(np.nanmean(prep.prime["w"] * prep.prime["Ts"]))
            wT_all[i, ih] = wT
            for s in signals:
                if not prep.accepted.get(s):
                    continue
                x = prep.prime[s]
                if not np.isfinite(x).all():
                    continue
                slope = _slope_for(s, getattr(rc, f"slope_{s}", "auto"), wT)
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
            if do_tke and all(prep.accepted.get(c) for c in ("u", "v", "w")):
                up, vp, wp = (prep.prime[c] for c in ("u", "v", "w"))
                if np.isfinite(up).all() and np.isfinite(vp).all() and np.isfinite(wp).all():
                    ev = detect_tke(up, vp, wp, hf.fs, a_s=rc.tke_a_s, lp_s=rc.tke_lp_s,
                                    thresh=rc.tke_thresh, edge_scales=rc.edge_scales)
                    tke_n[i, ih] = ev.n
                    tke_spacing[i, ih] = ev.mean_spacing
                    tke_A[i, ih] = ev.A_mean
                    k = min(ev.n, ne)
                    tke_times[i, ih, :k] = ev.times[:k]
                    tke_sweep[i, ih, :k] = ev.sweep_frac[:k]

    ds = xr.Dataset(coords={"record": ("record", hf.records), "height": ("height", hf.heights),
                            "scale": ("scale", scales), "event": ("event", np.arange(ne)),
                            "sr_lag": ("sr_lag", lags)})
    ds["scale"].attrs.update(units="s", long_name="wavelet dilation a (seconds)")
    ds["sr_lag"].attrs.update(units="s", long_name="structure-function time lag r (Spano 1997)")
    for s in signals:
        ds[f"a0_{s}"] = (("record", "height"), a0[s], {"units": "s", "long_name": f"wavelet-variance peak scale of {s}'"})
        ds[f"D_{s}"] = (("record", "height"), D[s], {"units": "s", "long_name": f"duration scale D = a0 D_g of {s}' (CB 1993 I eq. 22)"})
        ds[f"n_events_{s}"] = (("record", "height"), n_ev[s], {"long_name": f"MHAT zero-crossings of {s}' at a0 (edges excluded)"})
        ds[f"mean_spacing_{s}"] = (("record", "height"), spacing[s], {"units": "s", "long_name": "mean interval between consecutive detections"})
        ds[f"event_time_{s}"] = (("record", "height", "event"), times[s], {"units": "s", "long_name": "detection time from window start (NaN padded)"})
        ds[f"slope_{s}"] = (("record", "height"), slope_code[s], {"long_name": "slope sign used: -1 negative, +1 positive, 0 both"})
        ds[f"W_{s}"] = (("record", "height", "scale"), W[s], {"long_name": f"wavelet variance W_1(a) of {s}' (CB 1993 I eq. 8)"})
    ds["wT"] = (("record", "height"), wT_all, {"long_name": "window covariance w'Ts' (sign drives slope='auto')"})
    dims3 = ("record", "height", "sr_lag")
    long = {"a": "linearized Van Atta ramp amplitude (sign carries the flux direction)",
            "period": "linearized ramp period l+s (Van Atta eq. 2.15)",
            "d": "two-lag ramp duration (Paw U et al. 2005 eq. 3a)",
            "s": "two-lag quiet gap (Paw U et al. 2005 eq. 5)",
            "a2": "two-lag P-corrected ramp amplitude (Paw U et al. 2005 eq. 4)",
            "S3_rate": "S^3(r)/r, the Chen et al. (1997) t_m diagnostic"}
    for s in sr_signals:
        base = "K" if s == "Ts" else "m s-1"
        units = {"a": base, "period": "s", "d": "s", "s": "s", "a2": base, "S3_rate": f"({base})^3 s-1"}
        for f in ("a", "period", "d", "s", "a2", "S3_rate"):
            ds[f"sr_{f}_{s}"] = (dims3, sr[s][f], {"units": units[f], "long_name": f"{long[f]} of {s}'"})
    if "Ts" in sr_signals:
        # kinematic SR flux F = alpha * a * z / (l+s); alpha per sr_alpha_mode (ruling 2026-08-23)
        z_h = np.asarray(hf.heights, dtype=float)[None, :, None]
        F0 = sr["Ts"]["a"] * z_h / sr["Ts"]["period"]
        alpha_arr = np.full((nr, nh, nl), np.nan)
        if rc.sr_alpha_mode == "fixed":
            alpha_arr[:] = rc.sr_alpha
        elif rc.sr_alpha_mode == "castellvi":
            ust = _ancillary(hf, "ustar", nr, nh)
            L_ob = _ancillary(hf, "L", nr, nh)
            for ih in range(nh):
                zz = float(hf.heights[ih])
                with np.errstate(divide="ignore", invalid="ignore"):
                    zeta = (zz - rc.sr_d) / L_ob[:, ih]
                for il in range(nl):
                    alpha_arr[:, ih, il] = alpha_castellvi(sr["Ts"]["period"][:, ih, il],
                                                           ust[:, ih], zeta, zz, rc.sr_d)
        elif rc.sr_alpha_mode == "fit":
            for il in range(nl):          # [SITE-TUNED] least squares through the origin vs w'Ts'
                x, y = F0[:, :, il].ravel(), wT_all.ravel()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 2 and float(x[m] @ x[m]) > 0.0:
                    alpha_arr[:, :, il] = float(x[m] @ y[m]) / float(x[m] @ x[m])
        else:
            raise ValueError(f"sr_alpha_mode {rc.sr_alpha_mode!r}; expected fixed, castellvi or fit")
        ds["sr_flux_Ts"] = (dims3, alpha_arr * F0,
                            {"units": "K m s^-1",
                             "long_name": "kinematic surface-renewal heat flux alpha a z/(l+s) (Spano 1997 eq. 2)"})
        ds["sr_alpha_Ts"] = (dims3, alpha_arr,
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
                    detrend_method=pc.detrend, signals=",".join(signals),
                    sr_signals=",".join(sr_signals), sr_alpha=rc.sr_alpha,
                    sr_alpha_mode=rc.sr_alpha_mode, sr_d=rc.sr_d,
                    sr_min_period_lags=rc.sr_min_period_lags,
                    tke_lp_s=rc.tke_lp_s, tke_a_s=rc.tke_a_s, tke_thresh=rc.tke_thresh,
                    library_note="library/writeups/ec_ramps.md")
    return ds
