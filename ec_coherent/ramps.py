"""Ramp / microfront detection -- wavelet path (Collineau & Brunet 1993 zero-crossing method).

Science and sources: library/writeups/ec_ramps.md. The structure-function
(Van Atta) detector and the TKE extension are not implemented yet.
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
    """``/ramps`` group: wavelet-detected microfronts of Ts and u per record and height."""
    rc, pc = cfg.ramps, cfg.preprocess
    signals = [s for s in rc.signals if s in hf.ds or s == "e"]
    signals = [s for s in signals if s != "e"]                 # TKE waits on Mangan 2022
    scales = log_scales(rc.a_min_s, rc.a_max_s, rc.n_scales_per_decade)
    nr, nh, ns, ne = len(hf.records), len(hf.heights), len(scales), rc.max_events
    a0 = {s: np.full((nr, nh), np.nan) for s in signals}
    D = {s: np.full((nr, nh), np.nan) for s in signals}
    n_ev = {s: np.zeros((nr, nh), dtype=int) for s in signals}
    spacing = {s: np.full((nr, nh), np.nan) for s in signals}
    times = {s: np.full((nr, nh, ne), np.nan) for s in signals}
    W = {s: np.full((nr, nh, ns), np.nan) for s in signals}
    slope_code = {s: np.full((nr, nh), np.nan) for s in signals}      # -1 negative, +1 positive, 0 both
    wT_all = np.full((nr, nh), np.nan)
    need = sorted(set(signals) | {"u", "v", "w", "Ts"})

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

    ds = xr.Dataset(coords={"record": ("record", hf.records), "height": ("height", hf.heights),
                            "scale": ("scale", scales), "event": ("event", np.arange(ne))})
    ds["scale"].attrs.update(units="s", long_name="wavelet dilation a (seconds)")
    for s in signals:
        ds[f"a0_{s}"] = (("record", "height"), a0[s], {"units": "s", "long_name": f"wavelet-variance peak scale of {s}'"})
        ds[f"D_{s}"] = (("record", "height"), D[s], {"units": "s", "long_name": f"duration scale D = a0 D_g of {s}' (CB 1993 I eq. 22)"})
        ds[f"n_events_{s}"] = (("record", "height"), n_ev[s], {"long_name": f"MHAT zero-crossings of {s}' at a0 (edges excluded)"})
        ds[f"mean_spacing_{s}"] = (("record", "height"), spacing[s], {"units": "s", "long_name": "mean interval between consecutive detections"})
        ds[f"event_time_{s}"] = (("record", "height", "event"), times[s], {"units": "s", "long_name": "detection time from window start (NaN padded)"})
        ds[f"slope_{s}"] = (("record", "height"), slope_code[s], {"long_name": "slope sign used: -1 negative, +1 positive, 0 both"})
        ds[f"W_{s}"] = (("record", "height", "scale"), W[s], {"long_name": f"wavelet variance W_1(a) of {s}' (CB 1993 I eq. 8)"})
    ds["wT"] = (("record", "height"), wT_all, {"long_name": "window covariance w'Ts' (sign drives slope='auto')"})
    ds.attrs.update(wavelet=rc.wavelet, D_g=float(D_G[rc.wavelet]), peak=rc.peak, refine=rc.refine,
                    D_min_s=rc.D_min_s,
                    edge_scales=rc.edge_scales, a_min_s=rc.a_min_s, a_max_s=rc.a_max_s,
                    detrend_method=pc.detrend, signals=",".join(signals),
                    library_note="library/writeups/ec_ramps.md")
    return ds
